from huggingface_hub import HfApi, hf_hub_download, list_models
from typing import List, Dict, Optional, Tuple, Any
import asyncio
import aiohttp
import json
import os
import threading
from tqdm import tqdm
import time
import re
import traceback
from datetime import datetime
from backend.logging_config import get_logger

try:
    # Optional import available in newer huggingface_hub versions
    from huggingface_hub import get_safetensors_metadata as hf_get_safetensors_metadata
except ImportError:  # pragma: no cover - fallback if function missing
    hf_get_safetensors_metadata = None

logger = get_logger(__name__)

# Initialize HF API - will be updated with token if provided
hf_api = HfApi()

# Check for environment variable on module initialization
_env_token = os.getenv('HUGGINGFACE_API_KEY')
if _env_token:
    hf_api = HfApi(token=_env_token)
    logger.info("HuggingFace API key loaded from environment variable")

# Simple cache for search results
_search_cache: Dict[str, Tuple[List[Dict], float]] = {}
_cache_timeout = 300  # 5 minutes

# Cache for safetensors metadata (per repo)
_safetensors_metadata_cache: Dict[str, Tuple[Dict, float]] = {}
_safetensors_metadata_ttl = 600  # 10 minutes

MODEL_FORMATS = ("gguf", "safetensors")


def _get_download_directory(model_format: str, huggingface_id: str) -> str:
    """Return the directory where files for the given format should be stored."""
    base_dir = os.path.join("data", "models")
    if model_format == "safetensors":
        safe_repo = huggingface_id.replace("/", "_")
        path = os.path.join(base_dir, "safetensors", safe_repo)
    else:
        path = base_dir
    os.makedirs(path, exist_ok=True)
    return path


SAFETENSORS_DIR = os.path.join("data", "models", "safetensors")
SAFETENSORS_MANIFEST = os.path.join(SAFETENSORS_DIR, "manifest.json")
_safetensors_manifest_lock = threading.Lock()
DEFAULT_LMDEPLOY_CONTEXT = 4096
MAX_LMDEPLOY_CONTEXT = 65536


def _load_safetensors_manifest() -> List[Dict]:
    os.makedirs(SAFETENSORS_DIR, exist_ok=True)
    if not os.path.exists(SAFETENSORS_MANIFEST):
        return []

    try:
        with open(SAFETENSORS_MANIFEST, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning(f"Failed to load safetensors manifest: {exc}")
        return []


def _save_safetensors_manifest(entries: List[Dict]):
    os.makedirs(SAFETENSORS_DIR, exist_ok=True)
    tmp_path = f"{SAFETENSORS_MANIFEST}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    os.replace(tmp_path, SAFETENSORS_MANIFEST)


def get_default_lmdeploy_config(max_context_length: Optional[int] = None) -> Dict[str, Any]:
    """Return default LMDeploy runtime configuration."""
    context_len = max_context_length or DEFAULT_LMDEPLOY_CONTEXT
    context_len = max(1024, min(context_len, MAX_LMDEPLOY_CONTEXT))
    return {
        "context_length": context_len,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "tensor_parallel": 1,
        "tensor_split": [],
        "max_batch_size": 4,
        "max_batch_tokens": context_len * 2,
        "kv_cache_percent": 1.0,
        "gpu_memory_utilization": 0.9,
        "use_streaming": True,
        "additional_args": "",
    }


def record_safetensors_download(
    huggingface_id: str,
    filename: str,
    file_path: str,
    file_size: int,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    tensor_summary: Optional[Dict[str, Any]] = None,
    lmdeploy_config: Optional[Dict[str, Any]] = None,
    model_id: Optional[int] = None,
):
    """Record safetensors download metadata in manifest."""
    metadata = metadata or {}
    tensor_summary = tensor_summary or {}
    lmdeploy_config = lmdeploy_config or get_default_lmdeploy_config(metadata.get("max_context_length"))
    entry = {
        "huggingface_id": huggingface_id,
        "filename": filename,
        "file_path": file_path,
        "file_size": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 2) if file_size else 0,
        "downloaded_at": datetime.utcnow().isoformat() + "Z",
        "model_id": model_id,
        "metadata": metadata,
        "tensor_summary": tensor_summary,
        "max_context_length": metadata.get("max_context_length"),
        "lmdeploy": {
            "config": lmdeploy_config,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        },
    }
    with _safetensors_manifest_lock:
        manifest = _load_safetensors_manifest()
        manifest = [e for e in manifest if not (e.get("huggingface_id") == huggingface_id and e.get("filename") == filename)]
        manifest.append(entry)
        _save_safetensors_manifest(manifest)
    return entry


def get_safetensors_manifest_entry(huggingface_id: str, filename: str) -> Optional[Dict[str, Any]]:
    """Return a manifest entry for the given safetensors file."""
    safe_filename = _sanitize_filename(filename)
    with _safetensors_manifest_lock:
        manifest = _load_safetensors_manifest()
        for entry in manifest:
            if entry.get("huggingface_id") == huggingface_id and entry.get("filename") == safe_filename:
                return entry
    return None


def update_lmdeploy_config(huggingface_id: str, filename: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Update the stored LMDeploy config for a safetensors entry."""
    safe_filename = _sanitize_filename(filename)
    with _safetensors_manifest_lock:
        manifest = _load_safetensors_manifest()
        updated_entry = None
        for entry in manifest:
            if entry.get("huggingface_id") == huggingface_id and entry.get("filename") == safe_filename:
                entry.setdefault("lmdeploy", {})
                entry["lmdeploy"]["config"] = config
                entry["lmdeploy"]["updated_at"] = datetime.utcnow().isoformat() + "Z"
                updated_entry = entry
                break
        if not updated_entry:
            raise ValueError(f"Safetensors manifest entry not found for {huggingface_id}/{safe_filename}")
        _save_safetensors_manifest(manifest)
        return updated_entry


def list_safetensors_downloads() -> List[Dict]:
    """Return safetensors downloads from manifest, pruning missing files."""
    with _safetensors_manifest_lock:
        manifest = _load_safetensors_manifest()
        updated_manifest = []
        result = []
        changed = False
        for entry in manifest:
            file_path = entry.get("file_path")
            if file_path and os.path.exists(file_path):
                size = os.path.getsize(file_path)
                entry["file_size"] = size
                entry["file_size_mb"] = round(size / (1024 * 1024), 2)
                result.append(entry)
                updated_manifest.append(entry)
            else:
                changed = True
                logger.debug(f"Pruning missing safetensors file: {file_path}")
        if changed:
            _save_safetensors_manifest(updated_manifest)
    return result


def delete_safetensors_download(huggingface_id: str, filename: str) -> Optional[Dict[str, Any]]:
    """Delete a safetensors file and remove it from manifest. Returns removed entry."""
    removed_entry: Optional[Dict[str, Any]] = None
    with _safetensors_manifest_lock:
        manifest = _load_safetensors_manifest()
        remaining = []
        for entry in manifest:
            if entry.get("huggingface_id") == huggingface_id and entry.get("filename") == filename:
                file_path = entry.get("file_path")
                try:
                    if file_path and os.path.exists(file_path):
                        os.remove(file_path)
                        parent_dir = os.path.dirname(file_path)
                        if os.path.isdir(parent_dir) and not os.listdir(parent_dir):
                            os.rmdir(parent_dir)
                    removed_entry = entry
                except Exception as exc:
                    logger.warning(f"Failed to delete safetensors file {file_path}: {exc}")
            else:
                remaining.append(entry)
        if removed_entry:
            _save_safetensors_manifest(remaining)
    return removed_entry

def clear_search_cache():
    """Clear the search cache to force fresh results"""
    global _search_cache
    _search_cache = {}
    global _safetensors_metadata_cache
    _safetensors_metadata_cache = {}

# Rate limiting
_last_request_time = 0
_min_request_interval = 0.5  # Reduced to 0.5 seconds since we're making fewer calls

def _sanitize_filename(filename: str) -> str:
    """Ensure filename is a safe relative path without traversal."""
    if not filename or filename.strip() == "":
        raise ValueError("filename is required")
    normalized = os.path.normpath(filename).replace("\\", "/")
    if normalized.startswith("../") or normalized.startswith("..\\") or normalized.startswith("/"):
        raise ValueError("invalid filename")
    parts = normalized.split("/")
    if any(part in ("", ".", "..") for part in parts):
        normalized = "/".join(part for part in parts if part not in ("", ".", ".."))
    if ".." in normalized.split("/"):
        raise ValueError("invalid filename")
    return normalized or os.path.basename(filename)

# Compiled regex patterns for better performance
QUANTIZATION_PATTERNS = [
    re.compile(r'IQ\d+_[A-Z]+'),  # IQ1_S, IQ2_M, etc.
    re.compile(r'Q\d+_K_[A-Z]+'),  # Q4_K_M, Q5_K_S, etc.
    re.compile(r'Q\d+_\d+'),      # Q4_0, Q5_1, etc.
    re.compile(r'Q\d+_K'),        # Q2_K, Q6_K, etc.
    re.compile(r'Q\d+'),          # Q3, Q4, etc. (fallback)
]

# Model size extraction pattern


def set_huggingface_token(token: str):
    """Set HuggingFace API token for authenticated requests"""
    global hf_api
    if token:
        hf_api = HfApi(token=token)
        logger.info("HuggingFace API token set - using authenticated requests")
    else:
        hf_api = HfApi()
        logger.info("HuggingFace API token cleared - using unauthenticated requests")


def get_huggingface_token() -> Optional[str]:
    """Get current HuggingFace API token"""
    return getattr(hf_api, 'token', None)


async def _rate_limit():
    """Async rate limiting to avoid hitting HuggingFace limits"""
    global _last_request_time
    current_time = time.time()
    time_since_last = current_time - _last_request_time
    if time_since_last < _min_request_interval:
        sleep_time = _min_request_interval - time_since_last
        logger.warning(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
        await asyncio.sleep(sleep_time)
    _last_request_time = time.time()


async def search_models(query: str, limit: int = 20, model_format: str = "gguf") -> List[Dict]:
    """Search HuggingFace for GGUF or safetensors models."""
    try:
        model_format = (model_format or "gguf").lower()
        if model_format not in MODEL_FORMATS:
            raise ValueError(f"Unsupported model format '{model_format}'. Must be one of {MODEL_FORMATS}")

        # Check cache first
        cache_key = f"{model_format}:{query.lower()}_{limit}"
        current_time = time.time()
        
        if cache_key in _search_cache:
            cached_data, cache_time = _search_cache[cache_key]
            if current_time - cache_time < _cache_timeout:
                logger.info(f"Returning cached results for '{query}'")
                return cached_data[:limit]  # Return only requested limit
        
        logger.info(f"Searching for models with query: '{query}', limit: {limit}, format: {model_format}")
        # Always attempt API search; authentication will be used automatically if a token is set
        return await _search_with_api(query, limit, model_format)
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise Exception(f"Failed to search models: {e}")


async def _search_with_api(query: str, limit: int, model_format: str) -> List[Dict]:
    """Search using HuggingFace Hub API (authenticated if token is configured)."""
    try:
        # Apply rate limiting
        await _rate_limit()
        
        # Use real HuggingFace API search with expand parameter for rich metadata
        filter_value = "gguf" if model_format == "gguf" else "safetensors"

        models_generator = list_models(
            search=query,
            limit=min(limit * 2, 50),  # Get more models to filter from
            sort="downloads",
            direction=-1,
            filter=filter_value,
            expand=["author", "cardData", "siblings"]  # Ensure author is present, plus metadata
        )
        
        # Convert generator to list
        models = list(models_generator)
        logger.info(f"Found {len(models)} models from HuggingFace API with expanded metadata")
        
        # Process models in parallel for better performance
        results = await _process_models_parallel(models, limit, model_format)
        
        # Cache the results
        cache_key = f"{model_format}:{query.lower()}_{limit}"
        _search_cache[cache_key] = (results, time.time())
        
        logger.info(f"Returning {len(results)} results from API")
        return results
        
    except Exception as e:
        logger.error(f"API search error: {e}")
        # Return empty results if API fails
        return []


async def _process_models_parallel(models: List, limit: int, model_format: str, max_concurrent: int = 5) -> List[Dict]:
    """Process models in parallel with semaphore for concurrency control"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_model(model):
        async with semaphore:
            return await _process_single_model(model, model_format)
    
    # Create tasks for all models
    tasks = [process_model(model) for model in models[:limit * 2]]
    
    # Execute in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions and None results
    valid_results = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Model processing error: {result}")
            continue
        if result is not None:
            valid_results.append(result)
    
    return valid_results[:limit]


async def _process_single_model(model, model_format: str) -> Optional[Dict]:
    """Process a single model and extract all metadata"""
    try:
        logger.info(f"Processing model: {model.id}")
        
        quantizations: Dict[str, Dict] = {}
        safetensors_files: List[Dict] = []
        repo_files: List[Dict[str, Any]] = []

        if hasattr(model, 'siblings') and model.siblings:
            if model_format == "gguf":
                gguf_files = [sibling.rfilename for sibling in model.siblings 
                              if sibling.rfilename.endswith('.gguf')]
                logger.info(f"Model {model.id}: {len(gguf_files)} GGUF files found")
                if not gguf_files:
                    return None
                for file in gguf_files:
                    quantization = _extract_quantization(file)
                    if quantization == "unknown":
                        continue
                    quantizations[quantization] = {
                        "filename": file
                    }
            else:
                safetensors_files = []
                for sibling in model.siblings:
                    filename = sibling.rfilename
                    size_bytes = getattr(sibling, 'size', 0) or 0
                    repo_files.append({
                        "filename": filename,
                        "is_safetensors": filename.endswith('.safetensors')
                    })
                    if not filename.endswith('.safetensors'):
                        continue
                    safetensors_files.append({
                        "filename": filename
                    })

                logger.info(f"Model {model.id}: {len(safetensors_files)} safetensors files found")
                if not safetensors_files:
                    return None
        else:
            return None

        # Extract rich metadata from model and cardData
        metadata = _extract_model_metadata(model)
        
        result = {
            "id": model.id,
            "name": getattr(model, 'modelId', model.id),  # Use modelId if available, fallback to id
            "author": getattr(model, 'author', ''),
            "downloads": model.downloads,
            "likes": getattr(model, 'likes', 0),
            "tags": model.tags or [],
            "model_format": model_format,
            "quantizations": quantizations if model_format == "gguf" else {},
            "safetensors_files": safetensors_files if model_format == "safetensors" else [],
            "repo_files": repo_files if model_format == "safetensors" else [],
            **metadata  # Include all extracted metadata
        }
        
        logger.info(f"Added model {model.id} to results")
        return result
        
    except Exception as e:
        logger.error(f"Error processing model {model.id}: {e}")
        return None


def _extract_model_metadata(model) -> Dict:
    """Extract rich metadata from ModelInfo and cardData"""
    metadata = {
        "description": "",
        "license": "",
        "pipeline_tag": getattr(model, 'pipeline_tag', ''),
        "library_name": getattr(model, 'library_name', ''),
        "language": [],
        "base_model": "",
        "architecture": "",
        "parameters": "",
        "context_length": None,
        "gated": getattr(model, 'gated', False),
        "private": getattr(model, 'private', False),
        "readme_url": f"https://huggingface.co/{model.id}",
        "created_at": getattr(model, 'createdAt', None),
        "updated_at": getattr(model, 'lastModified', None),
        "safetensors": {}
    }
    
    # Extract from cardData if available
    if hasattr(model, 'cardData') and model.cardData:
        card_data = model.cardData
        
        # Ensure card_data is not None and is a dict
        if card_data and isinstance(card_data, dict):
            # Extract basic info
            metadata["license"] = card_data.get('license', '')
            language_data = card_data.get('language', [])
            # Ensure language is always an array
            metadata["language"] = language_data if isinstance(language_data, list) else []
            metadata["base_model"] = card_data.get('base_model', '')
            
            # Extract from model_index if available
            model_index = card_data.get('model-index', [])
            if model_index:
                for item in model_index:
                    if isinstance(item, dict):
                        # Extract architecture
                        if 'name' in item:
                            metadata["architecture"] = item['name']
                        
                        # Extract parameters
                        if 'params' in item:
                            metadata["parameters"] = str(item['params'])
                        
                        # Extract context length
                        if 'context_length' in item:
                            metadata["context_length"] = item['context_length']
    
    # Extract model size from filename if not found in cardData
    if not metadata["parameters"]:
        # Try to extract model size from modelId using regex
        import re
        model_id = getattr(model, 'modelId', model.id)
        size_match = re.search(r'(\d+(?:\.\d+)?)[Bb]', model_id)
        if size_match:
            metadata["parameters"] = f"{size_match.group(1)}B"
    
    # Extract safetensors metadata from siblings
    if hasattr(model, 'siblings') and model.siblings:
        metadata["safetensors"] = _extract_safetensors_metadata(model.siblings)
    
    return metadata


def _extract_quantization(filename: str) -> str:
    """Extract quantization from filename using compiled regex patterns"""
    for pattern in QUANTIZATION_PATTERNS:
        match = pattern.search(filename)
        if match:
            return match.group()
    return "unknown"


def _extract_safetensors_metadata(siblings) -> Dict:
    """Extract safetensors metadata from siblings if available"""
    safetensors_info = {
        "has_safetensors": False,
        "safetensors_files": [],
        "total_tensors": 0,
        "total_size": 0
    }
    
    if not siblings:
        return safetensors_info
    
    safetensors_files = []
    total_size = 0
    
    for sibling in siblings:
        if sibling.rfilename.endswith('.safetensors'):
            safetensors_files.append({
                "filename": sibling.rfilename
            })
            total_size += sibling.size or 0
    
    if safetensors_files:
        safetensors_info.update({
            "has_safetensors": True,
            "safetensors_files": safetensors_files,
            "total_size": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        })
    
    return safetensors_info


async def get_safetensors_metadata_summary(model_id: str) -> Dict:
    """Fetch safetensors metadata on demand with caching and aggregation."""
    if not model_id:
        raise ValueError("model_id is required")
    
    cache_key = model_id
    current_time = time.time()
    cached_entry = _safetensors_metadata_cache.get(cache_key)
    if cached_entry:
        cached_data, cached_time = cached_entry
        if current_time - cached_time < _safetensors_metadata_ttl:
            return cached_data
    
    if not hf_get_safetensors_metadata and not hasattr(hf_api, "get_safetensors_metadata"):
        raise RuntimeError("Safetensors metadata is not supported by the installed huggingface_hub version")
    
    await _rate_limit()
    loop = asyncio.get_running_loop()
    
    def _fetch_metadata():
        if hasattr(hf_api, "get_safetensors_metadata"):
            return hf_api.get_safetensors_metadata(repo_id=model_id)
        return hf_get_safetensors_metadata(model_id)
    
    try:
        metadata = await loop.run_in_executor(None, _fetch_metadata)
    except Exception as err:
        logger.error(f"Failed to fetch safetensors metadata for {model_id}: {err}")
        raise
    
    files_summary = []
    dtype_totals: Dict[str, int] = {}
    total_tensors = 0
    
    files_metadata = getattr(metadata, "files_metadata", {}) or {}
    for filename, file_meta in files_metadata.items():
        tensors = getattr(file_meta, "tensors", {}) or {}
        parameter_count = getattr(file_meta, "parameter_count", {}) or {}
        
        tensor_details = []
        for tensor_name, tensor_info in tensors.items():
            tensor_details.append({
                "name": tensor_name,
                "dtype": getattr(tensor_info, "dtype", "unknown"),
                "shape": getattr(tensor_info, "shape", []),
            })
        
        dtype_counts = {}
        for dtype, count in parameter_count.items():
            dtype_counts[dtype] = count
            dtype_totals[dtype] = dtype_totals.get(dtype, 0) + count
        
        total_tensors += len(tensor_details)
        files_summary.append({
            "filename": filename,
            "tensor_count": len(tensor_details),
            "dtype_counts": dtype_counts,
            "tensors": tensor_details
        })
    
    summary = {
        "repo_id": model_id,
        "total_files": len(files_summary),
        "total_tensors": total_tensors,
        "dtype_totals": dtype_totals,
        "files": files_summary,
        "cached_at": datetime.utcnow().isoformat()
    }
    
    _safetensors_metadata_cache[cache_key] = (summary, current_time)
    return summary


async def get_model_details(model_id: str) -> Dict:
    """Get detailed model information including config and README"""
    try:
        # Get model info with expanded data
        model_info = hf_api.model_info(model_id, expand=["cardData", "siblings"])
        
        # Extract basic metadata
        metadata = _extract_model_metadata(model_info)
        
        # Add additional details
        details = {
            "id": model_info.id,
            "name": getattr(model_info, 'modelId', model_info.id),  # Use modelId if available, fallback to id
            "author": getattr(model_info, 'author', ''),
            "downloads": model_info.downloads,
            "likes": getattr(model_info, 'likes', 0),
            "tags": model_info.tags or [],
            **metadata
        }
        
        # Try to get config.json for architecture details
        try:
            config_files = [s for s in model_info.siblings if s.rfilename == 'config.json']
            if config_files:
                # Download and parse config.json
                config_path = hf_hub_download(
                    repo_id=model_id,
                    filename='config.json',
                    local_dir="data/temp",
                    local_dir_use_symlinks=False
                )
                
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Extract architecture details
                details["config"] = {
                    "architectures": config.get('architectures', []),
                    "model_type": config.get('model_type', ''),
                    "hidden_size": config.get('hidden_size'),
                    "num_attention_heads": config.get('num_attention_heads'),
                    "num_hidden_layers": config.get('num_hidden_layers'),
                    "vocab_size": config.get('vocab_size'),
                    "max_position_embeddings": config.get('max_position_embeddings')
                }
                
                # Clean up temp file
                os.remove(config_path)
                
        except Exception as e:
            logger.warning(f"Could not fetch config.json for {model_id}: {e}")
            details["config"] = {}
        
        return details
        
    except Exception as e:
        logger.error(f"Error getting model details for {model_id}: {e}")
        raise Exception(f"Failed to get model details: {e}")


async def download_model(huggingface_id: str, filename: str, model_format: str = "gguf") -> tuple[str, int]:
    """Download model from HuggingFace"""
    try:
        models_dir = _get_download_directory(model_format, huggingface_id)
        
        # Sanitize filename
        filename = _sanitize_filename(filename)

        # Download the file
        file_path = hf_hub_download(
            repo_id=huggingface_id,
            filename=filename,
            local_dir=models_dir,
            local_dir_use_symlinks=False
        )
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        return file_path, file_size
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


async def download_model_with_websocket_progress(
    huggingface_id: str,
    filename: str,
    websocket_manager,
    task_id: str,
    total_bytes: int = 0,
    model_format: str = "gguf"
):
    """Download model with WebSocket progress updates by tracking filesystem size"""
    import asyncio
    import time
    
    logger.info(f"=== DOWNLOAD PROGRESS START ===")
    logger.info(f"Download task: {task_id}")
    logger.info(f"HuggingFace ID: {huggingface_id}")
    logger.info(f"Filename: {filename}")
    logger.info(f"Total bytes from search: {total_bytes}")
    logger.info(f"WebSocket manager: {websocket_manager}")
    logger.info(f"Active connections: {len(websocket_manager.active_connections)}")
    
    try:
        models_dir = _get_download_directory(model_format, huggingface_id)
        
        # Sanitize filename and build path
        filename = _sanitize_filename(filename)
        file_path = os.path.join(models_dir, filename)
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        # Send initial progress
        logger.info(f"Sending initial progress message...")
        await websocket_manager.send_download_progress(
            task_id=task_id,
            progress=0,
            message=f"Starting download of {filename}",
            bytes_downloaded=0,
            total_bytes=total_bytes,
            speed_mbps=0,
            eta_seconds=0,
            filename=filename,
            model_format=model_format
        )
        logger.info(f"Initial progress message sent")
        
        # Get file size from HuggingFace API if not provided
        if total_bytes == 0:
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                file_info = api.repo_file_info(repo_id=huggingface_id, path=filename)
                total_bytes = file_info.size
                logger.info(f"Got file size from HuggingFace API: {total_bytes}")
            except Exception as e:
                logger.warning(f"Could not get file size from HuggingFace API: {e}")
                # If we can't get the size, we'll estimate it
                total_bytes = 0
        
        # Send total size update
        if total_bytes > 0:
            await websocket_manager.send_download_progress(
                task_id=task_id,
                progress=0,
                message=f"Downloading {filename}",
                bytes_downloaded=0,
                total_bytes=total_bytes,
                speed_mbps=0,
                eta_seconds=0,
                filename=filename,
                model_format=model_format
            )
        
        # Start the download with built-in progress tracking
        logger.info(f"🚀 Starting download with built-in progress tracking...")
        
        file_path, file_size = await download_with_progress_tracking(
            huggingface_id, filename, file_path, models_dir,
            websocket_manager, task_id, total_bytes, model_format
        )
        
        # Send final completion
        await websocket_manager.send_download_progress(
            task_id=task_id,
            progress=100,
            message=f"Download completed: {filename}",
            bytes_downloaded=file_size,
            total_bytes=file_size,
            speed_mbps=0,
            eta_seconds=0,
            filename=filename,
            model_format=model_format
        )
        
        return file_path, file_size
        
    except Exception as e:
        # Send error notification
        if websocket_manager and task_id:
            await websocket_manager.send_download_progress(
                task_id=task_id,
                progress=0,
                message=f"Download failed: {str(e)}",
                bytes_downloaded=0,
                total_bytes=0,
                speed_mbps=0,
                eta_seconds=0,
                filename=filename,
                model_format=model_format
            )
            await websocket_manager.send_notification(
                "error", "Download Failed", f"Failed to download {filename}: {str(e)}", task_id
            )
        raise


async def download_with_progress_tracking(
    huggingface_id: str,
    filename: str,
    file_path: str,
    models_dir: str,
    websocket_manager,
    task_id: str,
    total_bytes: int,
    model_format: str
):
    """Download the file using custom http_get method with progress tracking"""
    try:
        import aiofiles
        
        logger.info(f"📁 Starting download of {filename} ({total_bytes} bytes) [{model_format}]")
        
        # Use the standard HuggingFace resolve URL (this is the default/preferred method)
        safe_filename = _sanitize_filename(filename)
        download_url = f"https://huggingface.co/{huggingface_id}/resolve/main/{safe_filename}"
        actual_file_size = total_bytes  # Start with the provided size
        
        # Optionally get exact file size from HuggingFace API
        try:
            api = HfApi()
            file_info = api.repo_file_info(repo_id=huggingface_id, filename=safe_filename)
            if hasattr(file_info, 'size') and file_info.size:
                actual_file_size = file_info.size
                logger.info(f"📊 Got file size from HuggingFace API: {actual_file_size} bytes ({actual_file_size / (1024*1024):.2f} MB)")
        except Exception as e:
            logger.debug(f"Could not get file size from API: {e}, using provided size: {total_bytes}")
        
        logger.info(f"📁 Download URL: {download_url}")
        
        # Build headers manually
        hf_headers = {
            "User-Agent": "llama-cpp-studio/1.0.0",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate",
        }
        
        # Create final destination path
        final_path = os.path.join(models_dir, safe_filename)
        final_dir = os.path.dirname(final_path)
        if final_dir and not os.path.exists(final_dir):
            os.makedirs(final_dir, exist_ok=True)
        
        # Custom progress bar that sends WebSocket updates
        class WebSocketProgressBar(tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.websocket_manager = websocket_manager
                self.task_id = task_id
                self.filename = filename
                self.start_time = time.time()
                self.last_update_time = self.start_time
            
            def update(self, n=1):
                super().update(n)
                # Send WebSocket update with current progress
                current_time = time.time()
                if current_time - self.last_update_time >= 0.5:  # Update every 0.5 seconds
                    if self.total > 0:
                        progress = int((self.n / self.total) * 100)
                        current_bytes = int(self.n)
                        
                        # Calculate speed and ETA
                        elapsed_time = current_time - self.start_time
                        speed_bytes_per_sec = current_bytes / elapsed_time if elapsed_time > 0 else 0
                        speed_mbps = speed_bytes_per_sec / (1024 * 1024)
                        
                        remaining_bytes = self.total - self.n
                        eta_seconds = int(remaining_bytes / speed_bytes_per_sec) if speed_bytes_per_sec > 0 else 0
                        
                        logger.debug(f"📊 Progress: {progress}% ({current_bytes}/{self.total} bytes) - {speed_mbps:.1f} MB/s")
                        
                        # Send WebSocket update
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                asyncio.create_task(self.websocket_manager.send_download_progress(
                                    task_id=self.task_id,
                                    progress=progress,
                                    message=f"Downloading {self.filename}",
                                    bytes_downloaded=current_bytes,
                                    total_bytes=self.total,
                                    speed_mbps=speed_mbps,
                                    eta_seconds=eta_seconds,
                                    filename=self.filename,
                                    model_format=model_format
                                ))
                        except Exception as e:
                            logger.error(f"Error sending progress update: {e}")
                        
                        self.last_update_time = current_time
        
        # Create our custom progress bar
        custom_progress_bar = WebSocketProgressBar(
            desc=safe_filename,
            total=actual_file_size,  # Use the actual file size
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            disable=False
        )
        
        # Download using aiohttp with timeout and our custom progress bar
        timeout = aiohttp.ClientTimeout(total=3600, connect=30)  # 1 hour total, 30s connect
        async with aiohttp.ClientSession(headers=hf_headers, timeout=timeout) as session:
            async with session.get(download_url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download: HTTP {response.status}")
                
                # Get actual file size from response headers
                content_length = response.headers.get('content-length')
                if content_length:
                    response_size = int(content_length)
                    if response_size != actual_file_size:
                        logger.debug(f"📏 Size difference: API said {actual_file_size}, response says {response_size} (diff: {abs(response_size - actual_file_size)} bytes)")
                        # Use the response size as it's more accurate
                        actual_file_size = response_size
                        custom_progress_bar.total = actual_file_size
                        logger.info(f"📊 Using response size: {actual_file_size} bytes ({actual_file_size / (1024*1024):.2f} MB)")
                
                # Download with progress tracking
                # Use 64KB chunks for better performance with large files
                chunk_size = 65536
                downloaded_bytes = 0
                async with aiofiles.open(final_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(chunk_size):
                        await f.write(chunk)
                        downloaded_bytes += len(chunk)
                        custom_progress_bar.update(len(chunk))
        
        # Close the progress bar
        custom_progress_bar.close()
        
        logger.info(f"📁 Downloaded to: {final_path}")
        
        # Validate downloaded file size
        file_size = os.path.getsize(final_path)
        if actual_file_size and actual_file_size > 0 and file_size != actual_file_size:
            logger.warning(f"⚠️ Download size mismatch: expected {actual_file_size}, got {file_size}")
            # Allow small differences (like metadata)
            if abs(file_size - actual_file_size) > 1024:  # More than 1KB difference
                raise Exception(f"Download incomplete: expected {actual_file_size} bytes, got {file_size} bytes")
        
        return final_path, file_size
        
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise


async def get_quantization_sizes_from_hf(huggingface_id: str, quantizations: Dict[str, Dict]) -> Dict[str, Dict]:
    """Return actual file sizes for provided quantizations using Hugging Face Hub API.
    Uses the shared hf_api instance and mirrors logic used elsewhere in this module.
    """
    try:
        # Prefer fetching only required files to reduce payload
        filenames = [qd.get("filename") for qd in (quantizations or {}).values() if isinstance(qd, dict) and qd.get("filename")]
        updated: Dict[str, Dict] = {}

        if filenames:
            try:
                # Newer API: batch query specific paths for metadata
                paths_info = hf_api.get_paths_info(repo_id=huggingface_id, paths=filenames)
                # Build lookup
                file_sizes: Dict[str, Optional[int]] = {pi.path: getattr(pi, 'size', None) for pi in paths_info}
            except Exception as batch_err:
                logger.warning(f"get_paths_info failed for {huggingface_id}: {batch_err}")
                # Fallback: fetch full metadata once
                model_info = hf_api.model_info(repo_id=huggingface_id, files_metadata=True)
                file_sizes = {}
                if hasattr(model_info, 'siblings') and model_info.siblings:
                    for sibling in model_info.siblings:
                        file_sizes[sibling.rfilename] = getattr(sibling, 'size', None)

            for quant_name, quant_data in (quantizations or {}).items():
                filename = quant_data.get("filename") if isinstance(quant_data, dict) else None
                if not filename:
                    continue
                actual_size = file_sizes.get(filename)
                if not actual_size or actual_size <= 0:
                    try:
                        file_info = hf_api.repo_file_info(repo_id=huggingface_id, path=filename)
                        actual_size = getattr(file_info, 'size', None)
                    except Exception as file_err:
                        logger.warning(f"repo_file_info failed for {huggingface_id}/{filename}: {file_err}")
                        actual_size = None
                if actual_size and actual_size > 0:
                    updated[quant_name] = {
                        "filename": filename,
                        "size": actual_size,
                        "size_mb": round(actual_size / (1024 * 1024), 2)
                    }
                else:
                    logger.warning(f"Unable to determine size for {huggingface_id}/{filename}")

        return updated
    except Exception as e:
        logger.error(f"Failed to fetch quantization sizes for {huggingface_id}: {e}")
        return {}