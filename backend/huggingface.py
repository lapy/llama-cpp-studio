from huggingface_hub import HfApi, hf_hub_download, list_models
from typing import List, Dict, Optional
import asyncio
import aiohttp
import json
import os
from tqdm import tqdm
import time
import re
from datetime import datetime
from backend.logging_config import get_logger

logger = get_logger(__name__)

# Initialize HF API - will be updated with token if provided
hf_api = HfApi()

# Check for environment variable on module initialization
_env_token = os.getenv('HUGGINGFACE_API_KEY')
if _env_token:
    hf_api = HfApi(token=_env_token)
    logger.info("HuggingFace API key loaded from environment variable")

# Simple cache for search results
_search_cache = {}
_cache_timeout = 300  # 5 minutes

def clear_search_cache():
    """Clear the search cache to force fresh results"""
    global _search_cache
    _search_cache = {}

# Rate limiting
_last_request_time = 0
_min_request_interval = 0.5  # Reduced to 0.5 seconds since we're making fewer calls

# Compiled regex patterns for better performance
QUANTIZATION_PATTERNS = [
    re.compile(r'IQ\d+_[A-Z]+'),  # IQ1_S, IQ2_M, etc.
    re.compile(r'Q\d+_K_[SML]'),  # Q4_K_M, Q5_K_S, etc.
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


def _rate_limit():
    """Simple rate limiting to avoid hitting HuggingFace limits"""
    global _last_request_time
    current_time = time.time()
    time_since_last = current_time - _last_request_time
    if time_since_last < _min_request_interval:
        sleep_time = _min_request_interval - time_since_last
        logger.warning(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
        time.sleep(sleep_time)
    _last_request_time = time.time()


async def search_models(query: str, limit: int = 20) -> List[Dict]:
    """Search HuggingFace for GGUF models - uses real API if token available, otherwise empty results"""
    try:
        # Check cache first
        cache_key = f"{query.lower()}_{limit}"
        current_time = time.time()
        
        if cache_key in _search_cache:
            cached_data, cache_time = _search_cache[cache_key]
            if current_time - cache_time < _cache_timeout:
                logger.info(f"Returning cached results for '{query}'")
                return cached_data[:limit]  # Return only requested limit
        
        logger.info(f"Searching for models with query: '{query}', limit: {limit}")
        
        # Check if we have a HuggingFace token for real API access
        token = get_huggingface_token()
        if token:
            logger.info("Using HuggingFace API with authentication")
            return await _search_with_api(query, limit)
        else:
            logger.warning("No HuggingFace token - returning empty results")
            logger.warning("Please set a HuggingFace API token to search for models")
            return []
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise Exception(f"Failed to search models: {e}")


async def _search_with_api(query: str, limit: int) -> List[Dict]:
    """Search using real HuggingFace API with authentication and expand parameter"""
    try:
        # Apply rate limiting
        _rate_limit()
        
        # Use real HuggingFace API search with expand parameter for rich metadata
        models_generator = list_models(
            search=query,
            limit=min(limit * 2, 50),  # Get more models to filter from
            sort="downloads",
            direction=-1,
            filter="gguf",  # Filter for models that have GGUF files
            expand=["cardData", "siblings"]  # Get cardData and siblings in one call!
        )
        
        # Convert generator to list
        models = list(models_generator)
        logger.info(f"Found {len(models)} models from HuggingFace API with expanded metadata")
        
        # Process models in parallel for better performance
        results = await _process_models_parallel(models, limit)
        
        # Cache the results
        cache_key = f"{query.lower()}_{limit}"
        _search_cache[cache_key] = (results, time.time())
        
        logger.info(f"Returning {len(results)} results from API")
        return results
        
    except Exception as e:
        logger.error(f"API search error: {e}")
        # Return empty results if API fails
        return []


async def _process_models_parallel(models: List, limit: int, max_concurrent: int = 5) -> List[Dict]:
    """Process models in parallel with semaphore for concurrency control"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_model(model):
        async with semaphore:
            return await _process_single_model(model)
    
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


async def _process_single_model(model) -> Optional[Dict]:
    """Process a single model and extract all metadata"""
    try:
        logger.info(f"Processing model: {model.id}")
        
        # Extract GGUF files from siblings (no additional API call needed!)
        gguf_files = []
        if hasattr(model, 'siblings') and model.siblings:
            gguf_files = [sibling.rfilename for sibling in model.siblings 
                          if sibling.rfilename.endswith('.gguf')]
        
        logger.info(f"Model {model.id}: {len(gguf_files)} GGUF files found")
        
        if not gguf_files:
            return None
        
        # Extract quantizations with file sizes from siblings
        quantizations = {}
        files_to_process = gguf_files  # Process ALL GGUF files
        
        for file in files_to_process:
            quantization = _extract_quantization(file)
            if quantization == "unknown":
                continue
            
            # NO SIZE ESTIMATION - only store filename for API call later
            quantizations[quantization] = {
                "filename": file
                # NO size or size_mb fields - sizes will come from API call only
            }
            logger.info(f"Found quantization: {quantization} for file: {file} (no size estimation)")
        
        if not quantizations:
            return None
        
        # Extract rich metadata from model and cardData
        metadata = _extract_model_metadata(model)
        
        result = {
            "id": model.id,
            "name": getattr(model, 'modelId', model.id),  # Use modelId if available, fallback to id
            "author": model.author,
            "downloads": model.downloads,
            "likes": getattr(model, 'likes', 0),
            "tags": model.tags or [],
            "quantizations": quantizations,
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
                "filename": sibling.rfilename,
                "size": sibling.size or 0,
                "size_mb": round((sibling.size or 0) / (1024 * 1024), 2)
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
            "author": model_info.author,
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


async def download_model(huggingface_id: str, filename: str) -> tuple[str, int]:
    """Download model from HuggingFace"""
    try:
        # Create models directory
        models_dir = "data/models"
        os.makedirs(models_dir, exist_ok=True)
        
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


async def download_model_with_progress(huggingface_id: str, filename: str, progress_callback=None):
    """Download model with progress tracking"""
    try:
        models_dir = "data/models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Use huggingface_hub's download with progress
        file_path = hf_hub_download(
            repo_id=huggingface_id,
            filename=filename,
            local_dir=models_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        file_size = os.path.getsize(file_path)
        
        if progress_callback:
            progress_callback(100, file_size)
        
        return file_path, file_size
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


async def download_model_with_websocket_progress(huggingface_id: str, filename: str, 
                                               websocket_manager, task_id: str, total_bytes: int = 0):
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
        models_dir = "data/models"
        os.makedirs(models_dir, exist_ok=True)
        
        file_path = os.path.join(models_dir, filename)
        
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
            filename=filename
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
                filename=filename
            )
        
        # Start the download with built-in progress tracking
        logger.info(f"🚀 Starting download with built-in progress tracking...")
        
        file_path, file_size = await download_with_progress_tracking(
            huggingface_id, filename, file_path, models_dir,
            websocket_manager, task_id, total_bytes
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
            filename=filename
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
                filename=filename
            )
            await websocket_manager.send_notification(
                "error", "Download Failed", f"Failed to download {filename}: {str(e)}", task_id
            )
        raise


async def download_with_progress_tracking(huggingface_id: str, filename: str, file_path: str, 
                                        models_dir: str, websocket_manager, task_id: str, 
                                        total_bytes: int):
    """Download the file using custom http_get method with progress tracking"""
    try:
        import aiofiles
        
        logger.info(f"📁 Starting download of {filename} ({total_bytes} bytes)")
        
        # Get the download URL from HuggingFace API
        api = HfApi()
        actual_file_size = total_bytes  # Start with the provided size
        
        # Use the correct method for getting file info
        try:
            # Try the newer method first
            file_info = api.repo_file_info(repo_id=huggingface_id, filename=filename)
            download_url = file_info.download_url
            # Get actual file size from HuggingFace API
            if hasattr(file_info, 'size') and file_info.size:
                actual_file_size = file_info.size
                logger.info(f"📊 Got actual file size from HuggingFace API: {actual_file_size} bytes ({actual_file_size / (1024*1024):.2f} MB)")
        except AttributeError:
            # Fallback to older method
            download_url = f"https://huggingface.co/{huggingface_id}/resolve/main/{filename}"
            logger.info(f"�� Using fallback URL, keeping provided size: {total_bytes} bytes")
        except Exception as e:
            logger.warning(f"�� Error getting file info: {e}, using fallback URL")
            download_url = f"https://huggingface.co/{huggingface_id}/resolve/main/{filename}"
        
        logger.info(f"📁 Download URL: {download_url}")
        
        # Build headers manually
        hf_headers = {
            "User-Agent": "llama-cpp-studio/1.0.0",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate",
        }
        
        # Create final destination path
        final_path = os.path.join(models_dir, filename)
        
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
                                    filename=self.filename
                                ))
                        except Exception as e:
                            logger.error(f"Error sending progress update: {e}")
                        
                        self.last_update_time = current_time
        
        # Create our custom progress bar
        custom_progress_bar = WebSocketProgressBar(
            desc=filename,
            total=actual_file_size,  # Use the actual file size
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            disable=False
        )
        
        # Download using aiohttp with our custom progress bar
        async with aiohttp.ClientSession(headers=hf_headers) as session:
            async with session.get(download_url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download: HTTP {response.status}")
                
                # Get actual file size from response headers
                content_length = response.headers.get('content-length')
                if content_length:
                    response_size = int(content_length)
                    if response_size != actual_file_size:
                        logger.warning(f"�� Size mismatch: API said {actual_file_size}, response says {response_size}")
                        # Use the response size as it's more accurate
                        actual_file_size = response_size
                        custom_progress_bar.total = actual_file_size
                        logger.info(f"📊 Updated to response size: {actual_file_size} bytes ({actual_file_size / (1024*1024):.2f} MB)")
                
                # Download with progress tracking
                downloaded_bytes = 0
                async with aiofiles.open(final_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):  # 8KB chunks
                        await f.write(chunk)
                        downloaded_bytes += len(chunk)
                        custom_progress_bar.update(len(chunk))
        
        # Close the progress bar
        custom_progress_bar.close()
        
        logger.info(f"📁 Downloaded to: {final_path}")
        
        file_size = os.path.getsize(final_path)
        return final_path, file_size
        
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise