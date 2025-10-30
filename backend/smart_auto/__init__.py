import json
import math
import os
import psutil
from typing import Dict, Any, Optional
from backend.database import Model
from backend.gpu_detector import get_gpu_info
from backend.logging_config import get_logger

logger = get_logger(__name__)

# Module-level cache for GGUF layer info to avoid repeated disk parsing
LAYER_INFO_CACHE: Dict[str, Dict[str, Any]] = {}


class SmartAutoConfig:
    """Smart configuration optimizer for llama.cpp parameters"""
    
    def __init__(self):
        self.model_size_cache = {}
        self.current_preset = None
    
    @staticmethod
    def _parse_compute_capability(value: str) -> float:
        """Parse compute capability like '8.0', '7.5' to a float safely."""
        try:
            parts = str(value).split('.')
            major = int(parts[0]) if parts and parts[0].isdigit() else 0
            minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            return major + minor / 10.0
        except Exception:
            return 0.0
    
    async def generate_config(self, model: Model, gpu_info: Dict[str, Any], preset: Optional[str] = None, debug: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate optimal configuration based on model and GPU capabilities
        
        Args:
            model: The model to configure
            gpu_info: GPU information dictionary
            preset: Optional preset name (coding, conversational, long_context) to use as tuning parameters
        """
        try:
            config = {}
            # Store preset for later use in generation params
            self.current_preset = preset
            
            # Get model metadata
            model_size_mb = model.file_size / (1024 * 1024) if model.file_size else 0
            model_name = model.name.lower()
            
            # Get comprehensive model layer information from unified helper
            from .model_metadata import get_model_metadata
            layer_info = get_model_metadata(model)
            layer_count = layer_info.get('layer_count', 32)
            architecture = layer_info.get('architecture', 'unknown')
            context_length = layer_info.get('context_length', 0)  # Start with 0, we'll fix if needed
            vocab_size = layer_info.get('vocab_size', 0)
            embedding_length = layer_info.get('embedding_length', 0)
            attention_head_count = layer_info.get('attention_head_count', 0)
            attention_head_count_kv = layer_info.get('attention_head_count_kv', 0)
            is_moe = layer_info.get('is_moe', False)
            expert_count = layer_info.get('expert_count', 0)

            if debug is not None:
                debug.update({
                    "model_name": model.name,
                    "model_size_mb": model_size_mb,
                    "layer_info": layer_info,
                })
            
            # Fallback to name-based detection if GGUF metadata is incomplete
            if architecture == 'unknown':
                architecture = self._detect_architecture(model_name)
            
            # Fix context_length if it's 0 by using architecture fallback
            if context_length == 0:
                context_length = self._get_architecture_default_context(architecture)
                logger.info(f"Context length was 0, using architecture default: {context_length}")
            
            # Get GPU capabilities
            total_vram = gpu_info.get("total_vram", 0)
            gpu_count = gpu_info.get("device_count", 0)
            gpus = gpu_info.get("gpus", [])
            if debug is not None:
                debug.update({
                    "gpu_count": gpu_count,
                    "total_vram": total_vram,
                })
            
            # Calculate available VRAM for optimization decisions
            available_vram_gb = 0
            if gpus:
                available_vram_gb = sum(gpu.get("memory", {}).get("free", 0) for gpu in gpus) / (1024**3)
            
            # Determine if flash attention is available (compute capability >= 8.0)
            flash_attn_available = False
            if gpus:
                try:
                    flash_attn_available = all(self._parse_compute_capability(gpu.get("compute_capability", "0.0")) >= 8.0 for gpu in gpus)
                except Exception:
                    flash_attn_available = False
            
            if not gpus:
                # CPU-only configuration
                cpu_cfg = self._generate_cpu_config(model_size_mb, architecture, layer_count, context_length, vocab_size, embedding_length, attention_head_count, debug=debug)
                return cpu_cfg
            
            # GPU configuration
            config.update(self._generate_gpu_config(
                model_size_mb, architecture, gpus, total_vram, gpu_count, gpu_info.get("nvlink_topology", {}), layer_count, context_length, vocab_size, embedding_length, attention_head_count, debug=debug
            ))

            # Hybrid consideration: if VRAM is tight, keep KV cache partly on CPU
            try:
                from .memory_estimator import get_cpu_memory_gb, ctx_tokens_budget_greedy
                cpu_total, cpu_used, cpu_available = get_cpu_memory_gb()
                # If we have some CPU RAM headroom and low VRAM, prefer no_kv_offload False only when enough VRAM
                if available_vram_gb < (model_size_mb / 1024) * 1.2:
                    # Signal to avoid KV offload to VRAM when VRAM is tight
                    config["no_kv_offload"] = True
                else:
                    config.setdefault("no_kv_offload", False)
            except Exception:
                pass
            
            # Add KV cache quantization optimization
            from .kv_cache import get_optimal_kv_cache_quant
            kv_cache_config = get_optimal_kv_cache_quant(
                available_vram_gb, context_length, architecture, flash_attn_available
            )
            config.update(kv_cache_config)
            
            # Add MoE offloading pattern if MoE model
            if is_moe:
                layer_info_for_flags = {
                    'is_moe': is_moe,
                    'expert_count': expert_count,
                    'model_size_mb': model_size_mb,
                    'available_vram_gb': available_vram_gb
                }
                moe_config = self._get_architecture_specific_flags(architecture, layer_info_for_flags)
                config['moe_offload_pattern'] = 'custom' if moe_config.get('moe_offload_custom') else 'none'
                config['moe_offload_custom'] = moe_config.get('moe_offload_custom', '')
                if moe_config.get('jinja'):
                    config['jinja'] = True
            
            # Add generation parameters with preset tuning if provided
            preset_overrides = None
            if self.current_preset:
                # Start with shared source-of-truth generation params (temp/top_p/top_k/repeat_penalty)
                try:
                    from backend.presets import get_architecture_and_presets
                    _, presets_map = get_architecture_and_presets(model)
                    preset_vals = presets_map.get(self.current_preset) or {}
                except Exception:
                    preset_vals = {}

                # Merge in internal adjustments deltas
                delta_overrides = self._get_preset_overrides(architecture, self.current_preset)

                # Combine into one override map
                preset_overrides = {**preset_vals, **delta_overrides}

                # Apply broader tuning (batch/ctx/parallel factors)
                self._apply_preset_tuning(config, architecture, self.current_preset)
            
            # Use the computed ctx_size from GPU/CPU config when generating params
            effective_ctx = int(config.get("ctx_size", context_length) or context_length)
            if debug is not None:
                debug["effective_ctx_before_gen_params"] = effective_ctx
            from .generation_params import build_generation_params
            config.update(build_generation_params(architecture, effective_ctx, preset_overrides))
            
            # Add server parameters
            config.update(self._generate_server_params())
            
            # Final sanitation and clamping
            config = self._sanitize_config(config, gpu_count)
            
            return config
            
        except Exception as e:
            raise Exception(f"Failed to generate smart config: {e}")
    
    async def _get_model_layer_info(self, model: Model) -> Dict[str, Any]:
        """Get comprehensive model layer information from GGUF metadata"""
        try:
            if model.file_path and os.path.exists(model.file_path):
                # Use module-level cache with mtime-based invalidation
                mtime = os.path.getmtime(model.file_path)
                cache_key = model.file_path
                cached = LAYER_INFO_CACHE.get(cache_key)
                if cached and cached.get("mtime") == mtime:
                    return cached.get("data", {})
                from backend.gguf_reader import get_model_layer_info
                layer_info = get_model_layer_info(model.file_path)
                if layer_info:
                    LAYER_INFO_CACHE[cache_key] = {"mtime": mtime, "data": layer_info}
                    return layer_info
        except Exception as e:
            logger.warning(f"Failed to get layer info for model {model.id}: {e}")
        
        # Fallback to architecture-based estimation
        return {
            'layer_count': self._estimate_layer_count_from_name(model.name.lower()),
            'architecture': 'unknown',
            'context_length': 4096,
            'vocab_size': 0,
            'embedding_length': 0,
            'attention_head_count': 0,
            'attention_head_count_kv': 0,
            'block_count': 0
        }
    
    async def _get_model_layer_count(self, model: Model) -> int:
        """Get model layer count from GGUF metadata (legacy method)"""
        layer_info = await self._get_model_layer_info(model)
        return layer_info.get('layer_count', 32)
    
    def _get_model_layer_info_sync(self, model: Model) -> Dict[str, Any]:
        """Synchronous version of comprehensive model layer information retrieval"""
        try:
            if model.file_path and os.path.exists(model.file_path):
                mtime = os.path.getmtime(model.file_path)
                cache_key = model.file_path
                cached = LAYER_INFO_CACHE.get(cache_key)
                if cached and cached.get("mtime") == mtime:
                    return cached.get("data", {})
                from backend.gguf_reader import get_model_layer_info
                layer_info = get_model_layer_info(model.file_path)
                if layer_info:
                    LAYER_INFO_CACHE[cache_key] = {"mtime": mtime, "data": layer_info}
                    return layer_info
        except Exception as e:
            logger.warning(f"Failed to get layer info for model {model.id}: {e}")
        
        # Fallback to architecture-based estimation
        return {
            'layer_count': self._estimate_layer_count_from_name(model.name.lower()),
            'architecture': 'unknown',
            'context_length': 4096,
            'vocab_size': 0,
            'embedding_length': 0,
            'attention_head_count': 0,
            'attention_head_count_kv': 0,
            'block_count': 0
        }
    
    def _estimate_layer_count_from_name(self, model_name: str) -> int:
        """Estimate layer count from model name"""
        if "7b" in model_name or "7B" in model_name:
            return 32
        elif "3b" in model_name or "3B" in model_name:
            return 28
        elif "1b" in model_name or "1B" in model_name:
            return 22
        elif "13b" in model_name or "13B" in model_name:
            return 40
        elif "30b" in model_name or "30B" in model_name:
            return 60
        elif "65b" in model_name or "65B" in model_name:
            return 80
        else:
            return 32  # Default fallback
    
    def _get_architecture_default_context(self, architecture: str) -> int:
        """Get default context length for architecture"""
        defaults = {
            "llama2": 4096,
            "llama3": 8192,
            "llama": 4096,
            "codellama": 16384,
            "mistral": 32768,
            "phi": 2048,
            "glm": 8192,
            "glm4": 204800,  # 200K context for GLM-4.6
            "deepseek": 32768,
            "deepseek-v3": 32768,
            "qwen": 32768,
            "qwen2": 32768,
            "qwen3": 131072,  # 128K context for Qwen3
            "gemma": 8192,
            "gemma3": 8192,
            "generic": 4096
        }
        return defaults.get(architecture, 4096)
    
    def _detect_architecture(self, model_name: str) -> str:
        """Detect model architecture from name"""
        model_name_lower = model_name.lower()
        
        if "llama" in model_name_lower:
            if "codellama" in model_name_lower:
                return "codellama"
            elif "llama2" in model_name_lower or "llama-2" in model_name_lower:
                return "llama2"
            elif "llama3" in model_name_lower or "llama-3" in model_name_lower:
                return "llama3"
            else:
                return "llama"
        elif "mistral" in model_name_lower:
            return "mistral"
        elif "phi" in model_name_lower:
            return "phi"
        elif "glm" in model_name_lower or "chatglm" in model_name_lower:
            if "glm-4" in model_name_lower or "glm4" in model_name_lower:
                return "glm4"
            return "glm"
        elif "deepseek" in model_name_lower:
            if "v3" in model_name_lower or "v3.1" in model_name_lower:
                return "deepseek-v3"
            return "deepseek"
        elif "qwen" in model_name_lower:
            if "qwen3" in model_name_lower:
                return "qwen3"
            elif "qwen2" in model_name_lower:
                return "qwen2"
            return "qwen"
        elif "gemma" in model_name_lower:
            if "gemma3" in model_name_lower or "gemma-3" in model_name_lower:
                return "gemma3"
            return "gemma"
        else:
            return "generic"
    
    def _generate_cpu_config(self, model_size_mb: float, architecture: str, layer_count: int = 32, context_length: int = 4096, vocab_size: int = 0, embedding_length: int = 0, attention_head_count: int = 0, debug: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate CPU-only configuration optimized for available RAM"""
        # Get system memory info (from centralized helper)
        from .memory_estimator import get_cpu_memory_gb, tokens_per_gb_by_model_size, ctx_tokens_budget_greedy
        total_ram_gb, used_ram_gb, available_ram_gb = get_cpu_memory_gb()
        if debug is not None:
            debug.update({
                "cpu_total_ram_gb": total_ram_gb,
                "cpu_available_ram_gb": available_ram_gb,
            })
        
        # Estimate CPU threads (leave some cores free for system)
        cpu_count_phys = psutil.cpu_count(logical=False) or 1
        logical_cpu_count = psutil.cpu_count(logical=True) or cpu_count_phys
        threads = max(1, cpu_count_phys - 1)  # Leave 1 core for system
        threads_batch = max(1, min(threads, max(1, logical_cpu_count - 2)))  # Guard negatives
        
        # Calculate optimal context size based on model's max and available RAM (no hard cap)
        base_ctx = max(512, context_length or 4096)
        model_gb = max(0.001, model_size_mb / 1024.0)
        # Tokens per GB heuristic (centralized)
        tokens_per_gb = tokens_per_gb_by_model_size(model_gb)
        # Reserve RAM for model + overhead using actual available RAM
        reserved_ram_gb = model_gb + 2.0
        available_for_ctx_gb = max(0.0, available_ram_gb - reserved_ram_gb)
        # Provide a small minimum window so we don't quantize to zero
        if available_for_ctx_gb <= 0:
            available_for_ctx_gb = max(0.25, available_ram_gb * 0.1)
        if debug is not None:
            debug.update({
                "model_gb": model_gb,
                "tokens_per_gb": tokens_per_gb,
                "reserved_ram_gb": reserved_ram_gb,
                "available_for_ctx_gb": available_for_ctx_gb,
            })
        # Initial cap ignoring batch/parallel
        max_tokens_by_ram = ctx_tokens_budget_greedy(model_gb, available_ram_gb, reserve_overhead_gb=2.0)
        optimal_ctx_size = max(512, min(base_ctx, max_tokens_by_ram))
        
        # Calculate optimal batch sizes based on model parameters
        if vocab_size > 0 and embedding_length > 0:
            # More sophisticated batch sizing based on model architecture
            base_batch_size = max(128, min(512, int(available_ram_gb * 50)))  # 50MB per GB RAM
            if attention_head_count > 0:
                # Adjust batch size based on attention complexity
                batch_size = max(64, min(base_batch_size, int(base_batch_size * (32 / attention_head_count))))
            else:
                batch_size = base_batch_size
        else:
            # Fallback to size-based estimation
            batch_size = max(128, min(512, int(available_ram_gb * 50)))
        
        # Adjust ctx_size to account for batch and parallel (ctx * batch * parallel <= tokens_budget)
        parallel = 1
        tokens_budget = int(tokens_per_gb * available_for_ctx_gb)
        if tokens_budget > 0:
            # Budget ctx tokens directly from available RAM; batch is handled separately
            safe_ctx = int(tokens_budget)
            optimal_ctx_size = max(512, min(optimal_ctx_size, safe_ctx))
        if debug is not None:
            debug.update({
                "tokens_budget": tokens_budget,
                "batch_size": batch_size,
                "parallel": parallel,
                "optimal_ctx_size": optimal_ctx_size,
            })

        config = {
            "threads": threads,
            "threads_batch": threads_batch,
            "ctx_size": optimal_ctx_size,
            "batch_size": batch_size,
            "ubatch_size": max(16, batch_size // 2),
            "parallel": parallel,
            "no_mmap": False,
            "mlock": False,
            "low_vram": False,
            "logits_all": False,  # Don't compute all logits to save memory
        }
        
        # Add architecture-specific optimizations
        config.update(self._get_cpu_architecture_optimizations(architecture, available_ram_gb))
        
        return config
    
    def _generate_gpu_config(self, model_size_mb: float, architecture: str, 
                           gpus: list, total_vram: int, gpu_count: int, nvlink_topology: Dict, layer_count: int = 32, context_length: int = 4096, vocab_size: int = 0, embedding_length: int = 0, attention_head_count: int = 0, debug: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate GPU-optimized configuration"""
        config = {}
        
        # Calculate optimal GPU layers
        if gpu_count == 1:
            config.update(self._single_gpu_config(model_size_mb, architecture, gpus[0], layer_count, embedding_length, attention_head_count))
        else:
            config.update(self._multi_gpu_config(model_size_mb, architecture, gpus, nvlink_topology, layer_count))
        
        # Context size based on available VRAM and model parameters
        available_vram = sum(gpu.get("memory", {}).get("free", 0) for gpu in gpus)
        ctx_size = self._get_optimal_context_size(architecture, available_vram, model_size_mb, layer_count, embedding_length, attention_head_count, 0)
        # Clamp GPU ctx size to sane bounds
        config["ctx_size"] = max(512, min(ctx_size, 262144))
        if debug is not None:
            debug.update({
                "gpu_available_vram_bytes": int(available_vram),
                "gpu_ctx_size": config["ctx_size"],
            })
        
        # Batch sizes based on actual memory requirements
        if embedding_length > 0 and layer_count > 0:
            optimal_batch_size = self._calculate_optimal_batch_size(available_vram / (1024**3), model_size_mb, config["ctx_size"], embedding_length, layer_count)
            config["batch_size"] = max(1, min(4096, optimal_batch_size))
            config["ubatch_size"] = max(1, min(config["batch_size"], max(1, config["batch_size"] // 2)))
        else:
            # Fallback to size-based estimation
            config["batch_size"] = min(1024, max(64, int(model_size_mb / 50)))
            config["ubatch_size"] = min(config["batch_size"], max(16, int(model_size_mb / 100)))
        
        # Parallel sequences (conservative for multi-GPU)
        if gpu_count > 1:
            config["parallel"] = max(1, min(4, gpu_count))
        else:
            config["parallel"] = 1
        
        return config
    
    def _single_gpu_config(self, model_size_mb: float, architecture: str, gpu: Dict, layer_count: int = 32, embedding_length: int = 0, attention_head_count: int = 0) -> Dict[str, Any]:
        """Configuration for single GPU"""
        vram_gb = gpu.get("memory", {}).get("total", 0) / (1024**3)
        free_vram_gb = gpu.get("memory", {}).get("free", 0) / (1024**3)
        
        # Estimate layers that fit in VRAM
        # Rough estimation: each layer takes ~1-2GB depending on model size
        estimated_layers_per_gb = 8 if model_size_mb < 1000 else 4  # Smaller models have more layers per GB
        
        max_layers = int(free_vram_gb * estimated_layers_per_gb * 0.8)  # Leave 20% buffer
        
        # Use actual layer count from model metadata
        total_layers = layer_count
        
        n_gpu_layers = min(max_layers, total_layers)
        
        config = {
            "n_gpu_layers": n_gpu_layers,
            "main_gpu": gpu.get("index", 0),
            "threads": max(1, (psutil.cpu_count(logical=False) or 2) - 2),
            "threads_batch": max(1, (psutil.cpu_count(logical=False) or 2) - 2)
        }
        
        # Calculate optimal batch sizes based on actual memory requirements
        if embedding_length > 0 and layer_count > 0:
            # Use data-driven calculation
            optimal_batch_size = self._calculate_optimal_batch_size(free_vram_gb, model_size_mb, 4096, embedding_length, layer_count)
            config["batch_size"] = max(1, min(4096, optimal_batch_size))
            config["ubatch_size"] = max(1, min(config["batch_size"], max(1, config["batch_size"] // 2)))
        else:
            # Fallback to VRAM-based estimation
            if vram_gb >= 24:    # High-end GPU
                config["batch_size"] = min(1024, max(256, int(vram_gb * 30)))
                config["ubatch_size"] = min(512, max(128, int(vram_gb * 15)))
            elif vram_gb >= 12:   # Mid-range GPU
                config["batch_size"] = min(512, max(128, int(vram_gb * 25)))
                config["ubatch_size"] = min(256, max(64, int(vram_gb * 12)))
            elif vram_gb >= 8:    # Lower-end GPU
                config["batch_size"] = min(256, max(64, int(vram_gb * 20)))
                config["ubatch_size"] = min(128, max(32, int(vram_gb * 10)))
            else:                 # Very limited VRAM
                config["batch_size"] = min(128, max(32, int(vram_gb * 15)))
                config["ubatch_size"] = min(64, max(16, int(vram_gb * 7)))
        
        # Enable flash attention for supported GPUs
        if self._parse_compute_capability(gpu.get("compute_capability", "0.0")) >= 8.0:  # Ampere and newer
            config["flash_attn"] = True
        
        return config
    
    def _multi_gpu_config(self, model_size_mb: float, architecture: str, gpus: list, nvlink_topology: Dict, layer_count: int = 32) -> Dict[str, Any]:
        """Configuration for multiple GPUs with NVLink awareness"""
        config = {
            "main_gpu": 0,
            "n_gpu_layers": -1,  # Use all layers
            "threads": max(1, psutil.cpu_count(logical=False) - 2),
            "threads_batch": max(1, psutil.cpu_count(logical=False) - 2)
        }
        
        # Enable flash attention if all GPUs support it
        if all(self._parse_compute_capability(gpu.get("compute_capability", "0.0")) >= 8.0 for gpu in gpus):
            config["flash_attn"] = True
        
        # Configure based on NVLink topology
        strategy = nvlink_topology.get("recommended_strategy", "pcie_only")
        
        if strategy == "nvlink_unified":
            # All GPUs connected via NVLink - use unified memory approach
            config.update(self._nvlink_unified_config(gpus, nvlink_topology))
        elif strategy == "nvlink_clustered":
            # Multiple NVLink clusters - optimize per cluster
            config.update(self._nvlink_clustered_config(gpus, nvlink_topology))
        elif strategy == "nvlink_partial":
            # Partial NVLink connectivity - hybrid approach
            config.update(self._nvlink_partial_config(gpus, nvlink_topology))
        else:
            # PCIe only - traditional tensor splitting
            config.update(self._pcie_only_config(gpus))
        
        return config
    
    def _nvlink_unified_config(self, gpus: list, nvlink_topology: Dict) -> Dict[str, Any]:
        """Configuration for unified NVLink cluster"""
        # With NVLink, we can use more aggressive tensor splitting
        vram_sizes = [gpu.get("memory", {}).get("total", 0) for gpu in gpus]
        total_vram = sum(vram_sizes)
        
        # Create optimized tensor split ratios
        tensor_split = []
        for vram in vram_sizes:
            ratio = vram / total_vram
            tensor_split.append(f"{ratio:.3f}")  # More precision for NVLink
        
        return {
            "tensor_split": ",".join(tensor_split),
            "parallel": min(8, len(gpus) * 2),  # Higher parallelism with NVLink
            "batch_size": min(4096, max(512, int(total_vram / (1024**3) * 150))),  # Larger batches for high VRAM
            "ubatch_size": min(2048, max(256, int(total_vram / (1024**3) * 75)))
        }
    
    def _nvlink_clustered_config(self, gpus: list, nvlink_topology: Dict) -> Dict[str, Any]:
        """Configuration for multiple NVLink clusters"""
        clusters = nvlink_topology.get("clusters", [])
        
        if not clusters:
            return self._pcie_only_config(gpus)
        
        # Use the largest cluster for primary processing
        largest_cluster = max(clusters, key=lambda c: len(c["gpus"]))
        cluster_gpus = [gpus[i] for i in largest_cluster["gpus"]]
        
        # Configure tensor split for the largest cluster
        vram_sizes = [gpu.get("memory", {}).get("total", 0) for gpu in cluster_gpus]
        total_vram = sum(vram_sizes)
        
        tensor_split = []
        for i, gpu in enumerate(gpus):
            if i in largest_cluster["gpus"]:
                ratio = gpu.get("memory", {}).get("total", 0) / total_vram
                tensor_split.append(f"{ratio:.3f}")
            else:
                tensor_split.append("0.0")  # Exclude GPUs not in main cluster
        
        return {
            "tensor_split": ",".join(tensor_split),
            "parallel": min(6, len(largest_cluster["gpus"]) * 2),
            "batch_size": min(3072, max(384, int(total_vram / (1024**3) * 120))),
            "ubatch_size": min(1536, max(192, int(total_vram / (1024**3) * 60)))
        }
    
    def _nvlink_partial_config(self, gpus: list, nvlink_topology: Dict) -> Dict[str, Any]:
        """Configuration for partial NVLink connectivity"""
        # Use conservative approach for partial NVLink
        vram_sizes = [gpu.get("memory", {}).get("total", 0) for gpu in gpus]
        total_vram = sum(vram_sizes)
        
        # Create balanced tensor split
        tensor_split = []
        for vram in vram_sizes:
            ratio = vram / total_vram
            tensor_split.append(f"{ratio:.2f}")
        
        return {
            "tensor_split": ",".join(tensor_split),
            "parallel": min(4, len(gpus)),
            "batch_size": min(2048, max(256, int(total_vram / (1024**3) * 100))),
            "ubatch_size": min(1024, max(128, int(total_vram / (1024**3) * 50)))
        }
    
    def _pcie_only_config(self, gpus: list) -> Dict[str, Any]:
        """Configuration for PCIe-only multi-GPU setup"""
        # Calculate tensor split based on VRAM
        vram_sizes = [gpu.get("memory", {}).get("total", 0) for gpu in gpus]
        total_vram = sum(vram_sizes)
        
        # Create tensor split ratios
        tensor_split = []
        for vram in vram_sizes:
            ratio = vram / total_vram
            tensor_split.append(f"{ratio:.2f}")
        
        return {
            "tensor_split": ",".join(tensor_split),
            "parallel": min(2, len(gpus)),  # Conservative parallelism for PCIe
            "batch_size": min(1024, max(128, int(total_vram / (1024**3) * 80))),
            "ubatch_size": min(512, max(64, int(total_vram / (1024**3) * 40)))
        }
    
    def _get_optimal_context_size(self, architecture: str, available_vram: int, model_size_mb: float = 0, layer_count: int = 32, embedding_length: int = 0, attention_head_count: int = 0, attention_head_count_kv: int = 0) -> int:
        """Calculate optimal context size based on actual memory requirements and architecture"""
        
        # Base context sizes by architecture (from model specifications)
        base_contexts = {
            "llama2": 4096,
            "llama3": 8192,
            "mistral": 32768,
            "codellama": 16384,
            "qwen3": 131072,  # 128K context
            "qwen": 32768,    # 32K context
            "qwen2": 32768,   # 32K context
            "phi": 2048,
            "generic": 4096
        }
        
        base_context = base_contexts.get(architecture, 4096)
        
        if available_vram == 0:
            # CPU mode - conservative context
            return max(512, min(base_context, 2048))
        
        # Use data-driven calculation if we have model parameters
        if model_size_mb > 0 and layer_count > 0 and embedding_length > 0:
            vram_gb = available_vram / (1024**3)
            calculated_max = self._calculate_max_context_size(vram_gb, model_size_mb, layer_count, embedding_length, attention_head_count, attention_head_count_kv)
            result = min(base_context, calculated_max) if calculated_max > 0 else base_context
            return max(512, min(result, 262144))  # Clamp
        
        # Fallback to architecture-based limits if no model data
        vram_gb = available_vram / (1024**3)
        
        # Conservative scaling based on VRAM capacity
        if vram_gb >= 24:    # High-end GPU
            return max(512, min(base_context, 262144))
        elif vram_gb >= 12:   # Mid-range GPU
            return max(512, min(base_context, int(base_context * 0.75)))
        elif vram_gb >= 8:    # Lower-end GPU
            return max(512, min(base_context, int(base_context * 0.5)))
        else:                 # Very limited VRAM
            return max(512, min(base_context, 2048))
    
    def _calculate_max_context_size(self, available_vram_gb: float, model_size_mb: float, layer_count: int, embedding_length: int, attention_head_count: int, attention_head_count_kv: int) -> int:
        """Calculate maximum context size based on actual memory requirements"""
        
        # Reserve memory for model
        model_memory_gb = model_size_mb / 1024
        reserved_memory_gb = model_memory_gb + 1.0  # Model + 1GB overhead
        available_for_context_gb = max(0, available_vram_gb - reserved_memory_gb)
        
        if available_for_context_gb <= 0:
            return 512  # Minimum context
        
        # Calculate KV cache memory per token based on transformer architecture
        if embedding_length > 0 and layer_count > 0:
            # KV cache = 2 * layers * hidden_size * 2 bytes (fp16)
            kv_cache_per_token_bytes = 2 * layer_count * embedding_length * 2
            
            # Adjust for Grouped Query Attention (GQA)
            if attention_head_count_kv > 0 and attention_head_count > 0:
                kv_cache_per_token_bytes *= (attention_head_count_kv / attention_head_count)
            
            # Convert to tokens per GB
            tokens_per_gb = (1024**3) / kv_cache_per_token_bytes
            
            # Calculate max context size with safety margin
            max_context_tokens = int(available_for_context_gb * tokens_per_gb * 0.8)  # 80% safety margin
            
            # Ensure minimum context size
            return max(512, max_context_tokens)
        else:
            # Fallback to conservative estimate: ~1000 tokens per GB
            estimated = int(available_for_context_gb * 1000)
            return max(512, min(4096, estimated))  # Ensure at least 512 tokens
    
    def _calculate_optimal_batch_size(self, available_vram_gb: float, model_size_mb: float, context_size: int, embedding_length: int, layer_count: int) -> int:
        """Calculate optimal batch size based on memory and throughput analysis"""
        
        # Memory requirements per batch item
        model_memory_gb = model_size_mb / 1024
        
        # KV cache memory per batch item
        if embedding_length > 0 and layer_count > 0:
            # KV cache = 2 * layers * hidden_size * context_length * 2 bytes (fp16)
            kv_cache_per_item_gb = (context_size * embedding_length * layer_count * 2 * 2) / (1024**3)
        else:
            # Conservative estimate: 64 bytes per token
            kv_cache_per_item_gb = context_size * 64 / (1024**3)
        
        total_per_item_gb = model_memory_gb + kv_cache_per_item_gb
        
        if total_per_item_gb <= 0:
            return 1
        
        # Calculate max batch size based on available memory
        max_batch_size = int(available_vram_gb * 0.7 / total_per_item_gb)  # 70% safety margin for fragmentation
        
        # Apply reasonable limits based on architecture
        if embedding_length > 2048:  # Large models (7B+)
            max_batch_size = min(max_batch_size, 512)
        elif embedding_length > 1024:  # Medium models (3B-7B)
            max_batch_size = min(max_batch_size, 1024)
        else:  # Small models (<3B)
            max_batch_size = min(max_batch_size, 2048)
        
        return max(1, max_batch_size)
    
    def _get_optimal_cpu_context_size(self, architecture: str, available_ram_gb: float, model_size_mb: float) -> int:
        """Calculate optimal context size for CPU-only mode based on available RAM"""
        # Base context sizes by architecture
        base_contexts = {
            "llama2": 4096,
            "llama3": 8192,
            "mistral": 32768,
            "codellama": 16384,
            "phi": 2048,
            "generic": 4096
        }
        
        base_context = base_contexts.get(architecture, 4096)
        
        # Calculate how much RAM we can allocate for context
        # Reserve space for model + overhead
        model_ram_gb = model_size_mb / 1024
        reserved_ram_gb = model_ram_gb + 2.0  # Model + 2GB overhead
        available_for_context = max(0, available_ram_gb - reserved_ram_gb)
        
        # Estimate context memory usage (rough: 1MB per 1000 tokens)
        max_context_tokens = int(available_for_context * 1000)  # 1GB = ~1000 tokens
        
        # Apply architecture-specific limits
        if architecture == "mistral":
            # Mistral can handle very large contexts
            optimal_context = min(base_context, max_context_tokens)
        elif architecture in ["llama3", "codellama"]:
            # Llama3 and CodeLlama have good context handling
            optimal_context = min(base_context, max_context_tokens)
        else:
            # Conservative for other architectures
            optimal_context = min(base_context, max_context_tokens, 8192)
        
        # Ensure minimum context size
        return max(512, optimal_context)
    
    def _calculate_optimal_batch_sizes(self, available_ram_gb: float, model_size_mb: float, 
                                      ctx_size: int, architecture: str) -> tuple[int, int]:
        """Calculate optimal batch sizes for CPU mode"""
        model_ram_gb = model_size_mb / 1024
        
        # Calculate available RAM for batching after model and context
        reserved_ram_gb = model_ram_gb + (ctx_size / 1000) + 1.0  # Model + context + overhead
        available_for_batch = max(0, available_ram_gb - reserved_ram_gb)
        
        # Estimate batch memory usage (rough: 1MB per batch item)
        max_batch_size = int(available_for_batch * 1000)  # 1GB = ~1000 batch items
        
        # Apply reasonable limits based on architecture
        if architecture == "mistral":
            # Mistral can handle larger batches
            batch_size = min(2048, max(64, max_batch_size))
            ubatch_size = min(1024, max(32, batch_size // 2))
        elif architecture in ["llama3", "codellama"]:
            # Good batch handling
            batch_size = min(1536, max(64, max_batch_size))
            ubatch_size = min(768, max(32, batch_size // 2))
        else:
            # Conservative for other architectures
            batch_size = min(1024, max(32, max_batch_size))
            ubatch_size = min(512, max(16, batch_size // 2))
        
        return batch_size, ubatch_size
    
    def _get_optimal_parallel_cpu(self, available_ram_gb: float, model_size_mb: float) -> int:
        """Calculate optimal parallel sequences for CPU mode"""
        model_ram_gb = model_size_mb / 1024
        
        # Calculate how many parallel sequences we can run
        # Each parallel sequence needs roughly 1GB of RAM
        max_parallel = int(available_ram_gb / (model_ram_gb + 1.0))
        
        # Apply reasonable limits
        if available_ram_gb >= 32:  # High RAM system
            return min(8, max(1, max_parallel))
        elif available_ram_gb >= 16:  # Mid RAM system
            return min(4, max(1, max_parallel))
        else:  # Low RAM system
            return min(2, max(1, max_parallel))
    
    def _get_cpu_architecture_optimizations(self, architecture: str, available_ram_gb: float) -> Dict[str, Any]:
        """Get architecture-specific optimizations for CPU mode"""
        optimizations = {}
        
        if architecture == "mistral":
            # Mistral optimizations
            optimizations.update({
                "rope_freq_base": 10000.0,
                "rope_freq_scale": 1.0,
                "use_mmap": True,  # Mistral benefits from mmap
            })
        elif architecture in ["llama3", "llama2"]:
            # Llama optimizations
            optimizations.update({
                "rope_freq_base": 10000.0,
                "rope_freq_scale": 1.0,
                "use_mmap": available_ram_gb < 16,  # Use mmap on lower RAM systems
            })
        elif architecture == "codellama":
            # CodeLlama optimizations
            optimizations.update({
                "rope_freq_base": 10000.0,
                "rope_freq_scale": 1.0,
                "use_mmap": True,  # CodeLlama benefits from mmap
                "logits_all": False,  # Don't compute all logits for code completion
            })
        elif architecture == "phi":
            # Phi optimizations
            optimizations.update({
                "rope_freq_base": 10000.0,
                "rope_freq_scale": 1.0,
                "use_mmap": True,
            })
        
        # Common CPU optimizations
        optimizations.update({
            "embedding": False,  # Disable embedding mode for inference
            "cont_batching": True,  # Enable continuous batching for efficiency
            "no_kv_offload": True,  # Don't offload KV cache (CPU mode)
        })
        
        return optimizations
    
    def _generate_generation_params(self, architecture: str, context_length: int = 4096, preset_overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive generation parameters tuned for the architecture and preset."""
        params: Dict[str, Any] = {
            # Core sampling controls
            "n_predict": -1,
            "temperature": 0.8,
            "temp": 0.8,  # alias retained for compatibility
            "top_k": 40,
            "top_p": 0.9,
            "min_p": 0.0,
            "typical_p": 1.0,
            "tfs_z": 1.0,
            "repeat_penalty": 1.1,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            # Mirostat
            "mirostat": 0,
            "mirostat_tau": 5.0,
            "mirostat_eta": 0.1,
            # Context and stopping
            "ctx_size": context_length,
            "stop": [],
            "seed": -1,
            # Decoding/format helpers
            "grammar": "",
            "json_schema": "",
            # Tool-calling / format hints
            "jinja": False,
        }

        # Architecture-specific tuning
        arch = (architecture or "").lower()
        if arch in ["llama3", "llama2", "llama", "codellama"]:
            params.update({
                "temperature": 0.7,
                "temp": 0.7,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.12,
                "typical_p": 1.0,
            })
        elif arch == "mistral":
            params.update({
                "temperature": 0.7,
                "temp": 0.7,
                "top_k": 50,
                "top_p": 0.95,
                "repeat_penalty": 1.1,
                "typical_p": 1.0,
            })
        elif arch.startswith("qwen"):
            params.update({
                "temperature": 0.6,
                "temp": 0.6,
                "top_k": 50,
                "top_p": 0.9,
                "repeat_penalty": 1.08,
                "typical_p": 1.0,
            })
        elif arch.startswith("gemma"):
            params.update({
                "temperature": 0.7,
                "temp": 0.7,
                "top_k": 40,
                "top_p": 0.95,
                "repeat_penalty": 1.12,
                "typical_p": 1.0,
            })

        # Preset-driven tuning
        if preset_overrides:
            for k, v in preset_overrides.items():
                params[k] = v

        # Keep consistency: ensure temp mirrors temperature
        params["temp"] = params.get("temperature", params.get("temp", 0.8))
        # Clamp some parameters to safe ranges
        def clamp(name: str, lo: float, hi: float, default: float):
            val = params.get(name, default)
            try:
                fv = float(val)
            except Exception:
                fv = default
            params[name] = max(lo, min(hi, fv))

        clamp("temperature", 0.0, 2.0, 0.8)
        clamp("top_p", 0.0, 1.0, 0.9)
        clamp("min_p", 0.0, 1.0, 0.0)
        clamp("typical_p", 0.0, 1.0, 1.0)
        clamp("tfs_z", 0.0, 1.0, 1.0)
        params["top_k"] = max(0, int(params.get("top_k", 40) or 0))
        params["repeat_penalty"] = max(0.0, float(params.get("repeat_penalty", 1.1) or 1.1))
        params["presence_penalty"] = float(params.get("presence_penalty", 0.0) or 0.0)
        params["frequency_penalty"] = float(params.get("frequency_penalty", 0.0) or 0.0)
        params["mirostat"] = max(0, min(2, int(params.get("mirostat", 0) or 0)))
        clamp("mirostat_tau", 0.1, 20.0, 5.0)
        clamp("mirostat_eta", 0.01, 2.0, 0.1)
        params["ctx_size"] = max(512, int(params.get("ctx_size", context_length) or context_length))
        if not isinstance(params.get("stop", []), list):
            params["stop"] = []

        return params
    
    def _get_preset_overrides(self, architecture: str, preset_name: str) -> Dict[str, Any]:
        """Get preset-specific parameter overrides to fine-tune the base configuration
        
        These are incremental adjustments to the architecture-specific defaults
        to optimize for different use cases.
        """
        # Base adjustments for each preset type (applied after architecture-specific settings)
        preset_adjustments = {
            "coding": {
                # Slightly lower temperature for more deterministic code generation
                "temp_delta": -0.1,
                # Keep repeat_penalty tight
                "repeat_penalty_delta": -0.05
            },
            "conversational": {
                # Standard settings, no major changes needed
                "temp_delta": 0,
                "repeat_penalty_delta": 0
            }
        }
        
        adjustments = preset_adjustments.get(preset_name, {})
        
        # Apply deltas to current params (will be calculated in _generate_generation_params)
        overrides = {}
        if "temp_delta" in adjustments:
            overrides["temp_adjustment"] = adjustments["temp_delta"]
        if "repeat_penalty_delta" in adjustments:
            overrides["repeat_penalty_adjustment"] = adjustments["repeat_penalty_delta"]
        
        logger.debug(f"Preset '{preset_name}' tuning adjustments: {adjustments}")
        
        return overrides
    
    def _apply_preset_tuning(self, config: Dict[str, Any], architecture: str, preset_name: str) -> None:
        """Apply preset-specific tuning to the entire configuration
        
        This fine-tunes ALL relevant parameters based on the preset type:
        - Batch sizes (for throughput vs latency)
        - Context size (for memory vs context)
        - Threading (for speed vs responsiveness)
        - GPU layers (for VRAM optimization)
        """
        preset_tunings = {
            "coding": {
                # Coding benefits from lower latency, can tolerate smaller batches
                "batch_size_factor": 0.8,  # Smaller batches for lower latency
                "ubatch_size_factor": 0.8,
                "ctx_size_factor": 1.0,  # Keep context size
                "parallel_factor": 1.2,  # Slightly more parallel for throughput
            },
            "conversational": {
                # Standard balanced settings
                "batch_size_factor": 1.0,
                "ubatch_size_factor": 1.0,
                "ctx_size_factor": 1.0,
                "parallel_factor": 1.0,
            }
        }
        
        tuning = preset_tunings.get(preset_name, {})
        if not tuning:
            return
        
        # Apply factors to relevant config parameters
        if "batch_size" in config and "batch_size_factor" in tuning:
            config["batch_size"] = max(1, int(config["batch_size"] * tuning["batch_size_factor"]))
        
        if "ubatch_size" in config and "ubatch_size_factor" in tuning:
            config["ubatch_size"] = max(1, int(config["ubatch_size"] * tuning["ubatch_size_factor"]))
        
        if "ctx_size" in config and "ctx_size_factor" in tuning:
            config["ctx_size"] = int(config["ctx_size"] * tuning["ctx_size_factor"])
        
        if "parallel" in config and "parallel_factor" in tuning:
            config["parallel"] = max(1, int(config["parallel"] * tuning["parallel_factor"]))
        
        logger.debug(f"Applied preset '{preset_name}' tuning to config: {tuning}")
    
    def _generate_server_params(self) -> Dict[str, Any]:
        """Generate server-specific parameters"""
        return {
            "host": "0.0.0.0",  # Allow external connections
            "timeout": 300,  # 5 minutes timeout
            "port": 0  # Will be assigned dynamically
        }
    
    def _generate_moe_offload_pattern(self, architecture: str, available_vram_gb: float, 
                                       model_size_mb: float, is_moe: bool = False, 
                                       expert_count: int = 0) -> str:
        """Generate optimal MoE offloading pattern based on VRAM availability
        
        Returns regex pattern for the -ot (offload type) parameter to control MoE layer placement
        """
        if not is_moe or expert_count == 0:
            return ""  # No MoE offloading for non-MoE models
        
        model_size_gb = model_size_mb / 1024
        
        # Calculate VRAM pressure
        vram_ratio = available_vram_gb / model_size_gb if model_size_gb > 0 else 1.0
        
        # Strategies based on available VRAM:
        # 1. Very tight VRAM (model barely fits): Offload all MoE expert layers
        if vram_ratio < 1.2:
            return ".ffn_.*_exps.=CPU"  # All MoE layers to CPU
        
        # 2. Tight VRAM: Offload up/down projections, keep gate on GPU
        elif vram_ratio < 1.5:
            return ".ffn_(up|down)_exps.=CPU"  # Up and down projection experts to CPU
        
        # 3. Moderate VRAM: Offload only up projection
        elif vram_ratio < 2.0:
            return ".ffn_(up)_exps.=CPU"  # Only up projection to CPU
        
        # 4. Ample VRAM: Keep all on GPU
        else:
            return ""  # No offloading needed
    
    def _get_optimal_kv_cache_quant(self, available_vram_gb: float, context_length: int, 
                                    architecture: str, flash_attn_available: bool = False) -> Dict[str, Any]:
        # Deprecated: kept for backward compatibility if called internally elsewhere
        from .kv_cache import get_optimal_kv_cache_quant
        return get_optimal_kv_cache_quant(available_vram_gb, context_length, architecture, flash_attn_available)
    
    def _get_architecture_specific_flags(self, architecture: str, layer_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get architecture-specific flags and settings
        
        Returns dict with flags like jinja, cache_type_k/v, moe patterns, etc.
        """
        flags = {
            "jinja": False,
            "moe_offload_custom": ""
        }
        
        # GLM models require jinja template
        if architecture in ["glm", "glm4"]:
            flags["jinja"] = True
            logger.info("GLM architecture detected - enabling jinja template")
        
        # Qwen3-Coder models require jinja template for tool calling
        if architecture == "qwen3" and "coder" in layer_info.get('architecture', '').lower():
            flags["jinja"] = True
            logger.info("Qwen3-Coder architecture detected - enabling jinja template for tool calling")
        
        # Add MoE offloading pattern if MoE model detected
        is_moe = layer_info.get('is_moe', False)
        if is_moe:
            expert_count = layer_info.get('expert_count', 0)
            model_size_mb = layer_info.get('model_size_mb', 0)
            available_vram_gb = layer_info.get('available_vram_gb', 0)
            
            if available_vram_gb > 0:
                moe_pattern = self._generate_moe_offload_pattern(
                    architecture, available_vram_gb, model_size_mb, is_moe, expert_count
                )
                flags["moe_offload_custom"] = moe_pattern
        
        return flags
    
    def estimate_vram_usage(self, model: Model, config: Dict[str, Any], gpu_info: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate VRAM usage for given configuration using comprehensive model metadata"""
        try:
            model_size = model.file_size if model.file_size else 0
            n_gpu_layers = int(config.get("n_gpu_layers", 0) or 0)
            
            # Gather model dimensions
            layer_info = self._get_model_layer_info_sync(model)
            total_layers = max(1, int(layer_info.get('layer_count', 32) or 32))
            embedding_length = int(layer_info.get('embedding_length', 0) or 0)
            attention_head_count = int(layer_info.get('attention_head_count', 0) or 0)
            attention_head_count_kv = int(layer_info.get('attention_head_count_kv', 0) or 0)
            
            # Layer split between GPU and CPU
            layer_ratio = min(1.0, max(0.0, (n_gpu_layers / total_layers) if total_layers > 0 else 0.0))
            model_vram = int(model_size * layer_ratio)
            model_ram = max(0, int(model_size - model_vram))
            
            # KV cache and batch
            ctx_size = int(config.get("ctx_size", 4096) or 4096)
            batch_size = int(config.get("batch_size", 512) or 512)
            parallel = max(1, int(config.get("parallel", 1) or 1))
            cache_type_k = config.get("cache_type_k", "f16")
            cache_type_v = config.get("cache_type_v")
            
            if embedding_length > 0 and attention_head_count > 0:
                kv_cache_per_token = embedding_length * 2
                if attention_head_count_kv > 0:
                    kv_cache_per_token = (embedding_length * attention_head_count_kv) // attention_head_count
            else:
                kv_cache_per_token = 64
            
            quant_factor_k = self._get_kv_cache_quant_factor(cache_type_k)
            quant_factor_v = self._get_kv_cache_quant_factor(cache_type_v) if cache_type_v else quant_factor_k
            per_token_bytes = kv_cache_per_token * (quant_factor_k + (quant_factor_v if cache_type_v else quant_factor_k))
            kv_total = int(per_token_bytes * ctx_size * batch_size * parallel)
            kv_cache_vram = kv_total if (n_gpu_layers > 0 and cache_type_v) else 0
            kv_cache_ram = 0 if kv_cache_vram > 0 else kv_total
            
            batch_vram = int(0.1 * kv_cache_vram)
            batch_ram = int(0.1 * kv_cache_ram)
            
            estimated_vram = model_vram + kv_cache_vram + batch_vram
            estimated_ram = model_ram + kv_cache_ram + batch_ram
            
            # System RAM usage snapshot
            try:
                vm = psutil.virtual_memory()
                system_ram_used = int(vm.used)
                system_ram_total = int(vm.total)
            except Exception:
                system_ram_used = 0
                system_ram_total = 0
            
            # VRAM headroom check
            total_free_vram = sum(g.get("memory", {}).get("free", 0) for g in gpu_info.get("gpus", []))
            fits_in_gpu = (n_gpu_layers == 0) or (estimated_vram <= max(0, total_free_vram * 0.9))
            
            memory_mode = "ram_only"
            if n_gpu_layers > 0:
                if estimated_ram > 0:
                    memory_mode = "mixed"
                else:
                    memory_mode = "vram_only"
            
            return {
                "memory_mode": memory_mode,
                # VRAM
                "estimated_vram": estimated_vram,
                "model_vram": model_vram,
                "kv_cache_vram": kv_cache_vram,
                "batch_vram": batch_vram,
                # RAM
                "estimated_ram": estimated_ram,
                "model_ram": model_ram,
                "kv_cache_ram": kv_cache_ram,
                "batch_ram": batch_ram,
                # System RAM snapshot
                "system_ram_used": system_ram_used,
                "system_ram_total": system_ram_total,
                # Fit flag
                "fits_in_gpu": fits_in_gpu
            }
        except Exception:
            try:
                vm = psutil.virtual_memory()
                system_ram_used = int(vm.used)
                system_ram_total = int(vm.total)
            except Exception:
                system_ram_used = 0
                system_ram_total = 0
            return {
                "memory_mode": "unknown",
                "estimated_vram": 0,
                "model_vram": 0,
                "kv_cache_vram": 0,
                "batch_vram": 0,
                "estimated_ram": 0,
                "model_ram": 0,
                "kv_cache_ram": 0,
                "batch_ram": 0,
                "system_ram_used": system_ram_used,
                "system_ram_total": system_ram_total,
                "fits_in_gpu": True
            }

    def _sanitize_config(self, config: Dict[str, Any], gpu_count: int) -> Dict[str, Any]:
        """Clamp and sanitize final config values to enforce invariants and avoid edge-case crashes."""
        sanitized = dict(config)
        # Clamp integers
        def clamp(name: str, lo: int, hi: int, default: int):
            val = sanitized.get(name, default)
            try:
                iv = int(val)
            except Exception:
                iv = default
            sanitized[name] = max(lo, min(hi, iv))
        
        clamp("ctx_size", 512, 262144, 4096)
        clamp("batch_size", 1, 4096, 512)
        clamp("ubatch_size", 1, sanitized.get("batch_size", 512), max(1, sanitized.get("batch_size", 512)//2))
        clamp("parallel", 1, max(1, gpu_count if gpu_count > 0 else 1), 1)
        
        # Booleans defaults
        for b in ["no_mmap", "mlock", "low_vram", "logits_all", "flash_attn"]:
            if b in sanitized:
                sanitized[b] = bool(sanitized[b])
        
        return sanitized
    
    def _get_kv_cache_quant_factor(self, cache_type: str) -> float:
        """Get memory reduction factor for KV cache quantization"""
        factors = {
            'f32': 1.0,    # Full precision (no reduction)
            'f16': 0.5,    # Half precision
            'bf16': 0.5,   # Bfloat16
            'q8_0': 0.25,  # 8-bit quant
            'q5_1': 0.156, # 5-bit high quality
            'q5_0': 0.156, # 5-bit
            'q4_1': 0.125, # 4-bit high quality
            'q4_0': 0.125, # 4-bit
            'iq4_nl': 0.125 # 4-bit non-linear
        }
        return factors.get(cache_type, 1.0)
    
    def estimate_ram_usage(self, model: Model, config: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate RAM usage for given configuration"""
        try:
            model_size = model.file_size if model.file_size else 0
            n_gpu_layers = config.get("n_gpu_layers", 0)
            cache_type_k = config.get("cache_type_k", "f16")
            cache_type_v = config.get("cache_type_v")
            
            # Get system RAM info
            total_memory = psutil.virtual_memory().total
            available_memory = psutil.virtual_memory().available
            
            # Estimate model RAM (CPU layers + full model for GPU layers)
            layer_info = self._get_model_layer_info_sync(model)
            total_layers = layer_info.get('layer_count', 32)
            embedding_length = layer_info.get('embedding_length', 0)
            attention_head_count = layer_info.get('attention_head_count', 0)
            attention_head_count_kv = layer_info.get('attention_head_count_kv', 0)
            is_moe = layer_info.get('is_moe', False)
            
            cpu_layers = total_layers - n_gpu_layers if n_gpu_layers > 0 else total_layers
            
            if n_gpu_layers > 0:
                # GPU layers: full model loaded in RAM for GPU transfer
                model_ram = model_size
            else:
                # CPU-only: only CPU layers in RAM
                layer_ratio = cpu_layers / total_layers if cpu_layers > 0 else 1
                model_ram = int(model_size * layer_ratio)
            
            # Enhanced KV cache estimation using model architecture
            ctx_size = config.get("ctx_size", 4096)
            batch_size = config.get("batch_size", 512)
            parallel = config.get("parallel", 1)
            
            # More accurate KV cache calculation based on model parameters
            if embedding_length > 0 and attention_head_count > 0:
                # Use actual model dimensions for KV cache estimation
                kv_cache_per_token = embedding_length * 2  # Key + Value vectors
                if attention_head_count_kv > 0:
                    # Grouped Query Attention (GQA) - more efficient
                    kv_cache_per_token = (embedding_length * attention_head_count_kv) // attention_head_count
            else:
                # Fallback to rough estimation
                kv_cache_per_token = 64  # bytes per token (rough estimate)
            
            # Apply quantization reduction factor
            quant_factor_k = self._get_kv_cache_quant_factor(cache_type_k)
            quant_factor_v = self._get_kv_cache_quant_factor(cache_type_v) if cache_type_v else 1.0
            
            # KV cache with quantization
            kv_cache_ram = int(ctx_size * batch_size * parallel * kv_cache_per_token * 
                             (0.5 * quant_factor_k + 0.5 * quant_factor_v))
            
            # MoE models with CPU offloading use RAM for offloaded layers
            moe_cpu_ram = 0
            if is_moe and n_gpu_layers > 0:
                moe_pattern = config.get("moe_offload_custom", "")
                if moe_pattern:
                    # Estimate RAM usage for offloaded MoE layers
                    if ".*_exps" in moe_pattern:
                        # All MoE offloaded - ~30% of model
                        moe_cpu_ram = int(model_size * 0.3)
                    elif "up|down" in moe_pattern:
                        # Up/Down offloaded - ~20% of model
                        moe_cpu_ram = int(model_size * 0.2)
                    elif "_up_" in moe_pattern:
                        # Only Up offloaded - ~10% of model
                        moe_cpu_ram = int(model_size * 0.1)
            
            # Batch processing overhead
            batch_ram = batch_size * 1024  # Rough estimate
            
            # Additional overhead for llama.cpp
            llama_overhead = 256 * 1024 * 1024  # 256MB overhead
            
            total_ram = model_ram + kv_cache_ram + batch_ram + llama_overhead + moe_cpu_ram
            
            # Check if fits in available RAM
            fits_in_ram = total_ram <= available_memory
            
            return {
                "estimated_ram": total_ram,
                "model_ram": model_ram,
                "kv_cache_ram": kv_cache_ram,
                "batch_ram": batch_ram,
                "moe_cpu_ram": moe_cpu_ram,
                "llama_overhead": llama_overhead,
                "fits_in_ram": fits_in_ram,
                "available_ram": available_memory,
                "total_ram": total_memory,
                "utilization_percent": (total_ram / total_memory * 100) if total_memory > 0 else 0,
                "kv_cache_savings": int(ctx_size * batch_size * parallel * kv_cache_per_token * (1 - (0.5 * quant_factor_k + 0.5 * quant_factor_v)))
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "estimated_ram": 0,
                "fits_in_ram": False
            }


