import json
import math
import os
import psutil
from typing import Dict, Any
from backend.database import Model
from backend.gpu_detector import get_gpu_info
from backend.logging_config import get_logger

logger = get_logger(__name__)


class SmartAutoConfig:
    """Smart configuration optimizer for llama.cpp parameters"""
    
    def __init__(self):
        self.model_size_cache = {}
    
    async def generate_config(self, model: Model, gpu_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimal configuration based on model and GPU capabilities"""
        try:
            config = {}
            
            # Get model metadata
            model_size_mb = model.file_size / (1024 * 1024) if model.file_size else 0
            model_name = model.name.lower()
            
            # Get comprehensive model layer information from GGUF metadata
            layer_info = await self._get_model_layer_info(model)
            layer_count = layer_info.get('layer_count', 32)
            architecture = layer_info.get('architecture', 'unknown')
            context_length = layer_info.get('context_length', 0)  # Start with 0, we'll fix if needed
            vocab_size = layer_info.get('vocab_size', 0)
            embedding_length = layer_info.get('embedding_length', 0)
            attention_head_count = layer_info.get('attention_head_count', 0)
            attention_head_count_kv = layer_info.get('attention_head_count_kv', 0)
            is_moe = layer_info.get('is_moe', False)
            expert_count = layer_info.get('expert_count', 0)
            
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
            
            # Calculate available VRAM for optimization decisions
            available_vram_gb = 0
            if gpus:
                available_vram_gb = sum(gpu["memory"]["free"] for gpu in gpus) / (1024**3)
            
            # Determine if flash attention is available (compute capability >= 8.0)
            flash_attn_available = False
            if gpus:
                flash_attn_available = all(
                    gpu.get("compute_capability", "0.0") >= "8.0" for gpu in gpus
                )
            
            if not gpus:
                # CPU-only configuration
                return self._generate_cpu_config(model_size_mb, architecture, layer_count, context_length, vocab_size, embedding_length, attention_head_count)
            
            # GPU configuration
            config.update(self._generate_gpu_config(
                model_size_mb, architecture, gpus, total_vram, gpu_count, gpu_info.get("nvlink_topology", {}), layer_count, context_length, vocab_size, embedding_length, attention_head_count
            ))
            
            # Add KV cache quantization optimization
            kv_cache_config = self._get_optimal_kv_cache_quant(
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
                if moe_config.get('use_jinja'):
                    config['use_jinja'] = True
            
            # Add generation parameters
            config.update(self._generate_generation_params(architecture, context_length))
            
            # Add server parameters
            config.update(self._generate_server_params())
            
            return config
            
        except Exception as e:
            raise Exception(f"Failed to generate smart config: {e}")
    
    async def _get_model_layer_info(self, model: Model) -> Dict[str, Any]:
        """Get comprehensive model layer information from GGUF metadata"""
        try:
            if model.file_path and os.path.exists(model.file_path):
                from backend.gguf_reader import get_model_layer_info
                layer_info = get_model_layer_info(model.file_path)
                if layer_info:
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
                from backend.gguf_reader import get_model_layer_info
                layer_info = get_model_layer_info(model.file_path)
                if layer_info:
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
    
    def _generate_cpu_config(self, model_size_mb: float, architecture: str, layer_count: int = 32, context_length: int = 4096, vocab_size: int = 0, embedding_length: int = 0, attention_head_count: int = 0) -> Dict[str, Any]:
        """Generate CPU-only configuration optimized for available RAM"""
        # Get system memory info
        memory = psutil.virtual_memory()
        total_ram_gb = memory.total / (1024**3)
        available_ram_gb = memory.available / (1024**3)
        
        # Estimate CPU threads (leave some cores free for system)
        cpu_count = psutil.cpu_count(logical=False)
        logical_cpu_count = psutil.cpu_count(logical=True)
        threads = max(1, cpu_count - 1)  # Leave 1 core for system
        threads_batch = min(threads, logical_cpu_count - 2)  # Batch threads can use logical cores
        
        # Calculate optimal context size based on model's actual context length
        optimal_ctx_size = min(context_length or 4096, 8192)  # Cap at 8K for CPU performance
        if context_length and context_length > 32768:  # Very long context models
            optimal_ctx_size = min(context_length // 4, 16384)  # Use 1/4 of max context
        
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
        
        ubatch_size = max(64, batch_size // 2)
        
        # Determine if we should use memory mapping
        # Use mmap for large models to reduce RAM usage
        use_mmap = model_size_mb > 2000 or available_ram_gb < (model_size_mb / 1024) * 1.5
        
        config = {
            "threads": threads,
            "threads_batch": threads_batch,
            "n_gpu_layers": 0,
            "ctx_size": optimal_ctx_size,
            "batch_size": batch_size,
            "ubatch_size": ubatch_size,
            "parallel": self._get_optimal_parallel_cpu(available_ram_gb, model_size_mb),
            "no_mmap": not use_mmap,  # Enable mmap for large models
            "mlock": not use_mmap,  # Lock memory if not using mmap
            "low_vram": False,  # Not applicable for CPU
            "f16_kv": True,  # Use 16-bit KV cache to save memory
            "logits_all": False,  # Don't compute all logits to save memory
        }
        
        # Add architecture-specific optimizations
        config.update(self._get_cpu_architecture_optimizations(architecture, available_ram_gb))
        
        return config
    
    def _generate_gpu_config(self, model_size_mb: float, architecture: str, 
                           gpus: list, total_vram: int, gpu_count: int, nvlink_topology: Dict, layer_count: int = 32, context_length: int = 4096, vocab_size: int = 0, embedding_length: int = 0, attention_head_count: int = 0) -> Dict[str, Any]:
        """Generate GPU-optimized configuration"""
        config = {}
        
        # Calculate optimal GPU layers
        if gpu_count == 1:
            config.update(self._single_gpu_config(model_size_mb, architecture, gpus[0], layer_count, embedding_length, attention_head_count))
        else:
            config.update(self._multi_gpu_config(model_size_mb, architecture, gpus, nvlink_topology, layer_count))
        
        # Context size based on available VRAM and model parameters
        available_vram = sum(gpu["memory"]["free"] for gpu in gpus)
        config["ctx_size"] = self._get_optimal_context_size(architecture, available_vram, model_size_mb, layer_count, embedding_length, attention_head_count, 0)
        
        # Batch sizes based on actual memory requirements
        if embedding_length > 0 and layer_count > 0:
            optimal_batch_size = self._calculate_optimal_batch_size(available_vram / (1024**3), model_size_mb, config["ctx_size"], embedding_length, layer_count)
            config["batch_size"] = optimal_batch_size
            config["ubatch_size"] = max(1, optimal_batch_size // 2)
        else:
            # Fallback to size-based estimation
            config["batch_size"] = min(1024, max(64, int(model_size_mb / 50)))
            config["ubatch_size"] = min(512, max(32, int(model_size_mb / 100)))
        
        # Parallel sequences (conservative for multi-GPU)
        if gpu_count > 1:
            config["parallel"] = min(4, gpu_count)
        else:
            config["parallel"] = 1
        
        return config
    
    def _single_gpu_config(self, model_size_mb: float, architecture: str, gpu: Dict, layer_count: int = 32, embedding_length: int = 0, attention_head_count: int = 0) -> Dict[str, Any]:
        """Configuration for single GPU"""
        vram_gb = gpu["memory"]["total"] / (1024**3)
        free_vram_gb = gpu["memory"]["free"] / (1024**3)
        
        # Estimate layers that fit in VRAM
        # Rough estimation: each layer takes ~1-2GB depending on model size
        estimated_layers_per_gb = 8 if model_size_mb < 1000 else 4  # Smaller models have more layers per GB
        
        max_layers = int(free_vram_gb * estimated_layers_per_gb * 0.8)  # Leave 20% buffer
        
        # Use actual layer count from model metadata
        total_layers = layer_count
        
        n_gpu_layers = min(max_layers, total_layers)
        
        config = {
            "n_gpu_layers": n_gpu_layers,
            "main_gpu": gpu["index"],
            "threads": max(1, psutil.cpu_count(logical=False) - 2),
            "threads_batch": max(1, psutil.cpu_count(logical=False) - 2)
        }
        
        # Calculate optimal batch sizes based on actual memory requirements
        if embedding_length > 0 and layer_count > 0:
            # Use data-driven calculation
            optimal_batch_size = self._calculate_optimal_batch_size(free_vram_gb, model_size_mb, 4096, embedding_length, layer_count)
            config["batch_size"] = optimal_batch_size
            config["ubatch_size"] = max(1, optimal_batch_size // 2)
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
        if gpu.get("compute_capability", "0.0") >= "8.0":  # Ampere and newer
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
        if all(gpu.get("compute_capability", "0.0") >= "8.0" for gpu in gpus):
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
        vram_sizes = [gpu["memory"]["total"] for gpu in gpus]
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
        vram_sizes = [gpu["memory"]["total"] for gpu in cluster_gpus]
        total_vram = sum(vram_sizes)
        
        tensor_split = []
        for i, gpu in enumerate(gpus):
            if i in largest_cluster["gpus"]:
                ratio = gpu["memory"]["total"] / total_vram
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
        vram_sizes = [gpu["memory"]["total"] for gpu in gpus]
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
        vram_sizes = [gpu["memory"]["total"] for gpu in gpus]
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
            return min(base_context, 2048)
        
        # Use data-driven calculation if we have model parameters
        if model_size_mb > 0 and layer_count > 0 and embedding_length > 0:
            vram_gb = available_vram / (1024**3)
            calculated_max = self._calculate_max_context_size(vram_gb, model_size_mb, layer_count, embedding_length, attention_head_count, attention_head_count_kv)
            result = min(base_context, calculated_max) if calculated_max > 0 else base_context
            return max(512, result)  # Ensure minimum 512
        
        # Fallback to architecture-based limits if no model data
        vram_gb = available_vram / (1024**3)
        
        # Conservative scaling based on VRAM capacity
        if vram_gb >= 24:    # High-end GPU
            return base_context
        elif vram_gb >= 12:   # Mid-range GPU
            return min(base_context, int(base_context * 0.75))
        elif vram_gb >= 8:    # Lower-end GPU
            return min(base_context, int(base_context * 0.5))
        else:                 # Very limited VRAM
            return min(base_context, 2048)
    
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
        max_batch_size = int(available_vram_gb * 0.8 / total_per_item_gb)  # 80% safety margin
        
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
    
    def _generate_generation_params(self, architecture: str, context_length: int = 4096) -> Dict[str, Any]:
        """Generate generation parameters based on architecture and context length"""
        params = {
            "n_predict": -1,  # Infinite generation
            "temp": 0.8,
            "top_k": 40,
            "top_p": 0.95,
            "repeat_penalty": 1.1
        }
        
        # Architecture-specific adjustments
        if architecture == "codellama":
            params["temp"] = 0.1  # Lower temperature for code
            params["repeat_penalty"] = 1.05
            params["top_k"] = 40
        elif architecture == "mistral":
            params["temp"] = 0.7
            params["top_k"] = 40
        elif architecture in ["llama3", "llama"]:
            params["temp"] = 0.8
            params["top_k"] = 40
        # GLM models (GLM-4.6, ChatGLM)
        elif architecture in ["glm", "glm4"]:
            params["temp"] = 1.0  # GLM-4.6 recommended setting
            params["top_p"] = 0.95  # GLM-4.6 recommended setting
            params["top_k"] = 40  # GLM-4.6 recommended setting
            params["repeat_penalty"] = 1.05
        # DeepSeek models (DeepSeek-V3, etc.)
        elif architecture in ["deepseek", "deepseek-v3"]:
            params["temp"] = 1.0  # DeepSeek-V3 recommended
            params["top_p"] = 0.95
            params["top_k"] = 40
            params["repeat_penalty"] = 1.05
        # Qwen models
        elif architecture in ["qwen", "qwen2", "qwen3"]:
            params["temp"] = 0.7  # Qwen3-Coder recommended setting
            params["top_p"] = 0.8  # Qwen3-Coder recommended (not 0.95)
            params["top_k"] = 20  # Qwen3-Coder recommended (not 40)
            params["repeat_penalty"] = 1.05  # Qwen3-Coder recommended (not 1.1)
        # Gemma models
        elif architecture in ["gemma", "gemma3"]:
            params["temp"] = 0.9  # Gemma balanced setting
            params["top_p"] = 0.95
            params["top_k"] = 40
            params["repeat_penalty"] = 1.1
        # Phi models
        elif architecture == "phi":
            params["temp"] = 0.7
            params["top_k"] = 50
            params["repeat_penalty"] = 1.1
        
        # Context length-based optimizations
        if context_length > 32768:  # Very long context models
            params["repeat_penalty"] = max(1.05, params["repeat_penalty"] - 0.05)  # Reduce repetition for long contexts
        elif context_length < 2048:  # Short context models
            params["repeat_penalty"] = min(1.2, params["repeat_penalty"] + 0.1)  # Increase repetition penalty
        
        return params
    
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
        """Determine optimal KV cache quantization to balance memory usage and quality
        
        Returns cache_type_k and optional cache_type_v
        """
        # Memory per token: ~2 * layer_count * hidden_size * 2 bytes (fp16)
        # With quantization, we can reduce this significantly
        
        # For very long contexts (>32K), use lower precision to fit in memory
        if context_length > 32768:
            # Use q4_1 or q5_1 for long contexts (good balance of size/quality)
            cache_type_k = "q5_1" if available_vram_gb > 40 else "q4_1"
            
            # V cache quantization requires Flash Attention
            cache_type_v = cache_type_k if flash_attn_available else None
            
            return {
                "cache_type_k": cache_type_k,
                "cache_type_v": cache_type_v
            }
        
        # For long contexts (8K-32K), use moderate quantization
        elif context_length > 8192:
            cache_type_k = "q8_0" if available_vram_gb > 24 else "q4_1"
            cache_type_v = cache_type_k if flash_attn_available else None
            
            return {
                "cache_type_k": cache_type_k,
                "cache_type_v": cache_type_v
            }
        
        # For standard contexts, prefer f16 for quality if VRAM allows
        elif available_vram_gb > 16:
            return {
                "cache_type_k": "f16",
                "cache_type_v": "f16" if flash_attn_available else None
            }
        
        # Lower VRAM: use quantization
        else:
            return {
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0" if flash_attn_available else None
            }
    
    def _get_architecture_specific_flags(self, architecture: str, layer_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get architecture-specific flags and settings
        
        Returns dict with flags like use_jinja, cache_type_k/v, moe patterns, etc.
        """
        flags = {
            "use_jinja": False,
            "moe_offload_custom": ""
        }
        
        # GLM models require jinja template
        if architecture in ["glm", "glm4"]:
            flags["use_jinja"] = True
            logger.info("GLM architecture detected - enabling jinja template")
        
        # Qwen3-Coder models require jinja template for tool calling
        if architecture == "qwen3" and "coder" in layer_info.get('architecture', '').lower():
            flags["use_jinja"] = True
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
            n_gpu_layers = config.get("n_gpu_layers", 0)
            
            if n_gpu_layers == 0:
                return {
                    "estimated_vram": 0,
                    "model_vram": 0,
                    "kv_cache_vram": 0,
                    "batch_vram": 0,
                    "fits_in_gpu": True
                }
            
            # Get comprehensive model layer information
            layer_info = self._get_model_layer_info_sync(model)
            total_layers = layer_info.get('layer_count', 32)
            embedding_length = layer_info.get('embedding_length', 0)
            attention_head_count = layer_info.get('attention_head_count', 0)
            attention_head_count_kv = layer_info.get('attention_head_count_kv', 0)
            is_moe = layer_info.get('is_moe', False)
            
            layer_ratio = n_gpu_layers / total_layers if n_gpu_layers > 0 and total_layers > 0 else 0
            model_vram = int(model_size * layer_ratio)
            
            # Enhanced KV cache estimation using model architecture
            ctx_size = config.get("ctx_size", 4096)
            batch_size = config.get("batch_size", 512)
            parallel = config.get("parallel", 1)
            cache_type_k = config.get("cache_type_k", "f16")
            cache_type_v = config.get("cache_type_v")
            
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
            
            # Apply quantization reduction factor based on cache type
            quant_factor_k = self._get_kv_cache_quant_factor(cache_type_k)
            quant_factor_v = self._get_kv_cache_quant_factor(cache_type_v) if cache_type_v else 1.0
            
            # KV cache consists of K and V caches
            # Approximate that K uses about 50% of cache and V uses 50%
            kv_cache_vram = int(ctx_size * batch_size * parallel * kv_cache_per_token * 
                               (0.5 * quant_factor_k + 0.5 * quant_factor_v))
            
            # MoE offloading reduces VRAM usage
            moe_savings = 0
            if is_moe:
                moe_pattern = config.get("moe_offload_custom", "")
                if moe_pattern:
                    # Estimate savings based on pattern aggressiveness
                    if ".*_exps" in moe_pattern:
                        # All MoE offloaded - significant savings
                        moe_savings = model_vram * 0.3  # ~30% of model size is MoE
                    elif "up|down" in moe_pattern:
                        # Up/Down offloaded - moderate savings
                        moe_savings = model_vram * 0.2  # ~20% of model size
                    elif "_up_" in moe_pattern:
                        # Only Up offloaded - minor savings
                        moe_savings = model_vram * 0.1  # ~10% of model size
            
            # Adjusted model VRAM after MoE offloading
            model_vram_adjusted = max(0, int(model_vram - moe_savings))
            
            # Batch processing overhead
            batch_vram = batch_size * 1024  # Rough estimate
            
            total_vram = model_vram_adjusted + kv_cache_vram + batch_vram
            
            # Check if fits in available VRAM
            available_vram = gpu_info.get("available_vram", 0)
            fits_in_gpu = total_vram <= available_vram
            
            return {
                "estimated_vram": total_vram,
                "model_vram": model_vram_adjusted,
                "kv_cache_vram": kv_cache_vram,
                "batch_vram": batch_vram,
                "fits_in_gpu": fits_in_gpu,
                "available_vram": available_vram,
                "utilization_percent": (total_vram / available_vram * 100) if available_vram > 0 else 0,
                "moe_savings": moe_savings,
                "kv_cache_savings": int(ctx_size * batch_size * parallel * kv_cache_per_token * (1 - (0.5 * quant_factor_k + 0.5 * quant_factor_v)))
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "estimated_vram": 0,
                "fits_in_gpu": False
            }
    
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