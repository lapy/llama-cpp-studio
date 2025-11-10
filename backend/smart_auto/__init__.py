from typing import Dict, Any, Optional
import psutil
from backend.database import Model
from backend.logging_config import get_logger

# Import all required modules at module level for better performance
from .model_metadata import get_model_metadata
from .architecture_config import get_architecture_default_context
from .cpu_config import generate_cpu_config
from .gpu_config import generate_gpu_config, parse_compute_capability
from .memory_estimator import get_cpu_memory_gb, estimate_vram_usage, estimate_ram_usage
from .kv_cache import get_optimal_kv_cache_quant
from .moe_handler import get_architecture_specific_flags
from .generation_params import build_generation_params
from .config_builder import generate_server_params, sanitize_config, apply_preset_tuning
from .models import SystemResources, ModelMetadata

logger = get_logger(__name__)

class SmartAutoConfig:
    """Smart configuration optimizer for llama.cpp parameters"""
    
    def __init__(self):
        self.current_preset = None
    
    def _generate_cpu_config(self, model_size_mb: float, metadata, architecture: str, 
                             layer_count: int, is_moe: bool, expert_count: int) -> Dict[str, Any]:
        """Generate CPU-only configuration with MoE and architecture-specific flags."""
        cpu_cfg = generate_cpu_config(
            model_size_mb, architecture, layer_count, metadata.context_length,
            metadata.vocab_size, metadata.embedding_length, metadata.attention_head_count, 
            debug=None
        )
        
        # Add MoE parameters for CPU-only mode (MoE layers stay on CPU)
        if is_moe:
            cpu_cfg['moe_offload_pattern'] = 'none'
            cpu_cfg['moe_offload_custom'] = ''
            logger.debug("MoE model in CPU-only mode - MoE layers will run on CPU")
        
        # Add jinja flag if needed (for architectures that require it)
        if is_moe or architecture in ["glm", "glm4", "qwen3"]:
            layer_info_for_flags = {
                'is_moe': is_moe,
                'expert_count': expert_count,
                'model_size_mb': model_size_mb,
                'available_vram_gb': 0,
                'architecture': architecture
            }
            moe_config = get_architecture_specific_flags(architecture, layer_info_for_flags)
            if moe_config.get('jinja'):
                cpu_cfg['jinja'] = True
        
        return cpu_cfg
    
    def _apply_moe_optimizations(self, config: Dict[str, Any], metadata, model_size_mb: float, 
                                 system_resources: SystemResources) -> None:
        """Apply MoE-specific optimizations to configuration."""
        if not metadata.is_moe:
            return
        
        layer_info_for_flags = {
            'is_moe': metadata.is_moe,
            'expert_count': metadata.expert_count,
            'model_size_mb': model_size_mb,
            'available_vram_gb': system_resources.available_vram_gb,
            'architecture': metadata.architecture
        }
        moe_config = get_architecture_specific_flags(metadata.architecture, layer_info_for_flags)
        
        # Set MoE parameters in config
        if moe_config.get('moe_offload_custom'):
            config['moe_offload_pattern'] = 'custom'
            config['moe_offload_custom'] = moe_config['moe_offload_custom']
        else:
            config['moe_offload_pattern'] = 'none'
            config['moe_offload_custom'] = ''
        
        # Set jinja flag if needed
        if moe_config.get('jinja'):
            config['jinja'] = True
    
    async def generate_config(self, model: Model, gpu_info: Dict[str, Any], preset: Optional[str] = None, 
                             usage_mode: str = "single_user", speed_quality: Optional[int] = None,
                             use_case: Optional[str] = None, debug: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate optimal configuration based on model and GPU capabilities
        
        Args:
            model: The model to configure
            gpu_info: GPU information dictionary
            preset: Optional preset name (coding, conversational, long_context) to use as tuning parameters
            usage_mode: 'single_user' (sequential, peak KV cache) or 'multi_user' (server, typical usage)
            speed_quality: Speed/quality balance (0-100), where 0 = max speed, 100 = max quality. Default: 50
            use_case: Optional use case ('chat', 'code', 'creative', 'analysis') for targeted optimization
        """
        from backend.presets import get_architecture_and_presets
        
        try:
            config = {}
            # Store preset for later use in generation params
            self.current_preset = preset
            
            # Get model metadata
            model_size_mb = model.file_size / (1024 * 1024) if model.file_size else 0
            model_name = model.name.lower()
            
            # Get comprehensive model layer information from unified helper
            metadata = get_model_metadata(model)
            
            # Now that get_model_metadata returns dataclass with architecture detection already done
            layer_count = metadata.layer_count
            architecture = metadata.architecture
            context_length = metadata.context_length
            vocab_size = metadata.vocab_size
            embedding_length = metadata.embedding_length
            attention_head_count = metadata.attention_head_count
            attention_head_count_kv = metadata.attention_head_count_kv
            is_moe = metadata.is_moe
            expert_count = metadata.expert_count

            if debug is not None:
                debug.update({
                    "model_name": model.name,
                    "model_size_mb": model_size_mb,
                    "layer_info": metadata.to_dict(),
                })
            
            # Prepare system resources (GPU capabilities calculated once in SystemResources)
            cpu_memory = get_cpu_memory_gb()
            cpu_cores = psutil.cpu_count(logical=False) or 1
            
            system_resources = SystemResources.from_gpu_info(
                gpu_info, cpu_memory, cpu_cores
            )
            
            # Check flash attention availability using pre-parsed compute capabilities
            flash_attn_available = all(cc >= 8.0 for cc in system_resources.compute_capabilities) if system_resources.compute_capabilities else False
            system_resources.flash_attn_available = flash_attn_available
            
            if debug is not None:
                debug.update({
                    "gpu_count": system_resources.gpu_count,
                    "total_vram": system_resources.total_vram,
                    "available_vram_gb": system_resources.available_vram_gb,
                    "flash_attn_available": flash_attn_available,
                })
            
            # CPU-only configuration path
            if not system_resources.gpus:
                cpu_cfg = self._generate_cpu_config(model_size_mb, metadata, architecture, layer_count, is_moe, expert_count)
                return cpu_cfg
            
            # Select KV cache quantization BEFORE GPU config generation
            # This affects M_kv which influences context and batch size calculations
            # Use architecture default context_length for initial selection (will be refined later)
            kv_cache_config = get_optimal_kv_cache_quant(
                system_resources.available_vram_gb, context_length, architecture, system_resources.flash_attn_available
            )
            cache_type_k = kv_cache_config.get("cache_type_k", "f16")
            cache_type_v = kv_cache_config.get("cache_type_v")
            
            # GPU configuration - pass selected KV cache quantization and usage mode
            config.update(generate_gpu_config(
                model_size_mb, architecture, system_resources.gpus, system_resources.total_vram, 
                system_resources.gpu_count, system_resources.nvlink_topology, layer_count, 
                context_length, vocab_size, embedding_length, attention_head_count, 
                attention_head_count_kv=attention_head_count_kv,
                compute_capabilities=system_resources.compute_capabilities,
                cache_type_k=cache_type_k,
                cache_type_v=cache_type_v,
                usage_mode=usage_mode,
                debug=debug
            ))
            
            # Apply KV cache quantization to config
            config.update(kv_cache_config)

            # Hybrid consideration: if VRAM is tight, keep KV cache partly on CPU
            try:
                # If we have some CPU RAM headroom and low VRAM, prefer no_kv_offload False only when enough VRAM
                if system_resources.available_vram_gb < (model_size_mb / 1024) * 1.2:
                    # Signal to avoid KV offload to VRAM when VRAM is tight
                    config["no_kv_offload"] = True
                else:
                    config.setdefault("no_kv_offload", False)
            except Exception:
                pass
            
            # Apply MoE optimizations
            self._apply_moe_optimizations(config, metadata, model_size_mb, system_resources)
            
            # Use the computed ctx_size from GPU/CPU config when generating params
            effective_ctx = int(config.get("ctx_size", context_length) or context_length)
            if debug is not None:
                debug["effective_ctx_before_gen_params"] = effective_ctx
            config.update(build_generation_params(architecture, effective_ctx, None))
            
            # Apply speed/quality balancing if provided (modifies config in-place)
            if speed_quality is not None:
                self._apply_speed_quality_balancing(config, speed_quality, use_case, metadata, system_resources, debug)
            
            # Apply preset tuning if provided (modifies config in-place)
            # Note: preset takes precedence over use_case if both are provided
            if self.current_preset:
                apply_preset_tuning(config, self.current_preset)
            elif use_case:
                # Apply use_case-specific tuning if no preset
                self._apply_use_case_tuning(config, use_case)
            
            # Add server parameters
            config.update(generate_server_params())
            
            # Final sanitation and clamping
            config = sanitize_config(config, system_resources.gpu_count)
            
            return config
            
        except Exception as e:
            raise Exception(f"Failed to generate smart config: {e}")
    
    def _apply_speed_quality_balancing(self, config: Dict[str, Any], speed_quality: int, use_case: Optional[str],
                                       metadata, system_resources, debug: Optional[Dict[str, Any]] = None) -> None:
        """Apply speed/quality balancing to configuration.
        
        Args:
            config: Configuration dictionary to modify in-place
            speed_quality: Speed/quality balance (0-100), where 0 = max speed, 100 = max quality
            use_case: Optional use case for additional tuning
            metadata: Model metadata
            system_resources: SystemResources object
            debug: Optional debug dictionary
        """
        quality_factor = speed_quality / 100.0  # 0.0 = max speed, 1.0 = max quality
        
        # Context size adjustment
        # Speed (0-33): reduce context, Balanced (34-66): moderate, Quality (67-100): maximize
        current_ctx = config.get("ctx_size", 4096)
        max_context = metadata.context_length
        
        if speed_quality < 34:
            # Max speed: reduce context (2048-4096 range)
            target_ctx = 2048 + int((speed_quality / 34) * 2048)
        elif speed_quality < 67:
            # Balanced: moderate context (4096-8192 range)
            target_ctx = 4096 + int(((speed_quality - 34) / 33) * 4096)
        else:
            # Max quality: maximize context (8192-max range)
            min_quality_ctx = 8192
            target_ctx = min_quality_ctx + int(((speed_quality - 67) / 33) * (max_context - min_quality_ctx))
        
        # Respect use_case minimums
        if use_case == "code" and target_ctx < 8192:
            target_ctx = 8192
        elif use_case == "analysis" and target_ctx < 16384:
            target_ctx = 16384
        
        config["ctx_size"] = min(target_ctx, max_context)
        
        # Batch size adjustment
        # Speed-focused: larger batches for throughput
        # Quality-focused: smaller batches for lower latency per request
        current_batch = config.get("batch_size", 256)
        current_ubatch = config.get("ubatch_size", 128)
        
        if speed_quality < 34:
            # Max speed: large batches
            config["batch_size"] = 512 + int((speed_quality / 34) * 256)  # 512-768
            config["ubatch_size"] = 256 + int((speed_quality / 34) * 128)  # 256-384
        elif speed_quality < 67:
            # Balanced: medium batches
            config["batch_size"] = 384 + int(((speed_quality - 34) / 33) * 128)  # 384-512
            config["ubatch_size"] = 192 + int(((speed_quality - 34) / 33) * 64)  # 192-256
        else:
            # Max quality: smaller batches
            config["batch_size"] = 256 + int(((speed_quality - 67) / 33) * 128)  # 256-384
            config["ubatch_size"] = 128 + int(((speed_quality - 67) / 33) * 64)  # 128-192
        
        # GPU layers adjustment
        # Quality factor affects how many layers to offload
        if config.get("n_gpu_layers", 0) > 0:
            layer_count = metadata.layer_count
            base_layers = config["n_gpu_layers"]
            # Adjust based on quality factor (70-100% of base)
            adjusted_layers = int(base_layers * (0.7 + (quality_factor * 0.3)))
            config["n_gpu_layers"] = min(adjusted_layers, layer_count)
        
        # Parallel processing adjustment
        # Higher for speed-focused, lower for quality-focused
        if speed_quality < 50:
            config["parallel"] = max(1, int(3 - (speed_quality / 50) * 2))  # 3 to 1
        else:
            config["parallel"] = 1  # Quality-focused: sequential processing
        
        # Threads optimization
        cpu_threads = system_resources.cpu_cores or 4
        if speed_quality < 50:
            # Speed-focused: use more threads
            config["threads"] = cpu_threads
            config["threads_batch"] = min(cpu_threads, 8)
        else:
            # Quality-focused: optimize threads
            config["threads"] = max(2, int(cpu_threads * 0.8))
            config["threads_batch"] = max(2, int(cpu_threads * 0.8))
        
        # Flash Attention: enable for quality-focused configs when available
        if system_resources.flash_attn_available and quality_factor > 0.6:
            config["flash_attn"] = True
            # Flash attention enables V cache quantization
            if quality_factor < 0.7:
                config["cache_type_v"] = "q8_0"  # Moderate quantization for balanced
            else:
                config["cache_type_v"] = "f16"  # Better quality
        
        # KV Cache quantization adjustment
        available_vram_gb = system_resources.available_vram_gb
        total_vram_gb = system_resources.total_vram / (1024**3) if system_resources.total_vram else 0
        
        if quality_factor < 0.5 and available_vram_gb < total_vram_gb * 0.5:
            # Low VRAM or speed-focused: use quantization
            if config.get("cache_type_k") is None or config.get("cache_type_k") == "f16":
                config["cache_type_k"] = "q8_0"
            if config.get("flash_attn") and config.get("cache_type_v") is None:
                config["cache_type_v"] = "q8_0"
        elif quality_factor > 0.7:
            # Quality-focused: use full precision
            config["cache_type_k"] = "f16"
            if config.get("flash_attn"):
                config["cache_type_v"] = "f16"
        
        # Low VRAM mode for tight memory situations
        if available_vram_gb < total_vram_gb * 0.3 or (quality_factor < 0.4 and available_vram_gb < total_vram_gb * 0.5):
            config["low_vram"] = True
        
        if debug is not None:
            debug["speed_quality"] = speed_quality
            debug["quality_factor"] = quality_factor
            debug["use_case"] = use_case
            debug["adjusted_ctx_size"] = config["ctx_size"]
            debug["adjusted_batch_size"] = config["batch_size"]
    
    def _apply_use_case_tuning(self, config: Dict[str, Any], use_case: str) -> None:
        """Apply use-case-specific generation parameter tuning.
        
        Args:
            config: Configuration dictionary to modify in-place
            use_case: Use case ('chat', 'code', 'creative', 'analysis')
        """
        if use_case == "code":
            config["temp"] = 0.3
            config["temperature"] = 0.3
            config["top_k"] = 30
            if config.get("ctx_size", 4096) < 8192:
                config["ctx_size"] = 8192
        elif use_case == "creative":
            config["temp"] = 1.2
            config["temperature"] = 1.2
            config["top_k"] = 50
            config["top_p"] = 0.95
        elif use_case == "analysis":
            config["temp"] = 0.7
            config["temperature"] = 0.7
            if config.get("ctx_size", 4096) < 16384:
                config["ctx_size"] = 16384
        elif use_case == "chat":
            config["temp"] = 0.8
            config["temperature"] = 0.8
    
    def estimate_vram_usage(
        self,
        model: Model,
        config: Dict[str, Any],
        gpu_info: Dict[str, Any],
        usage_mode: str = "single_user",
        metadata: Optional[ModelMetadata] = None,
    ) -> Dict[str, Any]:
        """Estimate VRAM usage for given configuration using comprehensive model metadata"""
        return estimate_vram_usage(model, config, gpu_info, metadata=metadata, usage_mode=usage_mode)
    
    def estimate_ram_usage(
        self,
        model: Model,
        config: Dict[str, Any],
        usage_mode: str = "single_user",
        metadata: Optional[ModelMetadata] = None,
    ) -> Dict[str, Any]:
        """Estimate RAM usage for given configuration"""
        return estimate_ram_usage(model, config, metadata=metadata, usage_mode=usage_mode)


