"""
Recommendation engine for model configuration parameters.
Uses smart_auto logic with balanced preset (speed_quality=50, conversational).
"""
from typing import Dict, Any, Optional
import asyncio
from backend.database import Model
from .architecture_config import resolve_architecture


def _create_minimal_model(layer_info: Dict[str, Any], model_name: str = "", file_path: Optional[str] = None) -> Model:
    """Create a minimal Model object from layer info for smart_auto."""
    model = Model()
    model.name = model_name or "Unknown"
    model.file_path = file_path  # Use provided file_path if available for metadata reading
    model.file_size = 0
    model.huggingface_id = model_name
    return model


def _create_minimal_gpu_info() -> Dict[str, Any]:
    """Create minimal GPU info for smart_auto (assumes GPU available but will work without)."""
    return {
        "gpus": [],
        "total_vram": 0,
        "available_vram": 0,
        "compute_capabilities": [],
        "nvlink_topology": None
    }


def _extract_recommendation_from_config(
    config: Dict[str, Any],
    key: str,
    layer_info: Dict[str, Any],
    recommendation_type: str
) -> Dict[str, Any]:
    """Extract recommendation structure from generated config value."""
    
    if recommendation_type == "gpu_layers":
        layer_count = layer_info.get('layer_count', 32)
        value = config.get('n_gpu_layers', layer_count)
        # Clamp value to max
        value = min(value, layer_count)
        return {
            'recommended_value': value,
            'description': f'Recommended {value} layers' + (' (full offload)' if value == layer_count else ''),
            'balanced_value': layer_count // 2 if layer_count > 0 else 0,
            'balanced_description': f'{layer_count // 2 if layer_count > 0 else 0} layers (balanced)',
            'min': 0,
            'max': layer_count,
            'ranges': [
                {'value': 0, 'description': 'CPU-only mode (slowest, lowest VRAM)'},
                {'value': layer_count // 2 if layer_count > 0 else 0, 'description': 'Balanced (good performance, moderate VRAM)'},
                {'value': layer_count, 'description': 'Full offload (fastest, highest VRAM)'}
            ]
        }
    
    elif recommendation_type == "context_size":
        context_length = layer_info.get('context_length', 131072)
        value = config.get('ctx_size', context_length)
        # Clamp value to max
        value = min(value, context_length)
        return {
            'recommended_value': value,
            'description': f'Recommended {value:,} tokens',
            'min': 512,
            'max': context_length,
            'ranges': [
                {'min': 512, 'max': 2048, 'description': 'Short conversations'},
                {'min': 4096, 'max': 8192, 'description': 'Standard conversations'},
                {'min': 16384, 'max': context_length, 'description': f'Long documents (max {context_length:,})'}
            ]
        }
    
    elif recommendation_type == "batch_size":
        value = config.get('batch_size', 512)
        # Calculate max based on attention heads, clamp to reasonable range
        attention_heads = layer_info.get('attention_head_count', 32)
        max_val = min(1024, max(512, attention_heads * 16))
        # Clamp value to max
        value = min(value, max_val)
        return {
            'recommended_value': value,
            'description': f'Recommended {value}',
            'min': 1,
            'max': max_val,
            'ranges': [
                {'min': 1, 'max': 128, 'description': 'Low memory usage'},
                {'min': 256, 'max': 512, 'description': 'Balanced (recommended)'},
                {'min': max_val, 'max': max_val, 'description': 'Maximum throughput'}
            ]
        }
    
    elif recommendation_type == "temperature":
        value = config.get('temperature', config.get('temp', 0.8))
        # Clamp value to max
        value = min(value, 2.0)
        arch = layer_info.get('architecture', '').lower()
        recommended_str = f'{value:.1f} for balanced conversation'
        
        if 'glm' in arch or 'deepseek' in arch:
            recommended_str = f'{value:.1f} for GLM/DeepSeek models'
        elif 'qwen' in arch:
            recommended_str = f'{value:.1f} for Qwen models'
        elif 'codellama' in arch:
            recommended_str = f'{value:.1f} for code generation'
        
        return {
            'recommended_value': value,
            'description': recommended_str,
            'min': 0.0,
            'max': 2.0,
            'ranges': [
                {'min': 0.1, 'max': 0.3, 'description': 'Code generation, technical tasks'},
                {'min': 0.7, 'max': 1.0, 'description': 'General conversation (recommended)'},
                {'min': 1.5, 'max': 2.0, 'description': 'Creative writing, brainstorming'}
            ]
        }
    
    elif recommendation_type == "top_k":
        value = config.get('top_k', 40)
        # Clamp value to max
        value = min(value, 200)
        arch = layer_info.get('architecture', '').lower()
        recommended_str = f'{value} for most models'
        
        if 'glm' in arch or 'deepseek' in arch:
            recommended_str = f'{value} for GLM/DeepSeek models'
        
        return {
            'recommended_value': value,
            'description': recommended_str,
            'min': 0,
            'max': 200,
            'ranges': [
                {'min': 10, 'max': 30, 'description': 'Focused, code-like outputs'},
                {'min': 40, 'max': 50, 'description': 'Balanced (recommended)'},
                {'min': 100, 'max': 200, 'description': 'High diversity, creative writing'}
            ]
        }
    
    elif recommendation_type == "top_p":
        value = config.get('top_p', 0.9)
        # Clamp value to max
        value = min(value, 1.0)
        arch = layer_info.get('architecture', '').lower()
        recommended_str = f'{value:.2f} for most models'
        
        if 'glm' in arch or 'deepseek' in arch:
            recommended_str = f'{value:.2f} for GLM/DeepSeek models'
        elif 'qwen' in arch:
            recommended_str = f'{value:.2f} for Qwen models'
        
        return {
            'recommended_value': value,
            'description': recommended_str,
            'min': 0.0,
            'max': 1.0,
            'ranges': [
                {'min': 0.7, 'max': 0.8, 'description': 'More conservative'},
                {'min': 0.9, 'max': 0.95, 'description': 'Balanced (recommended)'},
                {'min': 0.95, 'max': 1.0, 'description': 'Higher diversity'}
            ]
        }
    
    elif recommendation_type == "parallel":
        value = config.get('parallel', 1)
        # Clamp value to max
        value = min(value, 8)
        attention_heads = layer_info.get('attention_head_count', 32)
        return {
            'recommended_value': value,
            'description': f'Recommended {value} based on {attention_heads} attention heads',
            'min': 1,
            'max': 8
        }
    
    # Fallback
    return {
        'recommended_value': config.get(key, 0),
        'description': f'Recommended {config.get(key, 0)}',
        'min': 0,
        'max': 100
    }


async def _generate_balanced_config(model_layer_info: Dict[str, Any], model_name: str = "", file_path: Optional[str] = None) -> Dict[str, Any]:
    """Generate balanced configuration using smart_auto with speed_quality=50 and conversational preset."""
    from backend.smart_auto import SmartAutoConfig
    
    # Create minimal model object with file_path if available
    model = _create_minimal_model(model_layer_info, model_name, file_path)
    
    # Create minimal GPU info (will work for CPU-only too)
    gpu_info = _create_minimal_gpu_info()
    
    # Create smart_auto config generator
    smart_config = SmartAutoConfig()
    
    # Generate config with balanced settings:
    # - speed_quality=50 (balanced between speed and quality)
    # - preset="conversational" (balanced preset)
    # - usage_mode="single_user" (standard usage)
    config = await smart_config.generate_config(
        model=model,
        gpu_info=gpu_info,
        preset="conversational",  # Balanced preset
        usage_mode="single_user",
        speed_quality=50,  # Balanced (50 = equal speed/quality)
        use_case=None,
        debug=None
    )
    
    return config


async def get_model_recommendations(
    model_layer_info: Dict[str, Any], 
    model_name: str = "",
    file_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get all recommendations using smart_auto logic with balanced preset.
    
    Uses smart_auto's generate_config with:
    - speed_quality=50 (balanced)
    - preset="conversational" (balanced preset)
    - usage_mode="single_user"
    
    Args:
        model_layer_info: Layer information dict from GGUF metadata
        model_name: Optional model name for fallback
        file_path: Optional file path for metadata reading
        
    Returns:
        Dict with all recommendations extracted from smart_auto generated config
    """
    try:
        # Generate balanced config using smart_auto
        config = await _generate_balanced_config(model_layer_info, model_name, file_path)
        
        # Extract recommendations from generated config
        return {
            'gpu_layers': _extract_recommendation_from_config(config, 'n_gpu_layers', model_layer_info, 'gpu_layers'),
            'context_size': _extract_recommendation_from_config(config, 'ctx_size', model_layer_info, 'context_size'),
            'batch_size': _extract_recommendation_from_config(config, 'batch_size', model_layer_info, 'batch_size'),
            'temperature': _extract_recommendation_from_config(config, 'temperature', model_layer_info, 'temperature'),
            'top_k': _extract_recommendation_from_config(config, 'top_k', model_layer_info, 'top_k'),
            'top_p': _extract_recommendation_from_config(config, 'top_p', model_layer_info, 'top_p'),
            'parallel': _extract_recommendation_from_config(config, 'parallel', model_layer_info, 'parallel')
        }
    except Exception as e:
        # Fallback to basic recommendations if smart_auto fails
        from backend.logging_config import get_logger
        logger = get_logger(__name__)
        logger.warning(f"Failed to generate recommendations with smart_auto: {e}. Using fallback.")
        
        # Return basic fallback recommendations
        layer_count = model_layer_info.get('layer_count', 32)
        context_length = model_layer_info.get('context_length', 131072)
        attention_heads = model_layer_info.get('attention_head_count', 32)
        
        return {
            'gpu_layers': {
                'recommended_value': layer_count,
                'description': f'Recommended {layer_count} layers (full offload)',
                'min': 0,
                'max': layer_count,
                'ranges': [
                    {'value': 0, 'description': 'CPU-only mode'},
                    {'value': layer_count // 2, 'description': 'Balanced'},
                    {'value': layer_count, 'description': 'Full offload'}
                ]
            },
            'context_size': {
                'recommended_value': context_length,
                'description': f'Max {context_length:,} tokens',
                'min': 512,
                'max': context_length,
                'ranges': []
            },
            'batch_size': {
                'recommended_value': 512,
                'description': 'Recommended 512',
                'min': 1,
                'max': 1024,
                'ranges': []
            },
            'temperature': {
                'recommended_value': 0.8,
                'description': '0.8 for balanced conversation',
                'min': 0.0,
                'max': 2.0,
                'ranges': []
            },
            'top_k': {
                'recommended_value': 40,
                'description': '40 for most models',
                'min': 0,
                'max': 200,
                'ranges': []
            },
            'top_p': {
                'recommended_value': 0.9,
                'description': '0.9 for most models',
                'min': 0.0,
                'max': 1.0,
                'ranges': []
            },
            'parallel': {
                'recommended_value': min(8, max(1, attention_heads // 4)),
                'description': f'Recommended based on {attention_heads} attention heads',
                'min': 1,
                'max': 8
            }
        }
