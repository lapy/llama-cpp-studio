"""
GGUF file metadata reader for extracting model layer information
"""
import struct
import os
from typing import Dict, Optional, Any
from backend.logging_config import get_logger

logger = get_logger(__name__)

def _extract_moe_info(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract MoE (Mixture of Experts) information from GGUF metadata
    
    Args:
        metadata: GGUF metadata dictionary
        
    Returns:
        Dictionary with MoE information: is_moe, expert_count, experts_used_count
    """
    # Check for MoE indicators in metadata
    is_moe = False
    expert_count = 0
    experts_used_count = 0
    ffn_expert_count = 0
    
    # Get architecture for name-based MoE detection
    architecture = metadata.get('general.architecture', '').lower()
    
    # Check architecture name for MoE indicators
    if 'moe' in architecture or 'experts' in architecture:
        is_moe = True
        logger.debug(f"MoE architecture detected from name: {architecture}")
    
    # Try different keys for MoE detection - check all possible keys
    # Architecture-specific keys (check all, don't use elif)
    moe_keys = [
        'expert_count',
        'ffn.expert_count',
        'glm.expert_count',
        'deepseek.expert_count',
        'qwen.expert_count',
        'qwen3.expert_count',
        'qwen3moe.expert_count',
        'llama.expert_count',
        'num_experts',
        'ffn.num_experts',
        'n_experts',
        'ffn.n_experts',
    ]
    
    for key in moe_keys:
        if key in metadata:
            value = metadata[key]
            if isinstance(value, (int, float)) and value > 0:
                if not expert_count or expert_count == 0:
                    expert_count = int(value)
                else:
                    # Take the maximum if multiple keys found
                    expert_count = max(expert_count, int(value))
                is_moe = True
                logger.debug(f"Found expert_count from key '{key}': {expert_count}")
                # Don't break - continue checking for other keys
    
    # Also check ffn_expert_count separately
    if 'ffn.expert_count' in metadata:
        value = metadata['ffn.expert_count']
        if isinstance(value, (int, float)) and value > 0:
            ffn_expert_count = int(value)
            if not expert_count or expert_count == 0:
                expert_count = ffn_expert_count
            is_moe = True
    
    # Check for experts_used_count (number of active experts per token)
    experts_used_keys = [
        'experts_used_count',
        'ffn.experts_used_count',
        'glm.experts_used_count',
        'deepseek.experts_used_count',
        'qwen.experts_used_count',
        'qwen3.experts_used_count',
        'qwen3moe.experts_used_count',
        'num_experts_per_tok',
        'ffn.num_experts_per_tok',
        'n_active_experts',
        'ffn.n_active_experts',
    ]
    
    for key in experts_used_keys:
        if key in metadata:
            value = metadata[key]
            if isinstance(value, (int, float)) and value > 0:
                experts_used_count = int(value)
                logger.debug(f"Found experts_used_count from key '{key}': {experts_used_count}")
                break
    
    # Final expert_count (prefer non-ffn, but use ffn if that's all we have)
    final_expert_count = expert_count or ffn_expert_count
    
    # If we found expert_count but not experts_used_count, use common defaults
    if is_moe and experts_used_count == 0 and final_expert_count > 0:
        # GLM-4.6 uses 8 experts, DeepSeek-V3 uses 4, Qwen3 MoE uses varying amounts
        if final_expert_count >= 64:
            experts_used_count = 8
        elif final_expert_count >= 32:
            experts_used_count = 4
        else:
            experts_used_count = 2
        logger.debug(f"Using default experts_used_count: {experts_used_count} for expert_count: {final_expert_count}")
    
    # Additional check: if expert_count > 0, definitely MoE
    if final_expert_count > 0:
        is_moe = True
    
    result = {
        'is_moe': is_moe,
        'expert_count': final_expert_count,
        'experts_used_count': experts_used_count
    }
    
    if is_moe:
        logger.info(f"MoE detected: {final_expert_count} experts, {experts_used_count} active per token")
    
    return result

def read_gguf_metadata(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Read GGUF file metadata to extract model information including layer count
    
    Args:
        file_path: Path to the GGUF file
        
    Returns:
        Dictionary containing model metadata or None if failed
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"GGUF file not found: {file_path}")
            return None
            
        with open(file_path, 'rb') as f:
            # Read GGUF header
            magic = f.read(4)
            if magic != b'GGUF':
                logger.error(f"Invalid GGUF file: {file_path}")
                return None
                
            version = struct.unpack('<I', f.read(4))[0]
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
            
            logger.debug(f"GGUF version: {version}, tensors: {tensor_count}, metadata KV: {metadata_kv_count}")
            
            # Read metadata key-value pairs
            metadata = {}
            for _ in range(metadata_kv_count):
                # Read key
                key_len = struct.unpack('<Q', f.read(8))[0]
                key = f.read(key_len).decode('utf-8')
                
                # Read value type
                value_type = struct.unpack('<I', f.read(4))[0]
                
                # Read value based on type
                if value_type == 0:  # UINT8
                    value = struct.unpack('<B', f.read(1))[0]
                elif value_type == 1:  # INT8
                    value = struct.unpack('<b', f.read(1))[0]
                elif value_type == 2:  # UINT16
                    value = struct.unpack('<H', f.read(2))[0]
                elif value_type == 3:  # INT16
                    value = struct.unpack('<h', f.read(2))[0]
                elif value_type == 4:  # UINT32
                    value = struct.unpack('<I', f.read(4))[0]
                elif value_type == 5:  # INT32
                    value = struct.unpack('<i', f.read(4))[0]
                elif value_type == 6:  # FLOAT32
                    value = struct.unpack('<f', f.read(4))[0]
                elif value_type == 7:  # BOOL
                    value = struct.unpack('<B', f.read(1))[0] != 0
                elif value_type == 8:  # STRING
                    value_len = struct.unpack('<Q', f.read(8))[0]
                    value = f.read(value_len).decode('utf-8')
                elif value_type == 9:  # ARRAY
                    array_type = struct.unpack('<I', f.read(4))[0]
                    array_len = struct.unpack('<Q', f.read(8))[0]
                    value = []
                    for _ in range(array_len):
                        if array_type == 0:  # UINT8 array
                            value.append(struct.unpack('<B', f.read(1))[0])
                        elif array_type == 1:  # INT8 array
                            value.append(struct.unpack('<b', f.read(1))[0])
                        elif array_type == 2:  # UINT16 array
                            value.append(struct.unpack('<H', f.read(2))[0])
                        elif array_type == 3:  # INT16 array
                            value.append(struct.unpack('<h', f.read(2))[0])
                        elif array_type == 4:  # UINT32 array
                            value.append(struct.unpack('<I', f.read(4))[0])
                        elif array_type == 5:  # INT32 array
                            value.append(struct.unpack('<i', f.read(4))[0])
                        elif array_type == 6:  # FLOAT32 array
                            value.append(struct.unpack('<f', f.read(4))[0])
                        elif array_type == 7:  # BOOL array
                            value.append(struct.unpack('<B', f.read(1))[0] != 0)
                        elif array_type == 8:  # STRING array
                            str_len = struct.unpack('<Q', f.read(8))[0]
                            value.append(f.read(str_len).decode('utf-8'))
                else:
                    logger.warning(f"Unknown value type {value_type} for key {key}")
                    continue
                    
                metadata[key] = value
                
            logger.debug(f"Extracted metadata keys: {list(metadata.keys())}")
            
            # Extract layer count from metadata
            layer_count = _extract_layer_count(metadata)
            
            # Extract context length from metadata (with fallbacks for different architectures)
            context_length = _extract_context_length(metadata)
            
            # Extract MoE information
            moe_info = _extract_moe_info(metadata)
            
            # Extract embedding length with fallbacks for different architectures
            embedding_length = (
                metadata.get('llama.embedding_length') or
                metadata.get('qwen3moe.embedding_length') or
                metadata.get('qwen3.embedding_length') or
                metadata.get('qwen.embedding_length') or
                metadata.get('general.embedding_length') or
                0
            )
            
            # Extract attention head count with fallbacks
            attention_head_count = (
                metadata.get('llama.attention_head_count') or
                metadata.get('qwen3moe.attention_head_count') or
                metadata.get('qwen3.attention_head_count') or
                metadata.get('qwen.attention_head_count') or
                metadata.get('general.attention_head_count') or
                0
            )
            
            # Extract KV attention head count with fallbacks (for GQA)
            attention_head_count_kv = (
                metadata.get('llama.attention_head_count_kv') or
                metadata.get('qwen3moe.attention_head_count_kv') or
                metadata.get('qwen3.attention_head_count_kv') or
                metadata.get('qwen.attention_head_count_kv') or
                metadata.get('general.attention_head_count_kv') or
                0
            )
            
            return {
                'layer_count': layer_count,
                'architecture': metadata.get('general.architecture', ''),
                'context_length': context_length,
                'vocab_size': metadata.get('llama.vocab_size', 0) or metadata.get('qwen3moe.vocab_size', 0) or metadata.get('qwen3.vocab_size', 0) or metadata.get('qwen.vocab_size', 0) or 0,
                'embedding_length': int(embedding_length) if embedding_length else 0,
                'attention_head_count': int(attention_head_count) if attention_head_count else 0,
                'attention_head_count_kv': int(attention_head_count_kv) if attention_head_count_kv else 0,
                'block_count': metadata.get('llama.block_count', 0) or metadata.get('qwen3moe.block_count', 0) or metadata.get('qwen3.block_count', 0) or metadata.get('qwen.block_count', 0) or 0,
                'is_moe': moe_info['is_moe'],
                'expert_count': moe_info['expert_count'],
                'experts_used_count': moe_info['experts_used_count'],
                'metadata': metadata
            }
            
    except Exception as e:
        logger.error(f"Failed to read GGUF metadata from {file_path}: {e}")
        return None

def _extract_context_length(metadata: Dict[str, Any]) -> int:
    """
    Extract context length from GGUF metadata with multiple fallback keys
    
    Args:
        metadata: GGUF metadata dictionary
        
    Returns:
        Context length or 0 if not found
    """
    # Try different possible keys for context length
    context_keys = [
        'llama.context_length',  # Llama models
        'general.context_length',
        'general.max_sequence_length',
        'llama.max_seq_len',
        'llm.context_length',  # Some models use 'llm'
        'llm.max_seq_len',
        'context_length',  # Direct key
        'max_sequence_length',
        'max_seq_len',
        'qwen.context_length',
        'qwen3.context_length',
        'qwen3moe.context_length'
    ]
    
    detected_contexts = []
    for key in context_keys:
        if key in metadata:
            ctx_len = metadata[key]
            if isinstance(ctx_len, (int, float)) and ctx_len > 0:
                logger.info(f"Found context length candidate: {ctx_len} from key: {key}")
                detected_contexts.append(int(ctx_len))
    
    if detected_contexts:
        max_ctx = max(detected_contexts)
        if len(set(detected_contexts)) > 1:
            logger.debug(f"Multiple context lengths detected {detected_contexts}, using max={max_ctx}")
    else:
        max_ctx = 0
    
    # If not found in metadata, try to infer from architecture
    architecture = metadata.get('general.architecture', '').lower()
    if 'qwen' in architecture:
        # Qwen3 models typically have 131072 or 262144 context
        if 'qwen3' in architecture:
            logger.debug("Qwen3 detected - using default context length 262144")
            max_ctx = max(max_ctx, 262144)
        else:
            logger.debug("Qwen detected - using default context length 32768")
            max_ctx = max(max_ctx, 32768)
    elif 'gemma' in architecture:
        logger.debug("Gemma detected - using default context length 8192")
        max_ctx = max(max_ctx, 8192)
    elif 'deepseek' in architecture:
        logger.debug("DeepSeek detected - using default context length 32768")
        max_ctx = max(max_ctx, 32768)
    
    if max_ctx == 0:
        logger.warning("Could not determine context length from metadata")
    return max_ctx

def _extract_layer_count(metadata: Dict[str, Any]) -> int:
    """
    Extract layer count from GGUF metadata
    
    Args:
        metadata: GGUF metadata dictionary
        
    Returns:
        Number of layers or 0 if not found
    """
    # Try different possible keys for layer count
    layer_keys = [
        'llama.block_count',  # Most common for Llama models
        'qwen3.block_count',  # Qwen3 architecture
        'qwen3moe.block_count',  # Qwen3 MoE architecture
        'qwen.block_count',  # Qwen architecture
        'general.block_count',
        'llama.layer_count',
        'general.layer_count',
        'qwen.layer_count',
        'qwen3.layer_count',
        'llama.n_layer',
        'general.n_layer',
        'llama.num_layers',
        'general.num_layers'
    ]
    
    detected_layers = []
    
    for key in layer_keys:
        if key in metadata:
            layer_count = metadata[key]
            if isinstance(layer_count, (int, float)):
                logger.info(f"Found layer count candidate: {layer_count} from key: {key}")
                detected_layers.append(int(layer_count))
    
    if detected_layers:
        # Return the maximum detected value to avoid off-by-one mismatches between keys
        max_layers = max(detected_layers)
        if len(set(detected_layers)) > 1:
            logger.debug(f"Multiple layer counts detected {detected_layers}, using max={max_layers}")
        return max_layers
    
    # If no direct layer count, try to estimate from architecture
    architecture = metadata.get('general.architecture', '').lower()
    
    # Try Qwen/Qwen3 specific metadata
    if 'qwen' in architecture:
        # Try Qwen-specific keys
        block_count = metadata.get('qwen.block_count') or metadata.get('qwen3.block_count') or metadata.get('qwen3moe.block_count')
        if block_count:
            logger.debug(f"Found block_count from Qwen metadata: {block_count}")
            return int(block_count)
        
        # Estimate from Qwen architecture parameters
        embedding_length = metadata.get('qwen3moe.embedding_length') or metadata.get('qwen.embedding_length', 0)
        if embedding_length:
            # Rough estimation for Qwen models based on embedding size
            if embedding_length >= 8192:
                estimated_layers = 40
            elif embedding_length >= 4096:
                estimated_layers = 32
            elif embedding_length >= 2048:
                estimated_layers = 28
            else:
                estimated_layers = 24
                
            logger.debug(f"Estimated Qwen layer count: {estimated_layers} for embedding_length: {embedding_length}")
            return estimated_layers
    
    # Try Llama models
    if 'llama' in architecture:
        # For Llama models, try to estimate from model size
        vocab_size = metadata.get('llama.vocab_size', 0)
        embedding_length = metadata.get('llama.embedding_length', 0)
        
        if vocab_size > 0 and embedding_length > 0:
            # Rough estimation based on common Llama architectures
            if vocab_size >= 50000 and embedding_length >= 4096:
                # Likely 7B+ model
                estimated_layers = 32
            elif vocab_size >= 32000 and embedding_length >= 2048:
                # Likely 3B model
                estimated_layers = 28
            elif vocab_size >= 32000 and embedding_length >= 1024:
                # Likely 1B model
                estimated_layers = 22
            else:
                # Default fallback
                estimated_layers = 32
                
            logger.debug(f"Estimated layer count: {estimated_layers} for architecture: {architecture}")
            return estimated_layers
    
    # Log available keys for debugging
    logger.debug(f"Available metadata keys for layer detection: {[k for k in metadata.keys() if 'block' in k or 'layer' in k or 'block_count' in k]}")
    
    # Default fallback
    logger.warning("Could not determine layer count, using default: 32")
    return 32

def get_model_layer_info(model_path: str) -> Optional[Dict[str, Any]]:
    """
    Get layer information for a downloaded model
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary with layer information or None if failed
    """
    try:
        if not model_path.endswith('.gguf'):
            logger.error(f"Model file is not GGUF format: {model_path}")
            return None
            
        metadata = read_gguf_metadata(model_path)
        if metadata:
            return {
                'layer_count': metadata['layer_count'],
                'architecture': metadata['architecture'],
                'context_length': metadata['context_length'],
                'vocab_size': metadata['vocab_size'],
                'embedding_length': metadata['embedding_length'],
                'attention_head_count': metadata['attention_head_count'],
                'attention_head_count_kv': metadata['attention_head_count_kv'],
                'block_count': metadata['block_count'],
                'is_moe': metadata.get('is_moe', False),
                'expert_count': metadata.get('expert_count', 0),
                'experts_used_count': metadata.get('experts_used_count', 0)
            }
            
    except Exception as e:
        logger.error(f"Failed to get model layer info for {model_path}: {e}")
        
    return None
