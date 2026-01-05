"""
GGUF file metadata reader for extracting model layer information
"""
import struct
import os
import mmap
from enum import IntEnum
from typing import Dict, Optional, Any, List, Tuple, BinaryIO

from backend.logging_config import get_logger
from backend.architecture_profiles import compute_layers_for_architecture

logger = get_logger(__name__)


class GGUFValueType(IntEnum):
    """
    GGUF Value Types as defined in the specification.
    These map directly to the binary enum values stored in the file.
    """
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGUFReader:
    """
    A high-performance, secure reader for GGUF model files.
    
    Architectural Features:
    - **Memory Mapping**: Uses `mmap` for zero-copy access and instant loading of large files.
    - **Lazy Loading**: Parses metadata and tensor info but does not load weights until requested.
    - **Security Hardened**: Implements strict bounds checking to prevent heap overflows and DoS.
    - **Endian Safe**: Enforces Little-Endian parsing consistent with GGUF spec.
    """

    # The GGUF file signature "GGUF" in little-endian hex
    GGUF_MAGIC = 0x46554747
    
    # Supported GGUF versions
    SUPPORTED_VERSIONS = {1, 2, 3}
    
    # Mapping of GGUF types to struct format characters and byte sizes
    # Format: (struct_char, size_in_bytes)
    _TYPE_MAP = {
        GGUFValueType.UINT8:   ('B', 1),
        GGUFValueType.INT8:    ('b', 1),
        GGUFValueType.UINT16:  ('H', 2),
        GGUFValueType.INT16:   ('h', 2),
        GGUFValueType.UINT32:  ('I', 4),
        GGUFValueType.INT32:   ('i', 4),
        GGUFValueType.FLOAT32: ('f', 4),
        GGUFValueType.BOOL:    ('?', 1),
        GGUFValueType.UINT64:  ('Q', 8),
        GGUFValueType.INT64:   ('q', 8),
        GGUFValueType.FLOAT64: ('d', 8),
    }

    def __init__(self, file_path: str):
        """
        Initialize the reader. Does not open the file until used as a context manager.
        """
        self.file_path = file_path
        self._f: Optional[BinaryIO] = None
        self._mm: Optional[mmap.mmap] = None
        self._offset = 0
        
        # Public metadata containers
        self.version = 0
        self.tensor_count = 0
        self.kv_count = 0
        self.metadata: Dict[str, Any] = {}
        self.tensors: Dict[str, Dict[str, Any]] = {}
        self.tensor_data_start_offset: int = 0  # Track where tensor data begins (after alignment)

    def __enter__(self):
        """
        Context Manager Entry.
        Opens the file and maps it into memory. Parsing begins immediately.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"GGUF file not found: {self.file_path}")
            
        try:
            self._f = open(self.file_path, 'rb')
            # Map the whole file. accessing self._mm[x:y] returns bytes without reading the whole file.
            self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
            self._offset = 0
            
            # Start the parsing chain
            self._parse_header()
            self._parse_metadata()
            self._parse_tensor_info()
            
            return self
        except Exception as e:
            # clean up if initialization fails
            self.__exit__(type(e), e, None)
            raise e

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context Manager Exit.
        Ensures file handles and memory maps are closed to prevent resource leaks.
        """
        if self._mm:
            self._mm.close()
        if self._f:
            self._f.close()

    def _read_bytes(self, size: int) -> bytes:
        """
        Internal safe read method.
        Performs bounds checking against the memory map size.
        """
        if self._offset + size > len(self._mm):
            raise EOFError(f"Attempted to read {size} bytes at offset {self._offset}, but file ended.")
        
        # Slicing mmap returns bytes. This is a memory copy operation.
        data = self._mm[self._offset : self._offset + size]
        self._offset += size
        return data

    def _unpack(self, fmt: str, size: int) -> Tuple[Any, ...]:
        """
        Helper to read and unpack in one step.
        """
        data = self._read_bytes(size)
        try:
            return struct.unpack(fmt, data)
        except struct.error as e:
            raise ValueError(f"Struct unpack failed at offset {self._offset - size}: {e}")

    def _parse_header(self):
        """
        Parses the Global Header.
        Checks Magic Number and Version.
        """
        # 1. Magic (4 bytes)
        magic = self._read_bytes(4)
        if magic != b'GGUF':
            raise ValueError(f"Invalid GGUF Magic: {magic}. Expected b'GGUF'.")

        # 2. Version (4 bytes, uint32)
        (self.version,) = self._unpack('<I', 4)
        if self.version not in self.SUPPORTED_VERSIONS:
            logger.warning(f"GGUF Version {self.version} is not officially supported by this reader (Supports 1-3). Parsing may fail.")

        # 3. Counts (8 bytes each, uint64)
        (self.tensor_count,) = self._unpack('<Q', 8)
        (self.kv_count,) = self._unpack('<Q', 8)
        
        logger.debug(f"Header Parsed: v{self.version}, Tensors={self.tensor_count}, KV={self.kv_count}")

    def _read_string(self) -> str:
        """
        Reads a GGUF string: [Length (uint64)] + [UTF-8 bytes].
        Security: Limits string length to prevent OOM attacks.
        """
        (length,) = self._unpack('<Q', 8)
        
        # SECURITY: Max string length sanity check (e.g., 10MB)
        if length > 10 * 1024 * 1024: 
            raise ValueError(f"Security Alert: String length {length} exceeds safety limit (10MB).")
            
        b_str = self._read_bytes(length)
        return b_str.decode('utf-8')

    def _read_value(self, value_type: int, depth: int = 0) -> Any:
        """
        Dispatch method to read a value based on its type enum.
        
        Args:
            value_type: The GGUF value type enum ID
            depth: Current recursion depth for arrays (security limit)
        """
        if value_type in self._TYPE_MAP:
            fmt, size = self._TYPE_MAP[value_type]
            val = self._unpack(f'<{fmt}', size)
            result = val[0]  # Unpack returns a tuple, get first element
            
            # SECURITY: Strict validation for BOOL type (must be 0 or 1)
            if value_type == GGUFValueType.BOOL:
                if result not in (0, 1):
                    raise ValueError(f"Invalid BOOL value at offset {self._offset - size}: {result}. Must be 0 or 1.")
                return bool(result)
            
            return result
        elif value_type == GGUFValueType.STRING:
            return self._read_string()
        elif value_type == GGUFValueType.ARRAY:
            return self._read_array(depth=depth)
        else:
            raise ValueError(f"Unknown Value Type ID: {value_type}")

    def _read_array(self, depth: int = 0) -> List[Any]:
        """
        Parses a GGUF Array.
        Format: [Item Type (uint32)] [Count (uint64)] [Items...]
        Optimization: Uses batch unpacking for primitive types.
        
        Args:
            depth: Current recursion depth (for nested arrays, max 3)
        """
        # SECURITY: Recursion depth limit to prevent DoS via nested arrays
        MAX_DEPTH = 3
        if depth > MAX_DEPTH:
            raise ValueError(f"Array recursion depth {depth} exceeds maximum allowed depth {MAX_DEPTH}. Possible DoS attack.")
        
        (item_type_id,) = self._unpack('<I', 4)
        (count,) = self._unpack('<Q', 8)
        
        # SECURITY: Integer wrap-around protection
        # Check if count * size would overflow or be unreasonable
        if count > (2**63):  # Sanity check for count itself
            raise ValueError(f"Array count {count} is unreasonably large. Possible integer overflow attack.")
        
        # 1. Handle Fixed-Width Primitives (Fast Path)
        if item_type_id in self._TYPE_MAP:
            fmt_char, size = self._TYPE_MAP[item_type_id]
            
            # SECURITY: Check for integer wrap-around in size calculation
            try:
                total_bytes = count * size
            except OverflowError:
                raise ValueError(f"Array size calculation overflow: count={count}, size={size}")
            
            # Additional sanity check
            if total_bytes > len(self._mm):
                raise ValueError(f"Array claims {total_bytes} bytes, but file is only {len(self._mm)} bytes.")
            
            # Security: Pre-check file bounds
            if self._offset + total_bytes > len(self._mm):
                raise EOFError(f"Array claims {total_bytes} bytes, but file is too short.")
                
            # Optimization: Read all bytes at once
            data_block = self._read_bytes(total_bytes)
            
            # struct.unpack with repeat count (e.g., '<100f')
            # Limit standard unpack to reasonable chunks to avoid RAM spikes
            if count > 1_000_000:
                logger.debug(f"Large array detected ({count} items). Switching to iterative reading.")
                return self._read_array_iterative(item_type_id, count, depth)
            
            try:
                full_fmt = f'<{count}{fmt_char}'
                return list(struct.unpack(full_fmt, data_block))
            except struct.error:
                # Fallback if format string is too long for python
                logger.debug(f"Struct unpack failed for large array, using iterative method")
                # Rewind offset (we already advanced it in _read_bytes)
                self._offset -= total_bytes
                return self._read_array_iterative(item_type_id, count, depth)

        # 2. Handle Variable-Width (Strings/Arrays) or Fallback
        return self._read_array_iterative(item_type_id, count, depth)

    def _read_array_iterative(self, item_type_id: int, count: int, depth: int = 0) -> List[Any]:
        """
        Iterative reader for complex types or massive arrays.
        
        Args:
            item_type_id: The type ID of array items
            count: Number of items in the array
            depth: Current recursion depth
        """
        result = []
        for _ in range(count):
            result.append(self._read_value(item_type_id, depth=depth + 1))
        return result
    
    def _skip_array_items(self, item_type_id: int, count: int):
        """
        Skip array items without reading them (for vocabulary arrays).
        
        Args:
            item_type_id: The type ID of array items
            count: Number of items to skip
        """
        # For fixed-width primitives, calculate size and skip bytes
        if item_type_id in self._TYPE_MAP:
            _, size = self._TYPE_MAP[item_type_id]
            total_bytes = count * size
            # Security check
            if self._offset + total_bytes > len(self._mm):
                raise EOFError(f"Cannot skip {total_bytes} bytes: file too short")
            # Simply advance offset without reading
            self._offset += total_bytes
        elif item_type_id == GGUFValueType.STRING:
            # For strings, we need to read each length and skip that many bytes
            for _ in range(count):
                (length,) = self._unpack('<Q', 8)
                if self._offset + length > len(self._mm):
                    raise EOFError(f"Cannot skip {length} bytes: file too short")
                self._offset += length
        elif item_type_id == GGUFValueType.ARRAY:
            # For nested arrays, recursively skip
            for _ in range(count):
                (nested_item_type,) = self._unpack('<I', 4)
                (nested_count,) = self._unpack('<Q', 8)
                self._skip_array_items(nested_item_type, nested_count)
        else:
            # For unknown types, raise error rather than potentially corrupting file position
            raise ValueError(f"Cannot skip array items of unknown type {item_type_id}")

    def _parse_metadata(self):
        """
        Iterates over the Key-Value pairs defined in the header.
        Skips vocabulary arrays to save memory.
        """
        logger.debug("Parsing Metadata...")
        vocab_key_patterns = [
            'tokenizer.ggml.tokens',
            'tokenizer.ggml.token_type',
            'tokenizer.ggml.merges',
            'tokenizer.ggml.bos_token_id',
            'tokenizer.ggml.eos_token_id',
            'tokenizer.ggml.unk_token_id',
            'tokenizer.ggml.padding_token_id',
            'model.vocab',
            'vocab',
            'tokenizer.vocab',
            'tokenizer.tokens',
        ]
        
        for _ in range(self.kv_count):
            key = self._read_string()
            (val_type,) = self._unpack('<I', 4)
            
            # Skip vocabulary arrays to save memory
            key_lower = key.lower()
            is_vocab_key = any(pattern in key_lower for pattern in vocab_key_patterns)
            
            if is_vocab_key and val_type == GGUFValueType.ARRAY:
                # Skip array: read item type and count, then skip all items
                (item_type_id,) = self._unpack('<I', 4)
                (count,) = self._unpack('<Q', 8)
                logger.debug(f"Skipping vocabulary array '{key}' with {count} items")
                # Skip reading the array items
                self._skip_array_items(item_type_id, count)
            elif is_vocab_key:
                # Skip non-array vocabulary values (read but don't store)
                value = self._read_value(val_type)
                logger.debug(f"Skipping vocabulary key '{key}' (type {val_type})")
            else:
                # Normal metadata: read and store
                value = self._read_value(val_type)
                self.metadata[key] = value
                # Debug log for interesting keys
                if "architecture" in key_lower:
                    logger.debug(f"Architecture: {value}")

    def _parse_tensor_info(self):
        """
        Parses Tensor Information. 
        Note: Does NOT read tensor data. Records offsets for later retrieval.
        """
        logger.debug("Parsing Tensor Info...")
        for _ in range(self.tensor_count):
            name = self._read_string()
            (n_dims,) = self._unpack('<I', 4)
            
            # Dimensions are always uint64
            dims = []
            for _ in range(n_dims):
                (dim,) = self._unpack('<Q', 8)
                dims.append(dim)
            
            (tensor_type,) = self._unpack('<I', 4)
            (offset,) = self._unpack('<Q', 8)
            
            self.tensors[name] = {
                'shape': dims,
                'type': tensor_type,
                'offset': offset,  # This is relative to tensor_data_start_offset
            }
        
        # After parsing all tensor info, the current offset is where tensor data starts
        # GGUF spec requires alignment, but we'll track the actual offset
        # The offset in tensor info is relative to the start of tensor data block
        self.tensor_data_start_offset = self._offset
        
        # GGUF files typically align tensor data to page boundaries (4096 bytes)
        # However, the offsets in tensor info are already relative to the data start
        # So we just need to remember where we are now
        logger.debug(f"Tensor data starts at offset: {self.tensor_data_start_offset}")

    def get_tensor_data(self, tensor_name: str) -> bytes:
        """
        Retrieves the raw bytes for a specific tensor.
        
        Calculates the absolute position based on the tensor info offset
        (which is relative to tensor_data_start_offset).
        
        Args:
            tensor_name: Name of the tensor to retrieve
            
        Returns:
            Raw bytes of the tensor data
            
        Raises:
            KeyError: If tensor name is not found
            ValueError: If tensor data cannot be read (bounds check)
        """
        if tensor_name not in self.tensors:
            raise KeyError(f"Tensor '{tensor_name}' not found. Available tensors: {list(self.tensors.keys())[:10]}...")
        
        if not self._mm:
            raise RuntimeError("GGUFReader is not open. Use as context manager: 'with GGUFReader(path) as reader:'")
        
        info = self.tensors[tensor_name]
        shape = info['shape']
        tensor_type = info['type']
        relative_offset = info['offset']
        
        # Calculate absolute offset
        # The offset in tensor info is relative to tensor_data_start_offset
        absolute_offset = self.tensor_data_start_offset + relative_offset
        
        # SECURITY: Bounds check
        if absolute_offset > len(self._mm):
            raise ValueError(f"Tensor '{tensor_name}' offset {absolute_offset} exceeds file size {len(self._mm)}")
        
        # Calculate tensor size based on shape and type
        # Calculate total elements from shape
        total_elements = 1
        for dim in shape:
            if dim <= 0:
                raise ValueError(f"Invalid tensor dimension {dim} in shape {shape}")
            total_elements *= dim
        
        # Basic size calculation for common unquantized types
        # Quantized types require block-based calculation (complex, see GGUF spec)
        # This is a simplified implementation - full production code would need
        # quantization block size tables for all Q* types
        element_size_map = {
            0: 4,   # F32 (float32)
            1: 2,   # F16 (float16)
            # Quantized types (2-9) have variable block sizes
            # For now, estimate conservatively at 1 byte per element
            # A full implementation would calculate: blocks = (total_elements + block_size - 1) // block_size
        }
        
        # Estimate size based on type
        if tensor_type in element_size_map:
            estimated_size = total_elements * element_size_map[tensor_type]
        else:
            # For quantized types, use conservative estimate
            # Most quantized types are < 1 byte per element (e.g., Q4_0 is ~0.5 bytes/element)
            # But we'll use 1 byte/element as a safe upper bound
            estimated_size = total_elements
        
        # SECURITY: Ensure estimated size doesn't exceed remaining file
        max_available = len(self._mm) - absolute_offset
        if estimated_size > max_available:
            logger.warning(f"Tensor '{tensor_name}' estimated size {estimated_size} exceeds available {max_available}. Using available bytes.")
            estimated_size = max_available
        
        # Return the data slice
        # Note: For production use with quantized types, implement proper
        # quantization block size calculation based on GGUF quantization spec
        return bytes(self._mm[absolute_offset:absolute_offset + estimated_size])


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
        'minimax.expert_count',  # MiniMax models
        'minimax.ffn.expert_count',
        'm2.expert_count',
        'glm.expert_count',
        'glm4moe.expert_count',
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
        'minimax.experts_used_count',  # MiniMax models
        'minimax.ffn.experts_used_count',
        'm2.experts_used_count',
        'glm.experts_used_count',
        'glm4moe.experts_used_count',
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
                if not experts_used_count or experts_used_count == 0:
                    experts_used_count = int(value)
                else:
                    # Take the maximum if multiple keys found
                    experts_used_count = max(experts_used_count, int(value))
                is_moe = True
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
    Read GGUF metadata from a file and extract layer information.
    
    Args:
        file_path: Path to the GGUF file
        
    Returns:
        Dictionary with layer information and metadata, or None if failed
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"GGUF file not found: {file_path}")
            return None
        
        if not file_path.endswith('.gguf'):
            logger.error(f"File is not a GGUF file: {file_path}")
            return None
        
        # Use the new GGUFReader with context manager
        with GGUFReader(file_path) as reader:
            metadata = reader.metadata
            
            logger.debug(f"Extracted metadata keys: {list(metadata.keys())}")
            
            # First, extract a best-effort architectural depth (block count)
            base_block_count = _extract_layer_count(metadata)
            
            # Then compute architecture-aware block_count and effective_layer_count
            architecture = metadata.get('general.architecture', '').lower()
            layer_info = compute_layers_for_architecture(
                architecture=architecture,
                metadata=metadata,
                base_block_count=base_block_count,
            )
            block_count = int(layer_info.get('block_count', 0) or 0)
            effective_layer_count = int(layer_info.get('effective_layer_count', 0) or 0)
            
            # Extract context length from metadata (with fallbacks for different architectures)
            context_length = _extract_context_length(metadata)
            
            # Extract MoE information
            moe_info = _extract_moe_info(metadata)
            
            # Extract embedding length with fallbacks for different architectures
            embedding_length = (
                metadata.get('minimax.embedding_length')  # MiniMax models
                or metadata.get('m2.embedding_length')
                or metadata.get('glm4moe.embedding_length')
                or metadata.get('glm4.embedding_length')  # GLM4 models (non-MoE)
                or metadata.get('glm.embedding_length')  # GLM models (non-MoE)
                or metadata.get('llama.embedding_length')
                or metadata.get('qwen3moe.embedding_length')
                or metadata.get('qwen3.embedding_length')
                or metadata.get('qwen.embedding_length')
                or metadata.get('general.embedding_length')
                or 0
            )
            
            # Extract attention head count with fallbacks
            attention_head_count = (
                metadata.get('minimax.attention.head_count')  # MiniMax models
                or metadata.get('minimax.attention_head_count')
                or metadata.get('m2.attention.head_count')
                or metadata.get('m2.attention_head_count')
                or metadata.get('glm4moe.attention.head_count')
                or metadata.get('glm4.attention.head_count')  # GLM4 models (non-MoE)
                or metadata.get('glm4.attention_head_count')  # Alternative GLM4 format
                or metadata.get('glm.attention.head_count')  # GLM models (non-MoE)
                or metadata.get('glm.attention_head_count')  # Alternative GLM format
                or metadata.get('llama.attention_head_count')
                or metadata.get('qwen3moe.attention_head_count')
                or metadata.get('qwen3.attention_head_count')
                or metadata.get('qwen.attention_head_count')
                or metadata.get('general.attention_head_count')
                or 0
            )
            
            # Extract KV attention head count with fallbacks (for GQA)
            attention_head_count_kv = (
                metadata.get('minimax.attention.head_count_kv')  # MiniMax models
                or metadata.get('minimax.attention_head_count_kv')
                or metadata.get('m2.attention.head_count_kv')
                or metadata.get('m2.attention_head_count_kv')
                or metadata.get('glm4moe.attention.head_count_kv')
                or metadata.get('glm4.attention.head_count_kv')  # GLM4 models (non-MoE)
                or metadata.get('glm4.attention_head_count_kv')  # Alternative GLM4 format
                or metadata.get('glm.attention.head_count_kv')  # GLM models (non-MoE)
                or metadata.get('glm.attention_head_count_kv')  # Alternative GLM format
                or metadata.get('llama.attention_head_count_kv')
                or metadata.get('qwen3moe.attention_head_count_kv')
                or metadata.get('qwen3.attention_head_count_kv')
                or metadata.get('qwen.attention_head_count_kv')
                or metadata.get('general.attention_head_count_kv')
                or 0
            )
            
            # Extract parameter count
            parameter_count = _extract_parameter_count(metadata)
            
            return {
                'layer_count': int(effective_layer_count) if effective_layer_count else 0,
                'architecture': metadata.get('general.architecture', ''),
                'context_length': context_length,
                'parameter_count': parameter_count,  # Formatted as "32B", "36B", etc.
                'vocab_size': 0,  # Not extracted from metadata
                'embedding_length': int(embedding_length) if embedding_length else 0,
                'attention_head_count': int(attention_head_count) if attention_head_count else 0,
                'attention_head_count_kv': int(attention_head_count_kv) if attention_head_count_kv else 0,
                'block_count': block_count,
                'is_moe': moe_info['is_moe'],
                'expert_count': moe_info['expert_count'],
                'experts_used_count': moe_info['experts_used_count'],
                'metadata': metadata
            }
            
    except Exception as e:
        logger.error(f"Failed to read GGUF metadata from {file_path}: {e}", exc_info=True)
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
        # Architecture-specific keys
        # MiniMax models (MiniMax-M2.1 etc.)
        'minimax.context_length',
        'minimax.model_max_length',
        'minimax.max_position_embeddings',
        'minimax.max_sequence_length',
        'minimax.max_seq_len',
        'm2.context_length',
        'm2.model_max_length',
        'm2.max_position_embeddings',
        # Seed OSS models
        'seed_oss.context_length',  # Seed OSS models (with underscore)
        'seed_oss.model_max_length',
        'seed_oss.max_position_embeddings',  # Seed OSS models use this
        'seed_oss.max_sequence_length',
        'seed_oss.max_seq_len',
        'seed.context_length',  # Seed OSS models
        'seed.model_max_length',
        'seed.max_position_embeddings',
        'seed.max_sequence_length',
        'seed.max_seq_len',
        'seed-oss.context_length',
        'seed-oss.model_max_length',
        'seed-oss.max_position_embeddings',
        'seedoss.context_length',
        'seedoss.model_max_length',
        'seedoss.max_position_embeddings',
        'kimi.context_length',  # Kimi models (Moonshot AI)
        'kimi.model_max_length',  # Kimi models may use model_max_length
        'kimi.max_sequence_length',
        'kimi.max_seq_len',
        'moonshot.context_length',  # Kimi models may use moonshot prefix
        'moonshot.model_max_length',
        'moonshot.max_sequence_length',
        'moonshot.max_seq_len',
        'llama.context_length',  # Llama models
        'llama.max_seq_len',
        'glm4moe.context_length',  # GLM4 MoE models
        'glm4moe.model_max_length',
        'glm4moe.max_position_embeddings',  # GLM4 MoE models
        'glm4moe.max_sequence_length',  # GLM4 MoE models (alternative)
        'glm4moe.max_seq_len',  # GLM4 MoE models (alternative)
        'glm4.context_length',  # GLM4 models (non-MoE)
        'glm4.model_max_length',  # GLM4 models
        'glm4.max_position_embeddings',  # GLM4 models often use this
        'glm4.max_sequence_length',  # GLM4 models (non-MoE)
        'glm4.max_seq_len',  # GLM4 models (non-MoE)
        'glm.context_length',  # GLM models (non-MoE)
        'glm.model_max_length',
        'glm.max_position_embeddings',
        'glm.max_sequence_length',  # GLM models (non-MoE)
        'glm.max_seq_len',  # GLM models (non-MoE)
        'qwen.context_length',
        'qwen.model_max_length',  # Qwen models may use model_max_length
        'qwen.max_position_embeddings',  # Qwen models often use this for context length
        'qwen.max_sequence_length',
        'qwen.max_seq_len',
        'qwen2.context_length',
        'qwen2.model_max_length',
        'qwen2.max_position_embeddings',  # Qwen2 models often use this
        'qwen2.max_sequence_length',
        'qwen2.max_seq_len',
        'qwen3.context_length',
        'qwen3.model_max_length',
        'qwen3.max_position_embeddings',
        'qwen3.max_sequence_length',
        'qwen3.max_seq_len',
        'qwen3moe.context_length',
        'qwen3moe.model_max_length',
        'qwen3moe.max_position_embeddings',
        # Generic keys (check model_max_length early as it's commonly used)
        'model_max_length',  # Direct key (common in many models including Kimi)
        'max_position_embeddings',  # Direct key (common in many models)
        'general.model_max_length',  # Some models use general.model_max_length
        'general.max_position_embeddings',
        'general.context_length',
        'general.max_sequence_length',
        'llm.context_length',  # Some models use 'llm'
        'llm.max_seq_len',
        'llm.model_max_length',
        'context_length',  # Direct key
        'max_sequence_length',
        'max_seq_len',
    ]
    
    detected_contexts = []
    # Sanity check: cap at reasonable maximum (1 billion tokens)
    # This prevents corrupted metadata from causing display issues
    MAX_REASONABLE_CONTEXT = 1_000_000_000
    for key in context_keys:
        if key in metadata:
            ctx_len = metadata[key]
            if isinstance(ctx_len, (int, float)) and ctx_len > 0:
                ctx_int = int(ctx_len)
                # Validate reasonable bounds
                if ctx_int > MAX_REASONABLE_CONTEXT:
                    logger.warning(f"Unreasonably large context length detected from key '{key}': {ctx_len}, skipping")
                    continue
                logger.info(f"Found context length candidate: {ctx_len} from key: {key}")
                detected_contexts.append(ctx_int)
    
    if detected_contexts:
        max_ctx = max(detected_contexts)
        if len(set(detected_contexts)) > 1:
            logger.debug(f"Multiple context lengths detected {detected_contexts}, using max={max_ctx}")
        return max_ctx
    
    # Log available keys for debugging context length issues
    architecture = metadata.get('general.architecture', 'unknown')
    available_keys = [k for k in metadata.keys() if any(term in k.lower() for term in ['context', 'length', 'seq', 'position'])]
    logger.warning(
        f"Could not determine context length from GGUF metadata for architecture '{architecture}'. "
        f"Available relevant keys: {available_keys}. Defaulting to 0"
    )
    return 0

def _extract_parameter_count(metadata: Dict[str, Any]) -> Optional[str]:
    """
    Extract parameter count from GGUF metadata and format as human-readable string (e.g., "32B", "36B")
    
    Args:
        metadata: GGUF metadata dictionary
        
    Returns:
        Formatted parameter count string (e.g., "32B", "1.7B", "70B") or None if not found
    """
    # Common metadata keys for parameter count (as integer values)
    # Check "parameters" (plural, may already be formatted as "72B") first
    param_keys = [
        'general.parameters',  # May already be formatted as "72B"
        'parameters',  # Direct key, may already be formatted as "72B"
        'general.parameter_count',
        'general.num_parameters',
        'general.total_parameters',
        'minimax.parameters',  # MiniMax models
        'minimax.parameter_count',
        'minimax.num_parameters',
        'm2.parameters',
        'm2.parameter_count',
        'llama.parameters',
        'llama.parameter_count',
        'llama.num_parameters',
        'glm4.parameters',
        'glm4.parameter_count',
        'glm4moe.parameters',
        'glm4moe.parameter_count',
        'glm.parameters',
        'glm.parameter_count',
        'qwen.parameters',
        'qwen.parameter_count',
        'qwen3.parameters',
        'qwen3.parameter_count',
        'qwen3moe.parameters',
        'qwen3moe.parameter_count',
        'kimi.parameters',
        'kimi.parameter_count',
        'moonshot.parameters',
        'moonshot.parameter_count',
        'parameter_count',
        'num_parameters',
        'total_parameters',
    ]
    
    detected_params = []
    
    for key in param_keys:
        if key in metadata:
            param_count = metadata[key]
            if isinstance(param_count, (int, float)) and param_count > 0:
                logger.info(f"Found parameter count candidate: {param_count} from key: {key}")
                detected_params.append(int(param_count))
            elif isinstance(param_count, str):
                param_str = param_count.strip()
                # If already formatted as "72B", "1.7B", etc., return it directly
                if param_str and param_str[-1].upper() in ('B', 'M', 'K'):
                    try:
                        # Validate it's a valid format (number + suffix)
                        float_part = float(param_str[:-1])
                        if float_part > 0:
                            logger.info(f"Found pre-formatted parameter count: {param_str} from key: {key}")
                            # Return directly if already formatted (but normalize case)
                            return param_str.upper() if param_str[-1].isupper() else param_str
                    except (ValueError, AttributeError):
                        pass
                
                # Try to parse string representation (e.g., "32B", "32000000000")
                try:
                    # Handle string formats like "32B", "1.7B", etc.
                    param_str_upper = param_str.upper()
                    if param_str_upper.endswith('B'):
                        num_str = param_str_upper[:-1]
                        num = float(num_str)
                        detected_params.append(int(num * 1e9))
                    elif param_str_upper.endswith('M'):
                        num_str = param_str_upper[:-1]
                        num = float(num_str)
                        detected_params.append(int(num * 1e6))
                    elif param_str_upper.endswith('K'):
                        num_str = param_str_upper[:-1]
                        num = float(num_str)
                        detected_params.append(int(num * 1e3))
                    else:
                        # Try parsing as plain number string
                        detected_params.append(int(float(param_str)))
                except (ValueError, AttributeError):
                    pass
    
    if detected_params:
        # Use the maximum value if multiple found
        max_params = max(detected_params)
        
        # Format as human-readable string
        if max_params >= 1e9:
            # Billions
            billions = max_params / 1e9
            if billions == int(billions):
                return f"{int(billions)}B"
            else:
                # Round to 1 decimal place
                return f"{billions:.1f}B"
        elif max_params >= 1e6:
            # Millions
            millions = max_params / 1e6
            if millions == int(millions):
                return f"{int(millions)}M"
            else:
                return f"{millions:.1f}M"
        elif max_params >= 1e3:
            # Thousands
            thousands = max_params / 1e3
            if thousands == int(thousands):
                return f"{int(thousands)}K"
            else:
                return f"{thousands:.1f}K"
        else:
            return str(max_params)
    
    # Log available keys for debugging
    architecture = metadata.get('general.architecture', 'unknown')
    available_keys = [k for k in metadata.keys() if any(term in k.lower() for term in ['param', 'parameter'])]
    if available_keys:
        logger.debug(
            f"Parameter count keys found but not parseable for architecture '{architecture}': {available_keys}"
        )
    
    return None

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
        # MiniMax models (MiniMax-M2.1 etc.)
        'minimax.block_count',
        'minimax.n_layer',
        'minimax.num_hidden_layers',
        'minimax.layer_count',
        'm2.block_count',
        'm2.n_layer',
        'm2.num_hidden_layers',
        # Seed OSS models
        'seed_oss.block_count',  # Seed OSS models (with underscore)
        'seed_oss.n_layer',
        'seed_oss.num_hidden_layers',  # Seed OSS models use this
        'seed_oss.layer_count',
        'seed.block_count',  # Seed OSS models
        'seed.n_layer',
        'seed.num_hidden_layers',
        'seed.layer_count',
        'llama.block_count',  # Most common for Llama models (Seed models may use this)
        'glm4moe.block_count',  # GLM4 MoE architecture
        'glm4.block_count',  # GLM4 architecture (non-MoE)
        'glm.block_count',  # GLM architecture (non-MoE)
        'qwen3.block_count',  # Qwen3 architecture
        'qwen3moe.block_count',  # Qwen3 MoE architecture
        'qwen.block_count',  # Qwen architecture
        'general.block_count',
        'seed.layer_count',  # Seed OSS models
        'llama.layer_count',
        'glm4moe.layer_count',
        'glm4.layer_count',  # GLM4 architecture (non-MoE)
        'glm.layer_count',  # GLM architecture (non-MoE)
        'general.layer_count',
        'qwen.layer_count',
        'qwen3.layer_count',
        'seed.n_layer',  # Seed OSS models
        'llama.n_layer',
        'glm4moe.n_layer',
        'glm4.n_layer',  # GLM4 architecture (non-MoE)
        'glm.n_layer',  # GLM architecture (non-MoE)
        'general.n_layer',
        'seed.num_layers',  # Seed OSS models
        'llama.num_layers',
        'glm4moe.num_layers',
        'glm4moe.num_hidden_layers',
        'glm4.num_layers',  # GLM4 architecture (non-MoE)
        'glm4.num_hidden_layers',  # GLM4 models use this (e.g., GLM-4-32B has 61)
        'glm.num_layers',  # GLM architecture (non-MoE)
        'glm.num_hidden_layers',
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
            logger.info(f"Multiple layer counts detected {detected_layers}, using max={max_layers}")
        else:
            logger.info(f"Layer count detected: {max_layers}")
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
                'vocab_size': 0,  # Not extracted from metadata
                'embedding_length': metadata['embedding_length'],
                'attention_head_count': metadata['attention_head_count'],
                'attention_head_count_kv': metadata['attention_head_count_kv'],
                'block_count': metadata['block_count'],
                'is_moe': metadata['is_moe'],
                'expert_count': metadata['expert_count'],
                'experts_used_count': metadata['experts_used_count'],
            }
        return None
    except Exception as e:
        logger.error(f"Failed to get model layer info from {model_path}: {e}", exc_info=True)
        return None
