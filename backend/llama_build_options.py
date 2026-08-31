"""
Catalog of llama.cpp / ggml CMake build options.

Source of truth aligned with upstream ggml/CMakeLists.txt and CMakeLists.txt
(ggml-org/llama.cpp). Used for defaults, settings coercion, BuildConfig fields,
CMake flag emission, and the build-settings UI catalog API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class BuildOptionDef:
    """One user-facing build setting mapped to a CMake (or special) flag."""

    key: str
    field: str
    kind: str  # bool | str | enum
    default: Any
    label: str
    desc: str
    category: str
    cmake: Optional[str] = None
    requires: Optional[str] = None
    enum_values: Optional[tuple] = None
    special: Optional[str] = None  # handled outside generic set_flag
    # Which engines expose this option in the UI / emit it at build time.
    engines: frozenset = frozenset({"llama_cpp", "ik_llama"})


BOTH = frozenset({"llama_cpp", "ik_llama"})
LLAMA_ONLY = frozenset({"llama_cpp"})
IK_ONLY = frozenset({"ik_llama"})


# Categories shown in the build settings UI (order matters).
# collapsed=True → UI starts with the section closed (advanced / rarely needed).
CATEGORIES: Sequence[Dict[str, Any]] = (
    {"id": "backends", "label": "GPU & compute backends", "collapsed": False},
    {"id": "iqk", "label": "IQK optimizations (ik_llama)", "collapsed": True},
    {"id": "cuda", "label": "CUDA options", "collapsed": True},
    {"id": "hip", "label": "HIP / ROCm options", "collapsed": True},
    {"id": "musa", "label": "MUSA options", "collapsed": True},
    {"id": "vulkan", "label": "Vulkan options", "collapsed": True},
    {"id": "metal", "label": "Metal options", "collapsed": True},
    {"id": "sycl", "label": "SYCL options", "collapsed": True},
    {"id": "opencl", "label": "OpenCL options", "collapsed": True},
    {"id": "webgpu", "label": "WebGPU options", "collapsed": True},
    {"id": "cpu_accel", "label": "CPU / BLAS acceleration", "collapsed": True},
    {"id": "artifacts", "label": "Build artifacts", "collapsed": True},
    {"id": "ggml", "label": "GGML general", "collapsed": True},
    {"id": "cpu_isa", "label": "CPU instruction sets", "collapsed": True},
    {"id": "debug", "label": "Debug & sanitizers", "collapsed": True},
    {"id": "advanced", "label": "Advanced strings", "collapsed": True},
)

# Common backends shown first; the rest are grouped under "More backends".
PRIMARY_BACKEND_KEYS = frozenset(
    {"cuda", "hip", "vulkan", "metal", "sycl", "opencl", "blas"}
)

# Category visibility: show only when parent setting is enabled.
CATEGORY_REQUIRES: Dict[str, str] = {
    "cuda": "cuda",
    "hip": "hip",
    "musa": "musa",
    "vulkan": "vulkan",
    "metal": "metal",
    "sycl": "sycl",
    "opencl": "opencl",
    "webgpu": "webgpu",
}

# Categories only relevant for a given engine (others still may contain mixed options).
CATEGORY_ENGINES: Dict[str, frozenset] = {
    "iqk": IK_ONLY,
    "opencl": LLAMA_ONLY,
    "webgpu": LLAMA_ONLY,
    "musa": BOTH,  # both have MUSA
}


def normalize_engine_id(engine: Optional[str]) -> Optional[str]:
    if not engine:
        return None
    value = str(engine).strip().lower()
    if value in ("ik_llama", "ik_llama.cpp", "ik"):
        return "ik_llama"
    if value in ("llama_cpp", "llama.cpp", "llama"):
        return "llama_cpp"
    return value


def _b(
    key: str,
    field: str,
    default: bool,
    label: str,
    desc: str,
    category: str,
    cmake: Optional[str] = None,
    requires: Optional[str] = None,
    special: Optional[str] = None,
    engines: frozenset = BOTH,
) -> BuildOptionDef:
    return BuildOptionDef(
        key=key,
        field=field,
        kind="bool",
        default=default,
        label=label,
        desc=desc,
        category=category,
        cmake=cmake,
        requires=requires,
        special=special,
        engines=engines,
    )


def _s(
    key: str,
    field: str,
    default: str,
    label: str,
    desc: str,
    category: str,
    cmake: Optional[str] = None,
    requires: Optional[str] = None,
    special: Optional[str] = None,
    enum_values: Optional[tuple] = None,
    engines: frozenset = BOTH,
) -> BuildOptionDef:
    return BuildOptionDef(
        key=key,
        field=field,
        kind="enum" if enum_values else "str",
        default=default,
        label=label,
        desc=desc,
        category=category,
        cmake=cmake,
        requires=requires,
        special=special,
        enum_values=enum_values,
        engines=engines,
    )


# fmt: off
BUILD_OPTIONS: tuple[BuildOptionDef, ...] = (
    # ── Backends ──────────────────────────────────────────────
    _b("cuda", "enable_cuda", False, "CUDA", "GGML_CUDA — NVIDIA GPU", "backends", "GGML_CUDA", special="cuda"),
    _b("hip", "enable_hip", False, "HIP / ROCm", "GGML_HIP / GGML_HIPBLAS — AMD GPU", "backends", special="hip"),
    _b("vulkan", "enable_vulkan", False, "Vulkan", "GGML_VULKAN — cross-vendor GPU", "backends", "GGML_VULKAN"),
    _b("metal", "enable_metal", False, "Metal", "GGML_METAL — Apple GPU (default on macOS upstream)", "backends", "GGML_METAL"),
    _b("sycl", "enable_sycl", False, "SYCL", "GGML_SYCL — Intel oneAPI / Arc", "backends", "GGML_SYCL"),
    _b("opencl", "enable_opencl", False, "OpenCL", "GGML_OPENCL", "backends", "GGML_OPENCL", engines=LLAMA_ONLY),
    _b("musa", "enable_musa", False, "MUSA", "GGML_MUSA — Moore Threads", "backends", "GGML_MUSA"),
    _b("webgpu", "enable_webgpu", False, "WebGPU", "GGML_WEBGPU", "backends", "GGML_WEBGPU", engines=LLAMA_ONLY),
    _b("rpc", "enable_rpc", False, "RPC", "GGML_RPC — remote backend", "backends", "GGML_RPC"),
    _b("blas", "enable_blas", False, "BLAS", "GGML_BLAS — CPU BLAS (llama.cpp)", "backends", "GGML_BLAS", special="blas", engines=LLAMA_ONLY),
    # Backward-compatible alias: treated like blas + OpenBLAS vendor when blas unset
    _b("openblas", "enable_openblas", False, "OpenBLAS (legacy)", "Alias for BLAS + vendor OpenBLAS", "backends", special="openblas_alias", engines=LLAMA_ONLY),
    _b("zendnn", "enable_zendnn", False, "ZenDNN", "GGML_ZENDNN", "backends", "GGML_ZENDNN", engines=LLAMA_ONLY),
    _b("zdnn", "enable_zdnn", False, "zDNN", "GGML_ZDNN — IBM zDNN", "backends", "GGML_ZDNN", engines=LLAMA_ONLY),
    _b("openvino", "enable_openvino", False, "OpenVINO", "GGML_OPENVINO", "backends", "GGML_OPENVINO", engines=LLAMA_ONLY),
    _b("hexagon", "enable_hexagon", False, "Hexagon", "GGML_HEXAGON", "backends", "GGML_HEXAGON", engines=LLAMA_ONLY),
    _b("virtgpu", "enable_virtgpu", False, "VirtGPU", "GGML_VIRTGPU — Virgl remoting frontend", "backends", "GGML_VIRTGPU", engines=LLAMA_ONLY),
    _b("virtgpu_backend", "enable_virtgpu_backend", False, "VirtGPU backend", "GGML_VIRTGPU_BACKEND", "backends", "GGML_VIRTGPU_BACKEND", engines=LLAMA_ONLY),
    _b("et", "enable_et", False, "ET backend", "GGML_ET", "backends", "GGML_ET", engines=LLAMA_ONLY),
    _b("et_sysemu", "enable_et_sysemu", False, "ET via sysemu", "GGML_ET_SYSEMU", "backends", "GGML_ET_SYSEMU", requires="et", engines=LLAMA_ONLY),

    # ── IQK (ik_llama.cpp) ────────────────────────────────────
    _b("iqk_mul_mat", "enable_iqk_mul_mat", True, "IQK matmul", "GGML_IQK_MUL_MAT — optimized IQK matrix multiplies", "iqk", "GGML_IQK_MUL_MAT", engines=IK_ONLY),
    _b("iqk_flash_attention", "enable_iqk_flash_attention", True, "IQK FlashAttention", "GGML_IQK_FLASH_ATTENTION — CPU FA kernels", "iqk", "GGML_IQK_FLASH_ATTENTION", engines=IK_ONLY),
    _b("iqk_fa_all_quants", "enable_iqk_fa_all_quants", True, "IQK FA all quants", "GGML_IQK_FA_ALL_QUANTS — larger binary / longer compile", "iqk", "GGML_IQK_FA_ALL_QUANTS", engines=IK_ONLY),
    _b("expert_chunking", "enable_expert_chunking", True, "Expert chunking", "GGML_EXPERT_CHUNKING — MoE chunking", "iqk", "GGML_EXPERT_CHUNKING", engines=IK_ONLY),
    _b("nccl", "enable_nccl", True, "NCCL", "GGML_NCCL (ik_llama top-level)", "iqk", "GGML_NCCL", engines=IK_ONLY),
    _s("max_contexts", "max_contexts", "", "Max contexts", "GGML_MAX_CONTEXTS (blank = upstream default)", "iqk", "GGML_MAX_CONTEXTS", engines=IK_ONLY),

    # ── CUDA ──────────────────────────────────────────────────
    _b("cuda_fa", "enable_cuda_fa", True, "FlashAttention kernels", "GGML_CUDA_FA", "cuda", "GGML_CUDA_FA", requires="cuda", engines=LLAMA_ONLY),
    _b("flash_attention", "enable_flash_attention", False, "FA all quants", "GGML_CUDA_FA_ALL_QUANTS (larger binary)", "cuda", "GGML_CUDA_FA_ALL_QUANTS", requires="cuda", special="cuda_fa_all"),
    _b("cuda_graphs", "enable_cuda_graphs", True, "CUDA graphs", "GGML_CUDA_GRAPHS / GGML_CUDA_USE_GRAPHS", "cuda", special="cuda_graphs", requires="cuda"),
    _b("cuda_force_mmq", "enable_cuda_force_mmq", False, "Force MMQ", "GGML_CUDA_FORCE_MMQ — prefer mmq over cuBLAS", "cuda", "GGML_CUDA_FORCE_MMQ", requires="cuda"),
    _b("cuda_force_cublas", "enable_cuda_force_cublas", False, "Force cuBLAS", "GGML_CUDA_FORCE_CUBLAS", "cuda", "GGML_CUDA_FORCE_CUBLAS", requires="cuda"),
    _b("cuda_force_dmmv", "enable_cuda_force_dmmv", False, "Force DMMV", "GGML_CUDA_FORCE_DMMV (ik_llama)", "cuda", "GGML_CUDA_FORCE_DMMV", requires="cuda", engines=IK_ONLY),
    _b("cuda_iqk_force_bf16", "enable_cuda_iqk_force_bf16", False, "IQK force BF16", "GGML_CUDA_IQK_FORCE_BF16 — bf16 cuBLAS fallback", "cuda", "GGML_CUDA_IQK_FORCE_BF16", requires="cuda", engines=IK_ONLY),
    _b("cuda_f16", "enable_cuda_f16", False, "CUDA FP16 math", "GGML_CUDA_F16 (ik_llama)", "cuda", "GGML_CUDA_F16", requires="cuda", engines=IK_ONLY),
    _b("cuda_no_peer_copy", "enable_cuda_no_peer_copy", False, "Disable P2P copies", "GGML_CUDA_NO_PEER_COPY", "cuda", "GGML_CUDA_NO_PEER_COPY", requires="cuda"),
    _b("cuda_no_vmm", "enable_cuda_no_vmm", False, "Disable CUDA VMM", "GGML_CUDA_NO_VMM", "cuda", "GGML_CUDA_NO_VMM", requires="cuda"),
    _b("cuda_nccl", "enable_cuda_nccl", True, "CUDA NCCL", "GGML_CUDA_NCCL — NVIDIA Collective Comm.", "cuda", "GGML_CUDA_NCCL", requires="cuda", engines=LLAMA_ONLY),
    _s("cuda_architectures", "cuda_architectures", "", "CUDA architectures", "CMAKE_CUDA_ARCHITECTURES (blank = auto)", "cuda", requires="cuda", special="cuda_arch"),
    _s("cuda_peer_max_batch_size", "cuda_peer_max_batch_size", "128", "Peer max batch size", "GGML_CUDA_PEER_MAX_BATCH_SIZE", "cuda", "GGML_CUDA_PEER_MAX_BATCH_SIZE", requires="cuda"),
    _s("cuda_min_batch_offload", "cuda_min_batch_offload", "32", "Min batch offload", "GGML_CUDA_MIN_BATCH_OFFLOAD (ik_llama)", "cuda", "GGML_CUDA_MIN_BATCH_OFFLOAD", requires="cuda", engines=IK_ONLY),
    _s("cuda_dmmv_x", "cuda_dmmv_x", "32", "DMMV X stride", "GGML_CUDA_DMMV_X (ik_llama)", "cuda", "GGML_CUDA_DMMV_X", requires="cuda", engines=IK_ONLY),
    _s("cuda_mmv_y", "cuda_mmv_y", "1", "MMV Y block", "GGML_CUDA_MMV_Y (ik_llama)", "cuda", "GGML_CUDA_MMV_Y", requires="cuda", engines=IK_ONLY),
    _s("cuda_kquants_iter", "cuda_kquants_iter", "2", "K-quants iters", "GGML_CUDA_KQUANTS_ITER (ik_llama)", "cuda", "GGML_CUDA_KQUANTS_ITER", requires="cuda", engines=IK_ONLY),
    _s("cuda_fusion", "cuda_fusion", "1", "CUDA fusion", "GGML_CUDA_FUSION (ik_llama)", "cuda", "GGML_CUDA_FUSION", requires="cuda", engines=IK_ONLY),
    _s("cuda_compression_mode", "cuda_compression_mode", "size", "CUDA compression mode", "GGML_CUDA_COMPRESSION_MODE (CUDA 12.8+)", "cuda", "GGML_CUDA_COMPRESSION_MODE", requires="cuda", enum_values=("none", "speed", "balance", "size")),

    # ── HIP ───────────────────────────────────────────────────
    _b("hip_uma", "enable_hip_uma", False, "HIP UMA", "GGML_HIP_UMA — unified memory (ik_llama)", "hip", "GGML_HIP_UMA", requires="hip", engines=IK_ONLY),
    _b("hip_graphs", "enable_hip_graphs", True, "HIP graphs", "GGML_HIP_GRAPHS", "hip", "GGML_HIP_GRAPHS", requires="hip", engines=LLAMA_ONLY),
    _b("hip_rccl", "enable_hip_rccl", False, "RCCL", "GGML_HIP_RCCL", "hip", "GGML_HIP_RCCL", requires="hip", engines=LLAMA_ONLY),
    _b("hip_no_vmm", "enable_hip_no_vmm", True, "Disable HIP VMM", "GGML_HIP_NO_VMM", "hip", "GGML_HIP_NO_VMM", requires="hip", engines=LLAMA_ONLY),
    _b("hip_mmq_mfma", "enable_hip_mmq_mfma", True, "MFMA for MMQ", "GGML_HIP_MMQ_MFMA — CDNA MFMA MMA", "hip", "GGML_HIP_MMQ_MFMA", requires="hip", engines=LLAMA_ONLY),
    _b("hip_export_metrics", "enable_hip_export_metrics", False, "Export metrics", "GGML_HIP_EXPORT_METRICS", "hip", "GGML_HIP_EXPORT_METRICS", requires="hip", engines=LLAMA_ONLY),

    # ── MUSA ──────────────────────────────────────────────────
    _b("musa_graphs", "enable_musa_graphs", False, "MUSA graphs", "GGML_MUSA_GRAPHS (experimental)", "musa", "GGML_MUSA_GRAPHS", requires="musa", engines=LLAMA_ONLY),
    _b("musa_mudnn_copy", "enable_musa_mudnn_copy", False, "muDNN copy", "GGML_MUSA_MUDNN_COPY", "musa", "GGML_MUSA_MUDNN_COPY", requires="musa", engines=LLAMA_ONLY),

    # ── Vulkan ────────────────────────────────────────────────
    _b("vulkan_check_results", "enable_vulkan_check_results", False, "Check results", "GGML_VULKAN_CHECK_RESULTS", "vulkan", "GGML_VULKAN_CHECK_RESULTS", requires="vulkan"),
    _b("vulkan_debug", "enable_vulkan_debug", False, "Debug output", "GGML_VULKAN_DEBUG", "vulkan", "GGML_VULKAN_DEBUG", requires="vulkan"),
    _b("vulkan_memory_debug", "enable_vulkan_memory_debug", False, "Memory debug", "GGML_VULKAN_MEMORY_DEBUG", "vulkan", "GGML_VULKAN_MEMORY_DEBUG", requires="vulkan"),
    _b("vulkan_shader_debug_info", "enable_vulkan_shader_debug_info", False, "Shader debug info", "GGML_VULKAN_SHADER_DEBUG_INFO", "vulkan", "GGML_VULKAN_SHADER_DEBUG_INFO", requires="vulkan"),
    _b("vulkan_validate", "enable_vulkan_validate", False, "Validation layers", "GGML_VULKAN_VALIDATE", "vulkan", "GGML_VULKAN_VALIDATE", requires="vulkan"),
    _b("vulkan_run_tests", "enable_vulkan_run_tests", False, "Run Vulkan tests", "GGML_VULKAN_RUN_TESTS", "vulkan", "GGML_VULKAN_RUN_TESTS", requires="vulkan"),
    _b("vulkan_no_coopmat", "enable_vulkan_no_coopmat", False, "Disable coopmat", "GGML_VULKAN_NO_COOPMAT (ik_llama)", "vulkan", "GGML_VULKAN_NO_COOPMAT", requires="vulkan", engines=IK_ONLY),
    _b("vulkan_no_coopmat2", "enable_vulkan_no_coopmat2", False, "Disable coopmat2", "GGML_VULKAN_NO_COOPMAT2 (ik_llama)", "vulkan", "GGML_VULKAN_NO_COOPMAT2", requires="vulkan", engines=IK_ONLY),
    _b("vulkan_no_bf16", "enable_vulkan_no_bf16", False, "Disable Vulkan BF16", "GGML_VULKAN_NO_BF16 (ik_llama)", "vulkan", "GGML_VULKAN_NO_BF16", requires="vulkan", engines=IK_ONLY),
    _b("vulkan_no_int_dot", "enable_vulkan_no_int_dot", False, "Disable int dot", "GGML_VULKAN_NO_INT_DOT (ik_llama)", "vulkan", "GGML_VULKAN_NO_INT_DOT", requires="vulkan", engines=IK_ONLY),

    # ── Metal ─────────────────────────────────────────────────
    _b("metal_ndebug", "enable_metal_ndebug", False, "Disable Metal debug", "GGML_METAL_NDEBUG", "metal", "GGML_METAL_NDEBUG", requires="metal"),
    _b("metal_shader_debug", "enable_metal_shader_debug", False, "Shader debug", "GGML_METAL_SHADER_DEBUG — -fno-fast-math", "metal", "GGML_METAL_SHADER_DEBUG", requires="metal"),
    _b("metal_embed_library", "enable_metal_embed_library", True, "Embed Metal library", "GGML_METAL_EMBED_LIBRARY", "metal", "GGML_METAL_EMBED_LIBRARY", requires="metal"),
    _s("metal_macosx_version_min", "metal_macosx_version_min", "", "macOS version min", "GGML_METAL_MACOSX_VERSION_MIN", "metal", "GGML_METAL_MACOSX_VERSION_MIN", requires="metal"),
    _s("metal_std", "metal_std", "", "Metal standard", "GGML_METAL_STD (-std flag)", "metal", "GGML_METAL_STD", requires="metal"),

    # ── SYCL ──────────────────────────────────────────────────
    _b("sycl_f16", "enable_sycl_f16", False, "SYCL FP16", "GGML_SYCL_F16", "sycl", "GGML_SYCL_F16", requires="sycl"),
    _b("sycl_graph", "enable_sycl_graph", True, "SYCL graphs", "GGML_SYCL_GRAPH", "sycl", "GGML_SYCL_GRAPH", requires="sycl", engines=LLAMA_ONLY),
    _b("sycl_host_mem_fallback", "enable_sycl_host_mem_fallback", True, "Host mem fallback", "GGML_SYCL_HOST_MEM_FALLBACK", "sycl", "GGML_SYCL_HOST_MEM_FALLBACK", requires="sycl", engines=LLAMA_ONLY),
    _b("sycl_level_zero", "enable_sycl_level_zero", True, "Level Zero API", "GGML_SYCL_SUPPORT_LEVEL_ZERO_API", "sycl", "GGML_SYCL_SUPPORT_LEVEL_ZERO_API", requires="sycl", engines=LLAMA_ONLY),
    _b("sycl_dnn", "enable_sycl_dnn", True, "oneDNN", "GGML_SYCL_DNN", "sycl", "GGML_SYCL_DNN", requires="sycl", engines=LLAMA_ONLY),
    _s("sycl_target", "sycl_target", "INTEL", "SYCL target", "GGML_SYCL_TARGET", "sycl", "GGML_SYCL_TARGET", requires="sycl"),
    _s("sycl_device_arch", "sycl_device_arch", "", "Device arch", "GGML_SYCL_DEVICE_ARCH", "sycl", "GGML_SYCL_DEVICE_ARCH", requires="sycl", engines=LLAMA_ONLY),

    # ── OpenCL ────────────────────────────────────────────────
    _b("opencl_profiling", "enable_opencl_profiling", False, "Profiling", "GGML_OPENCL_PROFILING", "opencl", "GGML_OPENCL_PROFILING", requires="opencl", engines=LLAMA_ONLY),
    _b("opencl_embed_kernels", "enable_opencl_embed_kernels", True, "Embed kernels", "GGML_OPENCL_EMBED_KERNELS", "opencl", "GGML_OPENCL_EMBED_KERNELS", requires="opencl", engines=LLAMA_ONLY),
    _b("opencl_adreno_kernels", "enable_opencl_adreno_kernels", True, "Adreno kernels", "GGML_OPENCL_USE_ADRENO_KERNELS", "opencl", "GGML_OPENCL_USE_ADRENO_KERNELS", requires="opencl", engines=LLAMA_ONLY),
    _s("opencl_target_version", "opencl_target_version", "300", "API target version", "GGML_OPENCL_TARGET_VERSION", "opencl", "GGML_OPENCL_TARGET_VERSION", requires="opencl", engines=LLAMA_ONLY),

    # ── WebGPU ────────────────────────────────────────────────
    _b("webgpu_debug", "enable_webgpu_debug", False, "Debug output", "GGML_WEBGPU_DEBUG", "webgpu", "GGML_WEBGPU_DEBUG", requires="webgpu", engines=LLAMA_ONLY),
    _b("webgpu_cpu_profile", "enable_webgpu_cpu_profile", False, "CPU profiling", "GGML_WEBGPU_CPU_PROFILE", "webgpu", "GGML_WEBGPU_CPU_PROFILE", requires="webgpu", engines=LLAMA_ONLY),
    _b("webgpu_gpu_profile", "enable_webgpu_gpu_profile", False, "GPU profiling", "GGML_WEBGPU_GPU_PROFILE", "webgpu", "GGML_WEBGPU_GPU_PROFILE", requires="webgpu", engines=LLAMA_ONLY),
    _b("webgpu_jspi", "enable_webgpu_jspi", True, "JSPI", "GGML_WEBGPU_JSPI", "webgpu", "GGML_WEBGPU_JSPI", requires="webgpu", engines=LLAMA_ONLY),

    # ── CPU / BLAS ────────────────────────────────────────────
    _b("cpu", "enable_cpu", True, "CPU backend", "GGML_CPU", "cpu_accel", "GGML_CPU", engines=LLAMA_ONLY),
    _b("openmp", "enable_openmp", True, "OpenMP", "GGML_OPENMP", "cpu_accel", "GGML_OPENMP"),
    _b("accelerate", "enable_accelerate", True, "Accelerate", "GGML_ACCELERATE — Apple Accelerate", "cpu_accel", "GGML_ACCELERATE"),
    _b("llamafile", "enable_llamafile", True, "llamafile", "GGML_LLAMAFILE", "cpu_accel", "GGML_LLAMAFILE", engines=LLAMA_ONLY),
    _b("cpu_hbm", "enable_cpu_hbm", False, "CPU HBM", "GGML_CPU_HBM — memkind", "cpu_accel", "GGML_CPU_HBM"),
    _b("cpu_repack", "enable_cpu_repack", True, "CPU repack", "GGML_CPU_REPACK — Q4_0→Q4_X_X at runtime", "cpu_accel", "GGML_CPU_REPACK", engines=LLAMA_ONLY),
    _b("cpu_kleidiai", "enable_cpu_kleidiai", False, "KleidiAI", "GGML_CPU_KLEIDIAI — Arm optimized kernels", "cpu_accel", "GGML_CPU_KLEIDIAI", engines=LLAMA_ONLY),
    _s("blas_vendor", "blas_vendor", "OpenBLAS", "BLAS vendor", "GGML_BLAS_VENDOR", "cpu_accel", requires="blas", special="blas_vendor", engines=LLAMA_ONLY),

    # ── Artifacts ─────────────────────────────────────────────
    _b("build_common", "build_common", True, "Common lib", "LLAMA_BUILD_COMMON", "artifacts", "LLAMA_BUILD_COMMON", engines=LLAMA_ONLY),
    _b("build_tests", "build_tests", True, "Tests", "LLAMA_BUILD_TESTS", "artifacts", "LLAMA_BUILD_TESTS"),
    _b("build_tools", "build_tools", True, "Tools", "LLAMA_BUILD_TOOLS", "artifacts", "LLAMA_BUILD_TOOLS", engines=LLAMA_ONLY),
    _b("build_examples", "build_examples", True, "Examples", "LLAMA_BUILD_EXAMPLES", "artifacts", "LLAMA_BUILD_EXAMPLES"),
    _b("build_server", "build_server", True, "Server", "LLAMA_BUILD_SERVER (required for serving)", "artifacts", "LLAMA_BUILD_SERVER"),
    _b("build_app", "build_app", True, "Unified app", "LLAMA_BUILD_APP", "artifacts", "LLAMA_BUILD_APP", engines=LLAMA_ONLY),
    _b("build_ui", "build_ui", True, "Embedded Web UI", "LLAMA_BUILD_UI", "artifacts", "LLAMA_BUILD_UI", engines=LLAMA_ONLY),
    _b("use_prebuilt_ui", "use_prebuilt_ui", True, "Prebuilt UI", "LLAMA_USE_PREBUILT_UI", "artifacts", "LLAMA_USE_PREBUILT_UI", engines=LLAMA_ONLY),
    _b("build_mtmd", "build_mtmd", False, "Standalone mtmd", "LLAMA_BUILD_MTMD — multimodal lib without full tools", "artifacts", "LLAMA_BUILD_MTMD", engines=LLAMA_ONLY),
    _b("install_tools", "install_tools", True, "Install tools", "LLAMA_TOOLS_INSTALL", "artifacts", "LLAMA_TOOLS_INSTALL", engines=LLAMA_ONLY),
    _b("install_tests", "install_tests", True, "Install tests", "LLAMA_TESTS_INSTALL", "artifacts", "LLAMA_TESTS_INSTALL", engines=LLAMA_ONLY),
    _b("openssl", "enable_openssl", True, "OpenSSL", "LLAMA_OPENSSL — HTTPS support", "artifacts", "LLAMA_OPENSSL", engines=LLAMA_ONLY),
    _b("subprocess", "enable_subprocess", True, "Subprocess", "LLAMA_SUBPROCESS — server tools / router", "artifacts", "LLAMA_SUBPROCESS", engines=LLAMA_ONLY),
    _b("llguidance", "enable_llguidance", False, "LLGuidance", "LLAMA_LLGUIDANCE — structured output", "artifacts", "LLAMA_LLGUIDANCE"),

    # ── GGML general ──────────────────────────────────────────
    _b("native", "enable_native", True, "Native CPU", "GGML_NATIVE — optimize for this host", "ggml", "GGML_NATIVE"),
    _b("backend_dl", "enable_backend_dl", False, "Backend DL", "GGML_BACKEND_DL — backends as shared libs", "ggml", "GGML_BACKEND_DL", engines=LLAMA_ONLY),
    _b("cpu_all_variants", "enable_cpu_all_variants", False, "CPU all variants", "GGML_CPU_ALL_VARIANTS (requires Backend DL)", "ggml", "GGML_CPU_ALL_VARIANTS", engines=LLAMA_ONLY),
    _b("lto", "enable_lto", False, "LTO", "GGML_LTO — link-time optimization", "ggml", "GGML_LTO"),
    _b("ccache", "enable_ccache", True, "ccache", "GGML_CCACHE", "ggml", "GGML_CCACHE"),
    _b("static", "enable_static", False, "Static link", "GGML_STATIC", "ggml", "GGML_STATIC"),
    _b("sched_no_realloc", "enable_sched_no_realloc", False, "No realloc", "GGML_SCHED_NO_REALLOC (debug)", "ggml", "GGML_SCHED_NO_REALLOC", engines=LLAMA_ONLY),
    _s("backend_dir", "backend_dir", "", "Backend DL directory", "GGML_BACKEND_DIR", "ggml", "GGML_BACKEND_DIR", engines=LLAMA_ONLY),
    _s("sched_max_copies", "sched_max_copies", "", "Sched max copies", "GGML_SCHED_MAX_COPIES (blank = upstream default)", "ggml", "GGML_SCHED_MAX_COPIES"),
    _s("cpu_arm_arch", "cpu_arm_arch", "", "ARM CPU arch", "GGML_CPU_ARM_ARCH", "ggml", "GGML_CPU_ARM_ARCH", engines=LLAMA_ONLY),
    _s("cpu_powerpc_cputype", "cpu_powerpc_cputype", "", "PowerPC CPU type", "GGML_CPU_POWERPC_CPUTYPE", "ggml", "GGML_CPU_POWERPC_CPUTYPE", engines=LLAMA_ONLY),

    # ── CPU ISA (relevant when native=OFF for portable builds) ─
    _b("sse42", "enable_sse42", False, "SSE 4.2", "GGML_SSE42", "cpu_isa", "GGML_SSE42", engines=LLAMA_ONLY),
    _b("avx", "enable_avx", False, "AVX", "GGML_AVX", "cpu_isa", "GGML_AVX"),
    _b("avx_vnni", "enable_avx_vnni", False, "AVX-VNNI", "GGML_AVX_VNNI / GGML_AVXVNNI", "cpu_isa", special="avx_vnni"),
    _b("avx2", "enable_avx2", False, "AVX2", "GGML_AVX2", "cpu_isa", "GGML_AVX2"),
    _b("bmi2", "enable_bmi2", False, "BMI2", "GGML_BMI2", "cpu_isa", "GGML_BMI2", engines=LLAMA_ONLY),
    _b("fma", "enable_fma", False, "FMA", "GGML_FMA", "cpu_isa", "GGML_FMA"),
    _b("f16c", "enable_f16c", False, "F16C", "GGML_F16C", "cpu_isa", "GGML_F16C"),
    _b("avx512", "enable_avx512", False, "AVX512F", "GGML_AVX512", "cpu_isa", "GGML_AVX512"),
    _b("avx512_vbmi", "enable_avx512_vbmi", False, "AVX512-VBMI", "GGML_AVX512_VBMI", "cpu_isa", "GGML_AVX512_VBMI"),
    _b("avx512_vnni", "enable_avx512_vnni", False, "AVX512-VNNI", "GGML_AVX512_VNNI", "cpu_isa", "GGML_AVX512_VNNI"),
    _b("avx512_bf16", "enable_avx512_bf16", False, "AVX512-BF16", "GGML_AVX512_BF16", "cpu_isa", "GGML_AVX512_BF16"),
    _b("amx_tile", "enable_amx_tile", False, "AMX-TILE", "GGML_AMX_TILE", "cpu_isa", "GGML_AMX_TILE", engines=LLAMA_ONLY),
    _b("amx_int8", "enable_amx_int8", False, "AMX-INT8", "GGML_AMX_INT8", "cpu_isa", "GGML_AMX_INT8", engines=LLAMA_ONLY),
    _b("amx_bf16", "enable_amx_bf16", False, "AMX-BF16", "GGML_AMX_BF16", "cpu_isa", "GGML_AMX_BF16", engines=LLAMA_ONLY),
    _b("sve", "enable_sve", False, "SVE", "GGML_SVE — Arm SVE (ik_llama)", "cpu_isa", "GGML_SVE", engines=IK_ONLY),
    _b("lasx", "enable_lasx", True, "LASX", "GGML_LASX — LoongArch", "cpu_isa", "GGML_LASX"),
    _b("lsx", "enable_lsx", True, "LSX", "GGML_LSX — LoongArch", "cpu_isa", "GGML_LSX"),
    _b("rvv", "enable_rvv", True, "RVV", "GGML_RVV — RISC-V", "cpu_isa", "GGML_RVV", engines=LLAMA_ONLY),
    _b("rv_zfh", "enable_rv_zfh", True, "RISC-V Zfh", "GGML_RV_ZFH", "cpu_isa", "GGML_RV_ZFH", engines=LLAMA_ONLY),
    _b("rv_zvfh", "enable_rv_zvfh", True, "RISC-V Zvfh", "GGML_RV_ZVFH", "cpu_isa", "GGML_RV_ZVFH", engines=LLAMA_ONLY),
    _b("rv_zicbop", "enable_rv_zicbop", True, "RISC-V Zicbop", "GGML_RV_ZICBOP", "cpu_isa", "GGML_RV_ZICBOP", engines=LLAMA_ONLY),
    _b("rv_zihintpause", "enable_rv_zihintpause", True, "RISC-V Zihintpause", "GGML_RV_ZIHINTPAUSE", "cpu_isa", "GGML_RV_ZIHINTPAUSE", engines=LLAMA_ONLY),
    _b("rv_zvfbfwma", "enable_rv_zvfbfwma", False, "RISC-V Zvfbfwma", "GGML_RV_ZVFBFWMA", "cpu_isa", "GGML_RV_ZVFBFWMA", engines=LLAMA_ONLY),
    _b("xtheadvector", "enable_xtheadvector", False, "XTheadVector", "GGML_XTHEADVECTOR", "cpu_isa", "GGML_XTHEADVECTOR", engines=LLAMA_ONLY),
    _b("vxe", "enable_vxe", False, "VXE", "GGML_VXE — IBM Z", "cpu_isa", "GGML_VXE", engines=LLAMA_ONLY),

    # ── Debug / sanitizers ────────────────────────────────────
    _b("all_warnings", "enable_all_warnings", True, "All warnings", "LLAMA_ALL_WARNINGS / GGML_ALL_WARNINGS", "debug", "LLAMA_ALL_WARNINGS"),
    _b("fatal_warnings", "enable_fatal_warnings", False, "Fatal warnings", "LLAMA_FATAL_WARNINGS (-Werror)", "debug", "LLAMA_FATAL_WARNINGS"),
    _b("sanitize_thread", "enable_sanitize_thread", False, "Thread sanitizer", "LLAMA_SANITIZE_THREAD", "debug", "LLAMA_SANITIZE_THREAD"),
    _b("sanitize_address", "enable_sanitize_address", False, "Address sanitizer", "LLAMA_SANITIZE_ADDRESS", "debug", "LLAMA_SANITIZE_ADDRESS"),
    _b("sanitize_undefined", "enable_sanitize_undefined", False, "Undefined sanitizer", "LLAMA_SANITIZE_UNDEFINED", "debug", "LLAMA_SANITIZE_UNDEFINED"),
    _b("gprof", "enable_gprof", False, "gprof", "GGML_GPROF", "debug", "GGML_GPROF"),

    # ── Advanced freeform (not cmake-mapped 1:1) ──────────────
    _s("custom_cmake_args", "custom_cmake_args", "", "Custom CMake args", "Extra args appended to cmake (shlex)", "advanced", special="custom_cmake"),
    _s("cflags", "cflags", "", "CFLAGS", "Passed via CFLAGS env", "advanced", special="cflags"),
    _s("cxxflags", "cxxflags", "", "CXXFLAGS", "Passed via CXXFLAGS env", "advanced", special="cxxflags"),
)
# fmt: on

# build_type is separate (not a GGML_/LLAMA_ toggle)
BUILD_TYPE_DEFAULT = "Release"
BUILD_TYPE_VALUES = ("Debug", "Release", "RelWithDebInfo", "MinSizeRel")

# Keys that are UI aliases / not BuildConfig fields themselves for cmake
_ALIAS_KEYS = frozenset({"openblas"})

OPTION_BY_KEY: Dict[str, BuildOptionDef] = {o.key: o for o in BUILD_OPTIONS}
OPTION_BY_FIELD: Dict[str, BuildOptionDef] = {
    o.field: o for o in BUILD_OPTIONS if o.key not in _ALIAS_KEYS
}


def default_build_settings() -> Dict[str, Any]:
    """UI/API-shaped defaults (short keys)."""
    out: Dict[str, Any] = {"build_type": BUILD_TYPE_DEFAULT}
    for opt in BUILD_OPTIONS:
        out[opt.key] = opt.default
    return out


def build_config_field_defaults() -> Dict[str, Any]:
    """BuildConfig field-name defaults (enable_* / build_*)."""
    out: Dict[str, Any] = {"build_type": BUILD_TYPE_DEFAULT}
    for opt in BUILD_OPTIONS:
        if opt.key in _ALIAS_KEYS:
            continue
        out[opt.field] = opt.default
    out["env_vars"] = {}
    return out


def coerce_build_settings(settings: Optional[dict]) -> Dict[str, Any]:
    """Normalize a settings dict to the full UI/API shape."""
    base = default_build_settings()
    if not isinstance(settings, dict):
        return base

    def _bool(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        return bool(v)

    def _str(v: Any, default: str = "") -> str:
        return str(v).strip() if v is not None else default

    out = dict(base)

    build_type = _str(settings.get("build_type"), base["build_type"])
    if build_type not in BUILD_TYPE_VALUES:
        build_type = base["build_type"]
    out["build_type"] = build_type

    for opt in BUILD_OPTIONS:
        raw = settings.get(opt.key, base[opt.key])
        if opt.kind == "bool":
            out[opt.key] = _bool(raw)
        elif opt.kind == "enum" and opt.enum_values:
            val = _str(raw, str(opt.default))
            out[opt.key] = val if val in opt.enum_values else opt.default
        else:
            out[opt.key] = _str(raw, str(opt.default) if opt.default is not None else "")

    # Legacy: openblas alone → enable blas
    if out.get("openblas") and not settings.get("blas", None) and not out.get("blas"):
        out["blas"] = True
        if not settings.get("blas_vendor"):
            out["blas_vendor"] = "OpenBLAS"

    return out


def stored_config_to_settings(raw: Optional[dict]) -> Dict[str, Any]:
    """Normalize a version-row ``build_config`` (UI keys or BuildConfig fields) to UI keys."""
    if not isinstance(raw, dict):
        return default_build_settings()
    mapped: Dict[str, Any] = {}
    if raw.get("build_type") is not None:
        mapped["build_type"] = raw.get("build_type")
    for opt in BUILD_OPTIONS:
        if opt.key in raw:
            mapped[opt.key] = raw[opt.key]
        elif opt.field in raw:
            mapped[opt.key] = raw[opt.field]
    for key in ("custom_cmake_args", "cflags", "cxxflags", "cuda_architectures"):
        if key in raw and key not in mapped:
            mapped[key] = raw[key]
    return coerce_build_settings(mapped)


def settings_to_field_kwargs(settings: dict) -> Dict[str, Any]:
    """Map coerced settings keys → BuildConfig constructor kwargs."""
    normalized = coerce_build_settings(settings)
    kwargs: Dict[str, Any] = {"build_type": normalized["build_type"]}
    for opt in BUILD_OPTIONS:
        if opt.key in _ALIAS_KEYS:
            continue
        kwargs[opt.field] = normalized[opt.key]

    # Legacy openblas → also set enable_openblas for callers that read it
    kwargs["enable_openblas"] = bool(normalized.get("openblas")) or (
        bool(normalized.get("blas"))
        and str(normalized.get("blas_vendor", "")).lower() == "openblas"
    )
    if kwargs["enable_openblas"] and not kwargs.get("enable_blas"):
        kwargs["enable_blas"] = True
        if not kwargs.get("blas_vendor"):
            kwargs["blas_vendor"] = "OpenBLAS"
    return kwargs


def catalog_for_ui(engine: Optional[str] = None) -> Dict[str, Any]:
    """Payload for GET build-options: categories + defaults (optionally filtered by engine)."""
    eng = normalize_engine_id(engine)
    by_cat: Dict[str, List[dict]] = {c["id"]: [] for c in CATEGORIES}
    for opt in BUILD_OPTIONS:
        if opt.key == "openblas":
            # Hide legacy alias from UI; blas covers it
            continue
        if eng and eng not in opt.engines:
            continue
        entry = {
            "key": opt.key,
            "type": opt.kind,
            "label": opt.label,
            "desc": opt.desc,
            "default": opt.default,
            "requires": opt.requires,
            "cmake": opt.cmake,
            "primary": opt.category != "backends" or opt.key in PRIMARY_BACKEND_KEYS,
            "engines": sorted(opt.engines),
        }
        if opt.enum_values:
            entry["enum_values"] = list(opt.enum_values)
        by_cat.setdefault(opt.category, []).append(entry)

    categories = []
    for cat in CATEGORIES:
        cat_engines = CATEGORY_ENGINES.get(cat["id"])
        if eng and cat_engines and eng not in cat_engines:
            continue
        options = by_cat.get(cat["id"]) or []
        if not options:
            continue
        categories.append(
            {
                "id": cat["id"],
                "label": cat["label"],
                "requires": CATEGORY_REQUIRES.get(cat["id"]),
                "collapsed": bool(cat.get("collapsed", True)),
                "options": options,
            }
        )

    return {
        "engine": eng,
        "categories": categories,
        "defaults": default_build_settings(),
        "build_types": list(BUILD_TYPE_VALUES),
    }


def append_generic_cmake_flags(
    cmake_args: List[str],
    build_config: Any,
    *,
    set_flag,
    engine: Optional[str] = None,
) -> None:
    """
    Emit -D flags for options applicable to the target engine.
    Engine-specific flag names (HIP vs HIPBLAS, CUDA_GRAPHS vs CUDA_USE_GRAPHS,
    AVX_VNNI vs AVXVNNI) are handled via special=.
    """
    eng = normalize_engine_id(engine) or "llama_cpp"
    skip_emit = {
        "cuda",
        "blas",
        "openblas_alias",
        "cuda_fa_all",
        "cuda_arch",
        "blas_vendor",
        "custom_cmake",
        "cflags",
        "cxxflags",
    }
    for opt in BUILD_OPTIONS:
        if opt.key in _ALIAS_KEYS:
            continue
        if eng not in opt.engines:
            continue

        if opt.special == "hip":
            flag = "GGML_HIPBLAS" if eng == "ik_llama" else "GGML_HIP"
            set_flag(flag, bool(getattr(build_config, opt.field, False)))
            continue
        if opt.special == "cuda_graphs":
            flag = "GGML_CUDA_USE_GRAPHS" if eng == "ik_llama" else "GGML_CUDA_GRAPHS"
            parent_on = bool(getattr(build_config, "enable_cuda", False))
            value = bool(getattr(build_config, opt.field, opt.default))
            set_flag(flag, value if parent_on else False)
            continue
        if opt.special == "avx_vnni":
            flag = "GGML_AVXVNNI" if eng == "ik_llama" else "GGML_AVX_VNNI"
            set_flag(flag, bool(getattr(build_config, opt.field, False)))
            continue

        if not opt.cmake or opt.special in skip_emit:
            continue
        value = getattr(build_config, opt.field, opt.default)
        if opt.kind == "bool":
            if opt.requires:
                parent = OPTION_BY_KEY.get(opt.requires)
                if parent:
                    parent_val = getattr(build_config, parent.field, False)
                    if not parent_val:
                        set_flag(opt.cmake, False)
                        continue
            set_flag(opt.cmake, bool(value))
        else:
            if opt.requires:
                parent = OPTION_BY_KEY.get(opt.requires)
                if parent and not getattr(build_config, parent.field, False):
                    continue
            text = str(value or "").strip()
            if text:
                cmake_args.append(f"-D{opt.cmake}={text}")
