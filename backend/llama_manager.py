import os
import re
import subprocess
import requests
import json
import shutil
import time
import multiprocessing
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field, asdict
import asyncio
import aiohttp
from backend.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class BuildConfig:
    """Configuration for building llama.cpp from source"""
    build_type: str = "Release"  # Debug, Release, RelWithDebInfo
    
    # GPU backends
    enable_cuda: bool = False
    enable_vulkan: bool = False
    enable_metal: bool = False
    enable_openblas: bool = False
    enable_flash_attention: bool = False  # Enables -DGGML_CUDA_FA_ALL_QUANTS=ON

    # Build artifacts
    build_common: bool = True
    build_tests: bool = True
    build_tools: bool = True
    build_examples: bool = True
    build_server: bool = True
    install_tools: bool = True

    # GGML advanced options
    enable_backend_dl: bool = False
    enable_cpu_all_variants: bool = False
    enable_lto: bool = False
    enable_native: bool = True
    
    # Advanced options
    custom_cmake_args: str = ""
    cuda_architectures: str = ""
    cflags: str = ""
    cxxflags: str = ""
    
    # Environment variables
    env_vars: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.normalize()

    def normalize(self):
        """
        Normalize combinations that are known to be incompatible in upstream
        CMake configuration. See ggml/src/ggml-cpu/CMakeLists.txt.
        """
        if self.enable_backend_dl:
            if self.enable_native:
                logger.warning(
                    "GGML_BACKEND_DL is enabled; disabling GGML_NATIVE to avoid CMake build failure."
                )
                self.enable_native = False
            if not self.enable_cpu_all_variants:
                logger.info(
                    "GGML_BACKEND_DL is enabled; enabling GGML_CPU_ALL_VARIANTS to ensure CPU variants are available."
                )
                self.enable_cpu_all_variants = True


class LlamaManager:
    # Repository URLs
    LLAMA_CPP_REPO = "https://github.com/ggerganov/llama.cpp.git"
    IK_LLAMA_CPP_REPO = "https://github.com/ikawrakow/ik_llama.cpp.git"
    
    REPOSITORY_SOURCES = {
        "llama.cpp": LLAMA_CPP_REPO,
        "ik_llama.cpp": IK_LLAMA_CPP_REPO
    }
    
    def __init__(self):
        self.llama_dir = "data/llama-cpp"
        os.makedirs(self.llama_dir, exist_ok=True)
        self._cached_cuda_architectures: Optional[str] = None

    def _check_cuda_toolkit_available(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check if CUDA Toolkit is available on the system.
        
        Returns:
            Tuple of (is_available, cuda_root, error_message)
            - is_available: True if CUDA Toolkit is found
            - cuda_root: Path to CUDA root directory if found, None otherwise
            - error_message: Error message if not available, None otherwise
        """
        # First, check CUDA installer for installations in data directory
        try:
            from backend.cuda_installer import get_cuda_installer
            installer = get_cuda_installer()
            cuda_path = installer._get_cuda_path()
            if cuda_path and os.path.exists(cuda_path):
                # Verify it has nvcc
                nvcc_path = os.path.join(cuda_path, "bin", "nvcc")
                if os.path.exists(nvcc_path):
                    # Verify toolkit completeness
                    is_complete, missing = self._verify_cuda_toolkit_complete(cuda_path)
                    if is_complete:
                        return (True, cuda_path, None)
                    else:
                        logger.warning(f"CUDA toolkit at {cuda_path} is incomplete, missing: {missing}")
        except (ImportError, Exception) as e:
            # If CUDA installer is not available or fails, continue with standard checks
            pass
        
        env = os.environ.copy()
        possible_cuda_roots = [
            env.get("CUDA_PATH"),
            env.get("CUDA_HOME"),
            "/usr/local/cuda",
            "/usr/local/cuda-12.9",
            "/usr/local/cuda-12.8",
            "/usr/local/cuda-12.7",
            "/usr/local/cuda-12.6",
            "/usr/local/cuda-12.5",
            "/usr/local/cuda-12.4",
            "/usr/local/cuda-12.3",
            "/usr/local/cuda-12.2",
            "/usr/local/cuda-12.1",
            "/usr/local/cuda-12.0",
            "/usr/local/cuda-11.9",
            "/usr/local/cuda-11.8",
        ]
        
        # Filter out None values and check if paths exist
        for cuda_root in possible_cuda_roots:
            if not cuda_root or not os.path.exists(cuda_root):
                continue
            
            # Check for nvcc compiler
            nvcc_path = os.path.join(cuda_root, "bin", "nvcc")
            if not os.path.exists(nvcc_path):
                # On Windows, nvcc might be in a different location
                if os.name == 'nt':
                    nvcc_path = os.path.join(cuda_root, "bin", "nvcc.exe")
                    if not os.path.exists(nvcc_path):
                        continue
                else:
                    continue
            
            # Verify toolkit completeness (includes headers, libs, etc.)
            is_complete, missing = self._verify_cuda_toolkit_complete(cuda_root)
            if is_complete:
                return (True, cuda_root, None)
            else:
                logger.warning(f"CUDA toolkit at {cuda_root} is incomplete, missing: {missing}")
        
        # Try to find nvcc in PATH as a fallback
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # nvcc found in PATH, try to determine CUDA root
                nvcc_path = shutil.which("nvcc")
                if nvcc_path:
                    # nvcc is typically in <CUDA_ROOT>/bin/nvcc
                    potential_root = os.path.dirname(os.path.dirname(nvcc_path))
                    if os.path.exists(potential_root):
                        is_complete, missing = self._verify_cuda_toolkit_complete(potential_root)
                        if is_complete:
                            return (True, potential_root, None)
                        else:
                            logger.warning(f"CUDA toolkit at {potential_root} is incomplete, missing: {missing}")
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        
        error_msg = (
            "CUDA Toolkit not found or incomplete. Please either:\n"
            "1. Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads\n"
            "2. Set CUDA_PATH environment variable to your CUDA installation directory\n"
            "3. Disable CUDA in build configuration (set enable_cuda: false)"
        )
        return (False, None, error_msg)

    def _verify_cuda_toolkit_complete(self, cuda_root: str) -> Tuple[bool, List[str]]:
        """
        Verify that CUDA toolkit has all required components for building.
        
        Returns:
            Tuple of (is_complete, missing_components)
        """
        missing = []
        
        # Check for nvcc compiler
        nvcc_name = "nvcc.exe" if os.name == 'nt' else "nvcc"
        nvcc_path = os.path.join(cuda_root, "bin", nvcc_name)
        if not os.path.exists(nvcc_path):
            missing.append("nvcc compiler")
        
        # Check for CUDA headers (required for CUDA language support)
        include_dir = os.path.join(cuda_root, "include")
        if not os.path.exists(include_dir):
            missing.append("include directory")
        else:
            # Check for key headers
            key_headers = ["cuda.h", "cuda_runtime.h"]
            for header in key_headers:
                header_path = os.path.join(include_dir, header)
                if not os.path.exists(header_path):
                    missing.append(f"header: {header}")
        
        # Check for CUDA libraries
        lib_dirs = ["lib64", "lib"] if os.name != 'nt' else ["lib/x64", "lib"]
        has_libs = False
        cuda_lib_dir = None
        for lib_dir in lib_dirs:
            lib_path = os.path.join(cuda_root, lib_dir)
            if os.path.exists(lib_path):
                cuda_lib_dir = lib_path
                try:
                    lib_files = os.listdir(lib_path)
                    # Check for essential libraries
                    if os.name == 'nt':
                        if any("cudart" in f for f in lib_files):
                            has_libs = True
                            break
                    else:
                        if any("libcudart" in f for f in lib_files):
                            has_libs = True
                            break
                except OSError:
                    pass
        
        if not has_libs:
            missing.append("CUDA runtime library (cudart)")
        
        # Check for version.txt or version.json (indicates full toolkit)
        version_files = ["version.txt", "version.json"]
        has_version = any(os.path.exists(os.path.join(cuda_root, vf)) for vf in version_files)
        if not has_version:
            # Not critical, just log
            logger.debug(f"CUDA toolkit at {cuda_root} missing version file (not critical)")
        
        # Check for NCCL (optional but recommended for multi-GPU)
        # NCCL can be in the CUDA directory or system directories
        nccl_found = False
        nccl_search_paths = [
            os.path.join(cuda_root, "include", "nccl.h"),
            os.path.join(cuda_root, "include", "nccl_net.h"),
            "/usr/include/nccl.h",
            "/usr/local/include/nccl.h",
        ]
        for nccl_path in nccl_search_paths:
            if os.path.exists(nccl_path):
                nccl_found = True
                break
        
        # Also check for NCCL library
        if not nccl_found:
            nccl_lib_paths = [
                os.path.join(cuda_root, "lib64"),
                os.path.join(cuda_root, "lib"),
                "/usr/lib/x86_64-linux-gnu",
                "/usr/local/lib",
            ]
            for lib_dir in nccl_lib_paths:
                if os.path.exists(lib_dir):
                    try:
                        lib_files = os.listdir(lib_dir)
                        if any("libnccl" in f for f in lib_files):
                            nccl_found = True
                            break
                    except OSError:
                        pass
        
        if not nccl_found:
            # NCCL is optional, just log a warning
            logger.info("NCCL not found - multi-GPU support may be limited. Build will continue.")
        else:
            logger.debug("NCCL found - multi-GPU support available")
        
        return (len(missing) == 0, missing)

    def _get_cmake_version(self) -> Optional[Tuple[int, int, int]]:
        """Get CMake version as tuple (major, minor, patch)."""
        try:
            result = subprocess.run(
                ["cmake", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Parse "cmake version X.Y.Z"
                match = re.search(r'cmake version (\d+)\.(\d+)\.(\d+)', result.stdout)
                if match:
                    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except Exception:
            pass
        return None

    def _get_cuda_version(self, nvcc_path: str) -> Optional[Tuple[int, int]]:
        """Get CUDA version from nvcc as tuple (major, minor)."""
        try:
            result = subprocess.run(
                [nvcc_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Parse "release X.Y"
                match = re.search(r'release (\d+)\.(\d+)', result.stdout)
                if match:
                    return (int(match.group(1)), int(match.group(2)))
        except Exception:
            pass
        return None

    async def _detect_cuda_architectures(self) -> Optional[str]:
        """
        Determine CUDA architectures for the current environment by querying GPU capabilities.
        Results are cached because GPU detection can be relatively expensive.
        """
        if self._cached_cuda_architectures is not None:
            return self._cached_cuda_architectures

        try:
            from backend.gpu_detector import get_gpu_info
        except ImportError:
            return None

        try:
            gpu_info = await get_gpu_info()
        except Exception as exc:
            logger.debug(f"Failed to detect GPU architectures: {exc}")
            return None

        if gpu_info.get("vendor") != "nvidia":
            return None

        architectures = []
        for gpu in gpu_info.get("gpus", []):
            compute_capability = gpu.get("compute_capability")
            if not compute_capability:
                continue
            parts = compute_capability.replace(" ", "").split(".")
            if len(parts) != 2:
                continue
            major, minor = parts
            if major.isdigit() and minor.isdigit():
                architectures.append(f"{major}{minor}")

        if not architectures:
            return None

        # Ensure uniqueness and deterministic order
        unique_arches = sorted(set(architectures))
        detected = ";".join(unique_arches)
        self._cached_cuda_architectures = detected
        return detected
    
    def _fetch_release(self, tag_name: str) -> Dict:
        """Fetch release metadata for a tag."""
        response = requests.get(
            f"https://api.github.com/repos/ggerganov/llama.cpp/releases/tags/{tag_name}",
            allow_redirects=True,
        )
        response.raise_for_status()
        return response.json()
    
    def _tokenize_asset_name(self, asset_name: str) -> List[str]:
        return [token for token in re.split(r"[.\-_\s]+", asset_name.lower()) if token]
    
    def _is_asset_compatible(self, asset_name: str) -> Tuple[bool, Optional[str]]:
        tokens = self._tokenize_asset_name(asset_name)
        
        if not tokens:
            return False, "Unable to determine artifact metadata"
        
        exclusion_tokens = {
            "windows": "Windows artifacts are not compatible with the container",
            "win": "Windows artifacts are not compatible with the container",
            "darwin": "macOS artifacts are not compatible with the container",
            "mac": "macOS artifacts are not compatible with the container",
            "osx": "macOS artifacts are not compatible with the container",
            "android": "Android artifacts are not compatible with the container",
            "ios": "iOS artifacts are not compatible with the container",
        }
        for token, reason in exclusion_tokens.items():
            if token in tokens:
                return False, reason
        
        if any(token in {"arm", "arm64", "aarch64"} for token in tokens):
            return False, "ARM builds are not supported by the container (expects x86_64)"
        
        if not any(token in {"linux", "ubuntu"} for token in tokens):
            return False, "Only Linux/Ubuntu artifacts are supported in the container"
        
        if not any(token in {"x64", "x86_64", "amd64"} for token in tokens):
            return False, "Only x86_64/amd64 artifacts are supported in the container"
        
        allowed_extensions = (".zip", ".tar.gz", ".tgz", ".tar.xz")
        if not asset_name.lower().endswith(allowed_extensions):
            return False, f"Unsupported archive format for artifact '{asset_name}'"
        
        return True, None
    
    def _extract_asset_features(self, asset_name: str) -> List[str]:
        tokens = self._tokenize_asset_name(asset_name)
        features = []
        
        feature_map = {
            "cuda": "CUDA",
            "vulkan": "Vulkan",
            "metal": "Metal",
            "opencl": "OpenCL",
            "hip": "HIP/ROCm",
            "rocm": "HIP/ROCm",
            "server": "llama-server",
            "cli": "CLI tools",
            "avx512": "AVX-512",
            "avx2": "AVX2",
            "avx": "AVX",
            "noavx": "Portable (no AVX)",
            "sse": "SSE optimizations",
            "openmp": "OpenMP",
            "full": "Full toolchain",
            "tools": "CLI tools",
            "examples": "Examples",
            "tests": "Tests",
            "gguf": "GGUF tooling",
        }
        
        seen = set()
        for token in tokens:
            label = None
            if token.startswith("cu") and len(token) >= 3 and token[2:].replace(".", "", 1).isdigit():
                label = f"CUDA {token[2:].upper()}"
            elif token in feature_map:
                label = feature_map[token]
            
            if label and label not in seen:
                features.append(label)
                seen.add(label)
        
        # If the artifact looks like a CPU build but no CPU features detected, mark as Portable CPU
        cpu_tokens = {"avx", "avx2", "avx512", "noavx"}
        if not seen.intersection({"AVX-512", "AVX2", "AVX", "Portable (no AVX)"}):
            if any(token in cpu_tokens for token in tokens):
                pass  # already captured
            else:
                features.append("CPU build")
        
        if not any("llama-server" == feature for feature in features):
            if "server" in tokens or "llama-server" in tokens:
                features.append("llama-server")
        
        return features
    
    def _collect_release_assets(self, release: Dict) -> Tuple[List[Dict], List[Dict], Dict[int, Dict]]:
        compatible_assets: List[Dict] = []
        skipped_assets: List[Dict] = []
        asset_lookup: Dict[int, Dict] = {}
        
        for asset in release.get("assets", []):
            name = asset.get("name", "")
            compatible, reason = self._is_asset_compatible(name)
            features = self._extract_asset_features(name)
            archive_type = "zip"
            lowered_name = name.lower()
            if lowered_name.endswith(".tar.gz") or lowered_name.endswith(".tgz"):
                archive_type = "tar.gz"
            elif lowered_name.endswith(".tar.xz"):
                archive_type = "tar.xz"
            
            asset_entry = {
                "id": asset.get("id"),
                "name": name,
                "size": asset.get("size"),
                "download_count": asset.get("download_count"),
                "created_at": asset.get("created_at"),
                "updated_at": asset.get("updated_at"),
                "features": features,
                "archive_type": archive_type,
                "compatible": compatible,
                "compatibility_reason": reason,
            }
            
            asset_lookup[asset_entry["id"]] = {
                "asset": asset,
                "metadata": asset_entry,
            }
            
            if compatible:
                asset_entry["compatibility_reason"] = None
                compatible_assets.append(asset_entry)
            else:
                skipped_assets.append(asset_entry)
        
        compatible_assets.sort(key=lambda item: item["name"])
        skipped_assets.sort(key=lambda item: item["name"])
        
        return compatible_assets, skipped_assets, asset_lookup
    
    def _strip_archive_extension(self, asset_name: str) -> str:
        name = asset_name
        lowered = asset_name.lower()
        compound_suffixes = [".tar.gz", ".tar.xz", ".tar.bz2"]
        simple_suffixes = [".tgz", ".zip"]
        
        for suffix in compound_suffixes:
            if lowered.endswith(suffix):
                return asset_name[: -len(suffix)]
        for suffix in simple_suffixes:
            if lowered.endswith(suffix):
                return asset_name[: -len(suffix)]
        
        base, _ = os.path.splitext(asset_name)
        return base
    
    def _slugify(self, value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
        return slug or "artifact"
    
    def _generate_version_name(self, tag_name: str, asset_metadata: Dict) -> str:
        asset_name = asset_metadata.get("name") or "artifact"
        base_name = self._strip_archive_extension(asset_name)
        slug = self._slugify(base_name)
        tag_slug = self._slugify(tag_name)
        
        if tag_slug:
            tokens = [token for token in slug.split("-") if token and token != tag_slug]
            if tokens:
                slug = "-".join(tokens)
            else:
                slug = ""
        
        if not slug:
            return tag_name
        
        return f"{tag_name}-{slug}"
    
    def _select_release_asset(
        self,
        tag_name: str,
        asset_id: Optional[int],
        compatible_assets: List[Dict],
        skipped_assets: List[Dict],
        asset_lookup: Dict[int, Dict]
    ) -> Tuple[Dict, Dict]:
        if asset_id is not None:
            record = asset_lookup.get(asset_id)
            if not record:
                raise Exception(f"Selected artifact (id={asset_id}) not found in release {tag_name}")
            metadata = record["metadata"]
            if not metadata.get("compatible", False):
                reason = metadata.get("compatibility_reason") or "Artifact is marked as incompatible"
                raise Exception(f"Selected artifact is not compatible with this container: {reason}")
            return metadata, record["asset"]
        
        if not compatible_assets:
            if skipped_assets:
                reasons = "; ".join(
                    f"{asset['name']}: {asset.get('compatibility_reason', 'Unknown reason')}"
                    for asset in skipped_assets
                )
                raise Exception(
                    f"No compatible artifacts were found for release {tag_name}. "
                    f"Skipped artifacts: {reasons}"
                )
            raise Exception(f"No artifacts are available for release {tag_name}")
        
        preferred_asset = None
        for asset in compatible_assets:
            features = [feature.lower() for feature in asset.get("features", [])]
            if any(feature == "llama-server" for feature in features):
                preferred_asset = asset
                break
        if not preferred_asset:
            preferred_asset = compatible_assets[0]
        
        return preferred_asset, asset_lookup[preferred_asset["id"]]["asset"]
    
    def _resolve_release_asset(
        self,
        tag_name: str,
        asset_id: Optional[int] = None
    ) -> Dict:
        release = self._fetch_release(tag_name)
        compatible_assets, skipped_assets, asset_lookup = self._collect_release_assets(release)
        selected_metadata, binary_asset = self._select_release_asset(
            tag_name,
            asset_id,
            compatible_assets,
            skipped_assets,
            asset_lookup
        )
        
        return {
            "release": release,
            "compatible_assets": compatible_assets,
            "skipped_assets": skipped_assets,
            "asset_lookup": asset_lookup,
            "selected_metadata": selected_metadata,
            "binary_asset": binary_asset,
        }
    
    def get_release_assets(self, tag_name: str) -> Dict:
        resolution = self._resolve_release_asset(tag_name)
        compatible_assets = resolution["compatible_assets"]
        skipped_assets = resolution["skipped_assets"]
        default_metadata = resolution["selected_metadata"]
        
        return {
            "tag_name": tag_name,
            "release_name": resolution["release"].get("name"),
            "release_published_at": resolution["release"].get("published_at"),
            "html_url": resolution["release"].get("html_url"),
            "assets": compatible_assets,
            "skipped_assets": skipped_assets,
            "default_asset_id": default_metadata.get("id") if default_metadata else None,
        }
    
    def get_release_install_preview(self, tag_name: str, asset_id: Optional[int] = None) -> Dict:
        resolution = self._resolve_release_asset(tag_name, asset_id)
        selected_metadata = resolution["selected_metadata"]
        version_name = self._generate_version_name(tag_name, selected_metadata)
        
        return {
            "tag_name": tag_name,
            "asset": selected_metadata,
            "asset_id": selected_metadata.get("id"),
            "version_name": version_name,
            "html_url": resolution["release"].get("html_url"),
        }
    
    def get_optimal_build_threads(self) -> int:
        """Get optimal number of threads for building based on CPU cores"""
        try:
            cpu_count = multiprocessing.cpu_count()
            # Use 75% of cores, minimum 1, maximum cpu_count
            optimal = max(1, min(cpu_count, int(cpu_count * 0.75)))
            return optimal
        except:
            return 1  # Fallback to single thread
    
    async def validate_build(self, binary_path: str, websocket_manager=None, task_id: str = None) -> bool:
        """Run basic validation on built binary"""
        try:
            # Test 1: Check binary exists and is executable
            if not os.path.exists(binary_path) or not os.access(binary_path, os.X_OK):
                return False
            
            # Test 2: Run --version command
            process = await asyncio.create_subprocess_exec(
                binary_path, "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
            
            if process.returncode != 0:
                return False
            
            # Test 3: Check for expected output (either "llama" or "version:" string)
            output = stdout.decode() + stderr.decode()
            if "llama" not in output.lower() and "version:" not in output.lower():
                logger.debug(f"Validation output: {output}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Build validation failed: {e}")
            return False
    
    async def install_release(
        self,
        tag_name: str,
        websocket_manager=None,
        task_id: str = None,
        asset_id: Optional[int] = None
    ) -> str:
        """Install llama.cpp from GitHub release with WebSocket progress updates"""
        try:
            # Stage 1: Get release info
            if websocket_manager and task_id:
                await websocket_manager.send_build_progress(
                    task_id=task_id,
                    stage="fetch",
                    progress=10,
                    message=f"Fetching release information for {tag_name}...",
                    log_lines=[f"Fetching release info for {tag_name}..."]
                )
            
            resolution = self._resolve_release_asset(tag_name, asset_id)
            selected_metadata = resolution["selected_metadata"]
            binary_asset = resolution["binary_asset"]
            
            if not binary_asset:
                raise Exception("Failed to select a release artifact to install")
            
            version_label = self._generate_version_name(tag_name, selected_metadata)
            
            logger.info(
                f"Selected release artifact '{binary_asset['name']}' "
                f"for installation (features: {selected_metadata.get('features', [])}) "
                f"-> version name '{version_label}'"
            )
            
            # Stage 2: Download binary
            if websocket_manager and task_id:
                await websocket_manager.send_build_progress(
                    task_id=task_id,
                    stage="download",
                    progress=30,
                    message=f"Downloading {binary_asset['name']}...",
                    log_lines=[f"Downloading {binary_asset['name']}..."]
                )
            
            # Create version directory
            version_dir_name = f"release-{version_label}"
            version_dir = os.path.join(self.llama_dir, version_dir_name)
            
            # Clean up existing directory if it exists
            if os.path.exists(version_dir):
                logger.info(f"Cleaning up existing directory: {version_dir}")
                shutil.rmtree(version_dir, ignore_errors=True)
                time.sleep(1)
            
            os.makedirs(version_dir, exist_ok=True)
            
            # Download the binary
            download_path = os.path.join(version_dir, binary_asset["name"])
            async with aiohttp.ClientSession() as session:
                async with session.get(binary_asset["browser_download_url"]) as response:
                    response.raise_for_status()
                    with open(download_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
            
            logger.info(f"Downloaded {binary_asset['name']} to {download_path}")
            
            expected_size = binary_asset.get("size")
            if expected_size:
                try:
                    downloaded_size = os.path.getsize(download_path)
                    if abs(downloaded_size - expected_size) > 2 * 1024 * 1024:  # 2 MB tolerance
                        logger.warning(
                            f"Downloaded artifact size ({downloaded_size} bytes) deviates from expected size "
                            f"({expected_size} bytes) for {binary_asset['name']}"
                        )
                except OSError as exc:
                    logger.warning(f"Unable to verify downloaded artifact size: {exc}")
            
            # Stage 3: Extract binary
            if websocket_manager and task_id:
                await websocket_manager.send_build_progress(
                    task_id=task_id,
                    stage="extract",
                    progress=70,
                    message=f"Extracting {binary_asset['name']}...",
                    log_lines=[f"Extracting {binary_asset['name']}..."]
                )
            
            # Extract the archive
            lowered_name = binary_asset["name"].lower()
            if lowered_name.endswith(".zip"):
                import zipfile
                with zipfile.ZipFile(download_path, 'r') as zip_ref:
                    zip_ref.extractall(version_dir)
            elif lowered_name.endswith(".tar.gz") or lowered_name.endswith(".tgz"):
                import tarfile
                with tarfile.open(download_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(version_dir)
            elif lowered_name.endswith(".tar.xz"):
                import tarfile
                with tarfile.open(download_path, 'r:xz') as tar_ref:
                    tar_ref.extractall(version_dir)
            else:
                raise Exception(f"Unsupported archive format for {binary_asset['name']}")
            
            # Remove the downloaded archive
            os.remove(download_path)
            
            # Stage 4: Find and verify executable
            if websocket_manager and task_id:
                await websocket_manager.send_build_progress(
                    task_id=task_id,
                    stage="verify",
                    progress=90,
                    message="Verifying installation...",
                    log_lines=["Searching for llama-server executable..."]
                )
            
            # Find the llama-server executable
            final_server_path = None
            for root, _, files in os.walk(version_dir):
                if "llama-server" in files:
                    final_server_path = os.path.join(root, "llama-server")
                    break
            
            if not final_server_path or not os.path.exists(final_server_path):
                raise Exception("llama-server executable not found after extraction.")
            
            # Make sure it's executable
            os.chmod(final_server_path, 0o755)
            
            logger.info(f"llama-server executable found and verified: {final_server_path}")
            if websocket_manager and task_id:
                await websocket_manager.send_build_progress(
                    task_id=task_id,
                    stage="verify",
                    progress=100,
                    message="Installation completed successfully!",
                    log_lines=[f"llama-server found at: {final_server_path}"]
                )
            
            asset_summary = {
                "id": selected_metadata.get("id"),
                "name": selected_metadata.get("name"),
                "features": selected_metadata.get("features", []),
                "archive_type": selected_metadata.get("archive_type"),
                "size": selected_metadata.get("size"),
                "download_count": selected_metadata.get("download_count"),
                "created_at": selected_metadata.get("created_at"),
                "updated_at": selected_metadata.get("updated_at"),
            }
            
            return {
                "binary_path": final_server_path,
                "asset": asset_summary,
                "version_name": version_label
            }
            
        except Exception as e:
            logger.error(f"Installation failed with error: {e}")
            if websocket_manager and task_id:
                try:
                    await websocket_manager.send_build_progress(
                        task_id=task_id,
                        stage="error",
                        progress=0,
                        message=f"Installation failed: {str(e)}",
                        log_lines=[f"Error: {str(e)}"]
                    )
                except Exception as ws_error:
                    logger.error(f"Failed to send error to WebSocket: {ws_error}")
            raise Exception(f"Failed to install release {tag_name}: {e}")
    
    async def build_source(self, commit_sha: str, patches: List[str] = None, build_config: BuildConfig = None, websocket_manager=None, task_id: str = None, repository_url: str = None, version_name: str = None) -> str:
        """Build llama.cpp from source following official documentation - simplified approach"""
        try:
            # Use default repository if not specified
            if repository_url is None:
                repository_url = self.LLAMA_CPP_REPO
            
            # Determine repository source name for logging
            repo_source_name = "llama.cpp"
            for source_name, repo_url in self.REPOSITORY_SOURCES.items():
                if repo_url == repository_url:
                    repo_source_name = source_name
                    break
            
            # Send initial progress
            if websocket_manager and task_id:
                await websocket_manager.send_build_progress(
                    task_id=task_id,
                    stage="init",
                    progress=0,
                    message=f"Starting simplified build process for {repo_source_name}...",
                    log_lines=[f"Building {repo_source_name} from {commit_sha}"]
                )
            
            # Use provided version_name or generate default (shouldn't happen, but fallback)
            if version_name is None:
                version_name = f"source-{commit_sha[:8]}"
                logger.warning(f"No version_name provided, using default: {version_name}")
            
            version_dir = os.path.join(self.llama_dir, version_name)
            
            # Don't clean up existing directory - let API handle uniqueness check
            # This allows multiple builds of the same commit with different names
            os.makedirs(version_dir, exist_ok=True)
            
            # Stage 1: Clone repository (simplified)
            if websocket_manager and task_id:
                await websocket_manager.send_build_progress(
                    task_id=task_id,
                    stage="clone",
                    progress=20,
                    message=f"Cloning {repo_source_name} repository...",
                    log_lines=[f"Cloning {repo_source_name} repository..."]
                )
            
            clone_dir = os.path.join(version_dir, "llama.cpp")
            
            # Simple git clone with timeout
            try:
                clone_process = await asyncio.create_subprocess_exec(
                    "git", "clone", repository_url, clone_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                clone_stdout, clone_stderr = await asyncio.wait_for(
                    clone_process.communicate(),
                    timeout=300  # 5 minute timeout
                )
                
                if clone_process.returncode != 0:
                    error_msg = clone_stderr.decode().strip()
                    raise Exception(f"Git clone failed: {error_msg}")
                
                logger.info("Repository cloned successfully")
                
            except asyncio.TimeoutError:
                logger.error("Git clone timed out")
                clone_process.kill()
                await clone_process.wait()
                raise Exception("Git clone timed out - network issues")
            
            # Stage 2: Checkout specific commit/branch (simplified)
            if websocket_manager and task_id:
                await websocket_manager.send_build_progress(
                    task_id=task_id,
                    stage="checkout",
                    progress=40,
                    message=f"Checking out {commit_sha}...",
                    log_lines=[f"Checking out {commit_sha}..."]
                )
            
            try:
                checkout_process = await asyncio.create_subprocess_exec(
                    "git", "checkout", commit_sha,
                    cwd=clone_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                checkout_stdout, checkout_stderr = await asyncio.wait_for(
                    checkout_process.communicate(),
                    timeout=60
                )
                
                if checkout_process.returncode != 0:
                    error_msg = checkout_stderr.decode().strip()
                    # Try main as fallback for "master" (legacy support)
                    if commit_sha == "master":
                        logger.info("Failed to checkout 'master', trying 'main'")
                        main_process = await asyncio.create_subprocess_exec(
                            "git", "checkout", "main",
                            cwd=clone_dir,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        
                        main_stdout, main_stderr = await asyncio.wait_for(
                            main_process.communicate(),
                            timeout=60
                        )
                        
                        if main_process.returncode != 0:
                            raise Exception(f"Failed to checkout both 'master' and 'main': {main_stderr.decode()}")
                    else:
                        raise Exception(f"Failed to checkout {commit_sha}: {error_msg}")
                
                logger.info(f"Successfully checked out {commit_sha}")
                
            except asyncio.TimeoutError:
                raise Exception("Git checkout timed out")
            
            # Stage 3: Apply patches (if any)
            if patches:
                if websocket_manager and task_id:
                    await websocket_manager.send_build_progress(
                        task_id=task_id,
                        stage="patch",
                        progress=50,
                        message=f"Applying {len(patches)} patches...",
                        log_lines=[f"Applying {len(patches)} patches..."]
                    )
                
                for patch_url in patches:
                    await self._apply_patch(clone_dir, patch_url)
            
            # Stage 4: Build following official documentation
            if websocket_manager and task_id:
                await websocket_manager.send_build_progress(
                    task_id=task_id,
                    stage="configure",
                    progress=60,
                    message="Configuring build with CMake...",
                    log_lines=["Running CMake configuration..."]
                )
            
            # Create build directory
            build_dir = os.path.join(clone_dir, "build")
            os.makedirs(build_dir, exist_ok=True)
            
            # Prepare build configuration
            if build_config is None:
                build_config = BuildConfig()  # Use defaults
            else:
                # Ensure dependent options stay in sync
                if build_config.enable_cpu_all_variants:
                    build_config.enable_backend_dl = True
                build_config.normalize()
            
            # Enforce build_examples for ik_llama.cpp (server is in examples directory)
            if repo_source_name == "ik_llama.cpp" and not build_config.build_examples:
                logger.warning("ik_llama.cpp requires LLAMA_BUILD_EXAMPLES=ON (server is in examples directory). Enabling automatically.")
                build_config.build_examples = True
            
            # Validate CUDA Toolkit availability if CUDA is enabled
            validated_cuda_root = None
            if build_config.enable_cuda:
                cuda_available, cuda_root, cuda_error = self._check_cuda_toolkit_available()
                if not cuda_available:
                    # Check if CUDA installer is available
                    try:
                        from backend.cuda_installer import get_cuda_installer
                        installer = get_cuda_installer()
                        installer_status = installer.status()
                        if not installer_status.get("installed"):
                            error_msg = (
                                f"CUDA build requested but CUDA Toolkit not found.\n\n"
                                f"{cuda_error}\n\n"
                                f"You can install CUDA Toolkit using the CUDA installer in the LlamaCpp Manager, "
                                f"or install it manually from https://developer.nvidia.com/cuda-downloads"
                            )
                        else:
                            error_msg = f"CUDA build requested but CUDA Toolkit not found.\n\n{cuda_error}"
                    except ImportError:
                        error_msg = f"CUDA build requested but CUDA Toolkit not found.\n\n{cuda_error}"
                    
                    logger.error(error_msg)
                    if websocket_manager and task_id:
                        await websocket_manager.send_build_progress(
                            task_id=task_id,
                            stage="configure",
                            progress=60,
                            message="CUDA Toolkit validation failed",
                            log_lines=[error_msg]
                        )
                    raise Exception(error_msg)
                
                # Verify nvcc is actually executable
                nvcc_name = "nvcc.exe" if os.name == 'nt' else "nvcc"
                nvcc_path = os.path.join(cuda_root, "bin", nvcc_name)
                if os.path.exists(nvcc_path):
                    try:
                        # Test if nvcc can actually run
                        result = subprocess.run(
                            [nvcc_path, "--version"],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if result.returncode != 0:
                            error_msg = f"nvcc found at {nvcc_path} but failed to execute (exit code {result.returncode})"
                            logger.error(error_msg)
                            if websocket_manager and task_id:
                                await websocket_manager.send_build_progress(
                                    task_id=task_id,
                                    stage="configure",
                                    progress=60,
                                    message="CUDA compiler verification failed",
                                    log_lines=[error_msg]
                                )
                            raise Exception(error_msg)
                        else:
                            logger.info(f"CUDA Toolkit verified at: {cuda_root} (nvcc version: {result.stdout.split(chr(10))[3] if len(result.stdout.split(chr(10))) > 3 else 'unknown'})")
                    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
                        error_msg = f"Failed to verify nvcc at {nvcc_path}: {e}"
                        logger.error(error_msg)
                        if websocket_manager and task_id:
                            await websocket_manager.send_build_progress(
                                task_id=task_id,
                                stage="configure",
                                progress=60,
                                message="CUDA compiler verification failed",
                                log_lines=[error_msg]
                            )
                        raise Exception(error_msg)
                else:
                    error_msg = f"nvcc not found at expected path {nvcc_path} (CUDA root: {cuda_root})"
                    logger.error(error_msg)
                    if websocket_manager and task_id:
                        await websocket_manager.send_build_progress(
                            task_id=task_id,
                            stage="configure",
                            progress=60,
                            message="CUDA compiler not found",
                            log_lines=[error_msg]
                        )
                        raise Exception(error_msg)
                
                # Store validated CUDA root for later use
                validated_cuda_root = cuda_root
            
            # Build CMake arguments
            cmake_args = ["cmake", ".."]
            
            # Add build type
            cmake_args.append(f"-DCMAKE_BUILD_TYPE={build_config.build_type}")

            def set_flag(flag: str, value: bool):
                state = "ON" if value else "OFF"
                cmake_args.append(f"-D{flag}={state}")
            
            # Add GPU/compute backends
            set_flag("GGML_CUDA", build_config.enable_cuda)
            # Explicitly disable CUDA language if CUDA is disabled to prevent auto-detection
            if not build_config.enable_cuda:
                cmake_args.append("-DCMAKE_CUDA_COMPILER=")  # Empty string disables CUDA language
            elif validated_cuda_root:
                # === BULLETPROOF CUDA CONFIGURATION ===
                nvcc_name = "nvcc.exe" if os.name == 'nt' else "nvcc"
                nvcc_path = os.path.join(validated_cuda_root, "bin", nvcc_name)
                
                if os.path.exists(nvcc_path):
                    # 1. Verify CMake version (3.18+ required for CUDA, 3.20+ for CUDA20)
                    cmake_version = self._get_cmake_version()
                    if cmake_version:
                        logger.info(f"CMake version: {cmake_version[0]}.{cmake_version[1]}.{cmake_version[2]}")
                        if cmake_version[0] < 3 or (cmake_version[0] == 3 and cmake_version[1] < 18):
                            error_msg = (
                                f"CMake version {cmake_version[0]}.{cmake_version[1]}.{cmake_version[2]} is too old for CUDA builds.\n"
                                "CUDA builds require CMake 3.18 or newer (3.20+ recommended for CUDA20).\n"
                                "Please upgrade CMake or disable CUDA in build configuration."
                            )
                            raise Exception(error_msg)
                    
                    # 2. Verify CUDA version (11.2+ required for CUDA20 standard)
                    cuda_version = self._get_cuda_version(nvcc_path)
                    if cuda_version:
                        logger.info(f"CUDA version: {cuda_version[0]}.{cuda_version[1]}")
                        if cuda_version[0] < 11 or (cuda_version[0] == 11 and cuda_version[1] < 2):
                            logger.warning(
                                f"CUDA {cuda_version[0]}.{cuda_version[1]} may not fully support CUDA20 standard. "
                                "CUDA 11.2+ is recommended. Build may fail."
                            )
                    
                    # 3. Set CMAKE_CUDA_COMPILER (primary way to tell CMake where nvcc is)
                    cmake_args.append(f"-DCMAKE_CUDA_COMPILER={nvcc_path}")
                    
                    # 4. Set all CUDA toolkit path variables (different CMake versions use different ones)
                    cmake_args.append(f"-DCUDAToolkit_ROOT={validated_cuda_root}")
                    cmake_args.append(f"-DCUDA_TOOLKIT_ROOT_DIR={validated_cuda_root}")
                    cmake_args.append(f"-DCMAKE_CUDA_COMPILER_TOOLKIT_ROOT={validated_cuda_root}")
                    
                    # 5. Set CUDA include directories explicitly
                    cuda_include = os.path.join(validated_cuda_root, "include")
                    if os.path.exists(cuda_include):
                        cmake_args.append(f"-DCMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES={cuda_include}")
                    
                    # 6. Set CUDA library directories explicitly
                    for lib_dir in ["lib64", "lib"]:
                        cuda_lib = os.path.join(validated_cuda_root, lib_dir)
                        if os.path.exists(cuda_lib):
                            cmake_args.append(f"-DCMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES={cuda_lib}")
                            break
                    
                    # 7. Set CUDA host compiler explicitly (use system gcc/g++)
                    if os.name != 'nt':
                        gcc_path = shutil.which("gcc")
                        gxx_path = shutil.which("g++")
                        if gcc_path and gxx_path:
                            cmake_args.append(f"-DCMAKE_CUDA_HOST_COMPILER={gxx_path}")
                    
                    logger.info(f"CUDA configuration: compiler={nvcc_path}, toolkit={validated_cuda_root}")
            set_flag("GGML_VULKAN", build_config.enable_vulkan)
            set_flag("GGML_METAL", build_config.enable_metal)
            set_flag("GGML_BLAS", build_config.enable_openblas)
            if build_config.enable_openblas:
                cmake_args.append("-DGGML_BLAS_VENDOR=OpenBLAS")
            set_flag("GGML_CUDA_FA_ALL_QUANTS", build_config.enable_flash_attention and build_config.enable_cuda)

            # Auto-detect CUDA architectures when building in NVIDIA containers
            if build_config.enable_cuda:
                cuda_arch = build_config.cuda_architectures.strip()
                if not cuda_arch:
                    cuda_arch = await self._detect_cuda_architectures()
                if cuda_arch:
                    cmake_args.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}")

            # Build artifact selection
            set_flag("LLAMA_BUILD_COMMON", build_config.build_common)
            set_flag("LLAMA_BUILD_TESTS", build_config.build_tests)
            set_flag("LLAMA_BUILD_TOOLS", build_config.build_tools)
            set_flag("LLAMA_BUILD_EXAMPLES", build_config.build_examples)
            set_flag("LLAMA_BUILD_SERVER", build_config.build_server)
            set_flag("LLAMA_TOOLS_INSTALL", build_config.install_tools)

            # Advanced GGML options
            set_flag("GGML_BACKEND_DL", build_config.enable_backend_dl)
            set_flag("GGML_CPU_ALL_VARIANTS", build_config.enable_cpu_all_variants)
            set_flag("GGML_LTO", build_config.enable_lto)
            set_flag("GGML_NATIVE", build_config.enable_native)
            
            # Disable CURL if not available (llama.cpp requires it by default, but we can disable it)
            # Try to detect if CURL dev headers are available
            try:
                result = subprocess.run(
                    ["pkg-config", "--exists", "libcurl"],
                    capture_output=True,
                    timeout=2
                )
                if result.returncode != 0:
                    # CURL not found, disable it
                    set_flag("LLAMA_CURL", False)
                    logger.warning("CURL development headers not found, disabling LLAMA_CURL")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                # pkg-config not available or timeout, try to disable CURL to avoid build failure
                set_flag("LLAMA_CURL", False)
                logger.warning("Could not check for CURL, disabling LLAMA_CURL to avoid build failure")
            
            # Add custom CMake args if provided
            if build_config.custom_cmake_args:
                import shlex
                cmake_args.extend(shlex.split(build_config.custom_cmake_args))
            
            # Simple CMake configuration following official docs
            try:
                # Set environment variables if provided
                env = os.environ.copy()
                if build_config.env_vars:
                    env.update(build_config.env_vars)
                
                # Add compiler flags
                if build_config.cflags:
                    env["CFLAGS"] = build_config.cflags
                if build_config.cxxflags:
                    env["CXXFLAGS"] = build_config.cxxflags

                # === BULLETPROOF CUDA ENVIRONMENT SETUP ===
                if build_config.enable_cuda and validated_cuda_root:
                    cuda_root = validated_cuda_root
                    nvcc_name = "nvcc.exe" if os.name == 'nt' else "nvcc"
                    nvcc_path = os.path.join(cuda_root, "bin", nvcc_name)
                    
                    # 1. Set ALL CUDA-related environment variables
                    env["CUDA_PATH"] = cuda_root
                    env["CUDA_HOME"] = cuda_root
                    env["CUDA_ROOT"] = cuda_root
                    env["CUDACXX"] = nvcc_path
                    env["CUDA_COMPILER"] = nvcc_path
                    
                    # 2. Add CUDA bin to PATH (at the front for priority)
                    cuda_bin = os.path.join(cuda_root, "bin")
                    current_path = env.get("PATH", "")
                    if cuda_bin not in current_path:
                        env["PATH"] = f"{cuda_bin}{os.pathsep}{current_path}" if current_path else cuda_bin
                    
                    # 3. Set up library paths
                    if os.name != 'nt':
                        # Linux: Set LD_LIBRARY_PATH and LIBRARY_PATH
                        for lib_dir in ("lib64", "lib"):
                            cuda_lib = os.path.join(cuda_root, lib_dir)
                            if os.path.exists(cuda_lib):
                                # LD_LIBRARY_PATH for runtime
                                current_ld = env.get("LD_LIBRARY_PATH", "")
                                if cuda_lib not in current_ld:
                                    env["LD_LIBRARY_PATH"] = f"{cuda_lib}{os.pathsep}{current_ld}" if current_ld else cuda_lib
                                
                                # LIBRARY_PATH for linker
                                current_lib = env.get("LIBRARY_PATH", "")
                                if cuda_lib not in current_lib:
                                    env["LIBRARY_PATH"] = f"{cuda_lib}{os.pathsep}{current_lib}" if current_lib else cuda_lib
                                break
                        
                        # Set CPATH for CUDA headers
                        cuda_include = os.path.join(cuda_root, "include")
                        if os.path.exists(cuda_include):
                            current_cpath = env.get("CPATH", "")
                            if cuda_include not in current_cpath:
                                env["CPATH"] = f"{cuda_include}{os.pathsep}{current_cpath}" if current_cpath else cuda_include
                    else:
                        # Windows: Add lib to PATH for DLLs
                        for lib_dir in ("lib/x64", "lib"):
                            cuda_lib = os.path.join(cuda_root, lib_dir)
                            if os.path.exists(cuda_lib) and cuda_lib not in env["PATH"]:
                                env["PATH"] = f"{cuda_lib}{os.pathsep}{env['PATH']}"
                                break
                    
                    # 4. Log the complete CUDA environment
                    logger.info(
                        f"CUDA environment configured:\n"
                        f"  CUDA_PATH={env['CUDA_PATH']}\n"
                        f"  CUDACXX={env['CUDACXX']}\n"
                        f"  PATH includes: {cuda_bin}\n"
                        f"  LD_LIBRARY_PATH={env.get('LD_LIBRARY_PATH', 'not set')}"
                    )
                
                # Log cmake arguments for debugging
                logger.info(f"CMake command: {' '.join(cmake_args)}")
                
                cmake_process = await asyncio.create_subprocess_exec(
                    *cmake_args,
                    cwd=build_dir,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                cmake_stdout, cmake_stderr = await asyncio.wait_for(
                    cmake_process.communicate(),
                    timeout=180  # 3 minute timeout
                )
                
                if cmake_process.returncode != 0:
                    error_msg = cmake_stderr.decode().strip()
                    stdout_msg = cmake_stdout.decode().strip() if cmake_stdout else ""
                    logger.warning(f"CMake configuration failed: {error_msg}")
                    
                    # Provide more helpful error messages for CUDA-related failures
                    if build_config.enable_cuda and ("CUDA" in error_msg.upper() or "cuda" in error_msg.lower() or "CUDA20" in error_msg):
                        # Collect diagnostic information
                        diag_info = []
                        
                        if validated_cuda_root:
                            nvcc_name = "nvcc.exe" if os.name == 'nt' else "nvcc"
                            nvcc_path = os.path.join(validated_cuda_root, "bin", nvcc_name)
                            
                            # Check toolkit completeness
                            is_complete, missing = self._verify_cuda_toolkit_complete(validated_cuda_root)
                            if not is_complete:
                                diag_info.append(f"CUDA toolkit incomplete, missing: {', '.join(missing)}")
                            
                            # Get versions
                            cmake_ver = self._get_cmake_version()
                            cuda_ver = self._get_cuda_version(nvcc_path) if os.path.exists(nvcc_path) else None
                            
                            diag_info.append(f"CUDA_PATH: {validated_cuda_root}")
                            diag_info.append(f"nvcc exists: {os.path.exists(nvcc_path)}")
                            diag_info.append(f"include dir exists: {os.path.exists(os.path.join(validated_cuda_root, 'include'))}")
                            diag_info.append(f"CMake version: {cmake_ver[0]}.{cmake_ver[1]}.{cmake_ver[2] if cmake_ver else 'unknown'}")
                            diag_info.append(f"CUDA version: {cuda_ver[0]}.{cuda_ver[1] if cuda_ver else 'unknown'}")
                            
                            # Check for CUDA20 specific error
                            if "CUDA20" in error_msg:
                                diag_info.append("")
                                diag_info.append("CUDA20 dialect error: This requires CMake 3.20+ and CUDA 11.2+")
                                if cmake_ver and (cmake_ver[0] < 3 or (cmake_ver[0] == 3 and cmake_ver[1] < 20)):
                                    diag_info.append(f"Your CMake ({cmake_ver[0]}.{cmake_ver[1]}) is too old for CUDA20")
                                if cuda_ver and (cuda_ver[0] < 11 or (cuda_ver[0] == 11 and cuda_ver[1] < 2)):
                                    diag_info.append(f"Your CUDA ({cuda_ver[0]}.{cuda_ver[1]}) is too old for CUDA20")
                        
                        diagnostics = "\n".join(f"  - {d}" for d in diag_info) if diag_info else "  (no diagnostic info available)"
                        
                        enhanced_error = (
                            f"CMake configuration failed with CUDA error:\n\n{error_msg}\n\n"
                            f"Diagnostic information:\n{diagnostics}\n\n"
                            "Possible solutions:\n"
                            "1. Upgrade CMake to 3.20 or newer\n"
                            "2. Upgrade CUDA Toolkit to 11.2 or newer\n"
                            "3. Ensure CUDA Toolkit is fully installed (not just runtime)\n"
                            "4. Disable CUDA in build configuration (set enable_cuda: false)"
                        )
                        raise Exception(enhanced_error)
                    else:
                        raise Exception(f"CMake configuration failed: {error_msg}")
                
                logger.info("CMake configuration completed successfully")
                
                # List available targets for debugging (especially useful for ik_llama.cpp)
                try:
                    targets_process = await asyncio.create_subprocess_exec(
                        "cmake", "--build", ".", "--target", "help",
                        cwd=build_dir,
                        env=env,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    targets_stdout, targets_stderr = await asyncio.wait_for(
                        targets_process.communicate(),
                        timeout=30
                    )
                    if targets_process.returncode == 0:
                        targets_output = targets_stdout.decode('utf-8', errors='replace')
                        # Extract target names (look for lines with "..." which indicate targets)
                        target_lines = [line.strip() for line in targets_output.split('\n') 
                                      if '...' in line or 'llama' in line.lower() or 'server' in line.lower()]
                        if target_lines:
                            logger.info(f"Available CMake targets (sample): {target_lines[:10]}")
                            # Check if llama-server target exists
                            if not any('llama-server' in line.lower() or 'server' in line.lower() for line in target_lines):
                                logger.warning(f"llama-server target not found in available targets. Repository: {repo_source_name}")
                                if websocket_manager and task_id:
                                    await websocket_manager.send_build_progress(
                                        task_id=task_id,
                                        stage="configure",
                                        progress=65,
                                        message="Warning: llama-server target not found, will try building all targets",
                                        log_lines=["Available targets (sample):"] + target_lines[:5]
                                    )
                except Exception as targets_error:
                    logger.debug(f"Could not list CMake targets: {targets_error}")
                    # Non-critical, continue with build
                
            except asyncio.TimeoutError:
                raise Exception("CMake configuration timed out")
            
            # Stage 5: Build
            if websocket_manager and task_id:
                await websocket_manager.send_build_progress(
                    task_id=task_id,
                    stage="build",
                    progress=70,
                    message="Building llama.cpp...",
                    log_lines=["Starting compilation..."]
                )
            
            # Build with optimal thread count
            try:
                thread_count = self.get_optimal_build_threads()
                logger.info(f"Building with {thread_count} threads")
                
                # Set environment variables if provided
                env = os.environ.copy()
                if build_config.env_vars:
                    env.update(build_config.env_vars)
                
                # Explicitly build llama-server target
                build_process = await asyncio.create_subprocess_exec(
                    "cmake", "--build", ".", "--target", "llama-server", "--", "-j", str(thread_count),
                    cwd=build_dir,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT  # Merge stderr into stdout
                )
                
                # Stream build output for better diagnostics
                build_output_lines = []
                last_progress_update = time.time()
                async def read_output():
                    nonlocal last_progress_update
                    while True:
                        line = await build_process.stdout.readline()
                        if not line:
                            break
                        decoded_line = line.decode('utf-8', errors='replace').rstrip()
                        build_output_lines.append(decoded_line)
                        logger.debug(f"Build output: {decoded_line}")
                        # Send progress updates for important lines and periodically
                        if websocket_manager and task_id:
                            should_send = False
                            line_lower = decoded_line.lower()
                            # Always send errors and warnings
                            if any(keyword in line_lower for keyword in ["error", "warning", "fatal", "failed"]):
                                should_send = True
                            # Send periodic updates (every 5 seconds) to show progress
                            elif time.time() - last_progress_update > 5:
                                should_send = True
                                last_progress_update = time.time()
                            # Send important build milestones
                            elif any(keyword in line_lower for keyword in ["building", "linking", "built target", "scanning", "configuring"]):
                                should_send = True
                            
                            if should_send:
                                await websocket_manager.send_build_progress(
                                    task_id=task_id,
                                    stage="build",
                                    progress=70,
                                    message="Building llama.cpp...",
                                    log_lines=[decoded_line]
                                )
                
                # Start reading output
                read_task = asyncio.create_task(read_output())
                
                # Wait for process to complete
                returncode = await asyncio.wait_for(
                    build_process.wait(),
                    timeout=1800  # 30 minute timeout for build
                )
                
                # Wait for output reading to finish
                await read_task
                
                build_output = "\n".join(build_output_lines)
                
                if returncode != 0:
                    logger.error(f"Build failed with return code {returncode}")
                    logger.error(f"Build output:\n{build_output}")
                    # Send error output via websocket if available
                    if websocket_manager and task_id:
                        await websocket_manager.send_build_progress(
                            task_id=task_id,
                            stage="build",
                            progress=70,
                            message=f"Build failed (exit code {returncode})",
                            log_lines=build_output_lines[-50:]  # Last 50 lines
                        )
                    raise Exception(f"Build failed (exit code {returncode}). Check logs for details.")
                
                # Check if build output indicates actual success
                # Sometimes cmake returns 0 even if target wasn't built
                build_output_lower = build_output.lower()
                has_build_errors = any(
                    keyword in build_output_lower 
                    for keyword in ["error", "failed", "fatal", "undefined reference", "cannot find", "no rule to make target"]
                )
                
                # Check if llama-server was actually built (look for linking or building messages)
                # Also check for "up to date" which means target exists but wasn't rebuilt
                target_built = any(
                    indicator in build_output_lower 
                    for indicator in ["llama-server", "linking", "built target", "server", "up to date"]
                )
                
                # Check if target was skipped or not found
                target_not_found = any(
                    keyword in build_output_lower 
                    for keyword in ["no rule to make target", "target.*not found", "unknown target"]
                )
                
                if target_not_found:
                    logger.warning(f"Build target 'llama-server' not found, trying 'server' target (for examples/server)...")
                    if websocket_manager and task_id:
                        await websocket_manager.send_build_progress(
                            task_id=task_id,
                            stage="build",
                            progress=70,
                            message="Trying 'server' target instead...",
                            log_lines=["Target 'llama-server' not found, trying 'server' target..."]
                        )
                    
                    # Try 'server' target (used when server is in examples/)
                    logger.info("Attempting to build 'server' target...")
                    server_target_process = await asyncio.create_subprocess_exec(
                        "cmake", "--build", ".", "--target", "server", "--", "-j", str(thread_count),
                        cwd=build_dir,
                        env=env,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT
                    )
                    
                    server_target_output_lines = []
                    async def read_server_target_output():
                        while True:
                            line = await server_target_process.stdout.readline()
                            if not line:
                                break
                            decoded_line = line.decode('utf-8', errors='replace').rstrip()
                            server_target_output_lines.append(decoded_line)
                            logger.debug(f"Server target build output: {decoded_line}")
                    
                    read_server_task = asyncio.create_task(read_server_target_output())
                    server_target_returncode = await asyncio.wait_for(
                        server_target_process.wait(),
                        timeout=1800
                    )
                    await read_server_task
                    
                    server_target_output = "\n".join(server_target_output_lines)
                    
                    if server_target_returncode == 0:
                        logger.info("Successfully built 'server' target")
                        build_output = server_target_output  # Use server target output
                        build_output_lines = server_target_output_lines
                    else:
                        logger.error(f"Build target 'server' also failed, trying all targets as last resort")
                        logger.error(f"Server target build output:\n{server_target_output}")
                        if websocket_manager and task_id:
                            await websocket_manager.send_build_progress(
                                task_id=task_id,
                                stage="build",
                                progress=70,
                                message="Server target failed, building all targets...",
                                log_lines=["Target 'server' also not found, building all targets..."]
                            )
                        # Try building all targets as last resort
                        logger.info("Attempting to build all targets as fallback...")
                        all_targets_process = await asyncio.create_subprocess_exec(
                            "cmake", "--build", ".", "--", "-j", str(thread_count),
                            cwd=build_dir,
                            env=env,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.STDOUT
                        )
                    all_targets_output_lines = []
                    async def read_all_targets_output():
                        while True:
                            line = await all_targets_process.stdout.readline()
                            if not line:
                                break
                            decoded_line = line.decode('utf-8', errors='replace').rstrip()
                            all_targets_output_lines.append(decoded_line)
                            logger.debug(f"All targets build output: {decoded_line}")
                    
                    read_all_task = asyncio.create_task(read_all_targets_output())
                    all_targets_returncode = await asyncio.wait_for(
                        all_targets_process.wait(),
                        timeout=1800
                    )
                    await read_all_task
                    
                    if all_targets_returncode != 0:
                        all_targets_output = "\n".join(all_targets_output_lines)
                        logger.error(f"Building all targets failed with return code {all_targets_returncode}")
                        logger.error(f"Build output:\n{all_targets_output}")
                        raise Exception(f"Build target 'llama-server' not found and building all targets failed (exit code {all_targets_returncode})")
                    
                    # Update build output with all targets output
                    build_output_lines.extend(all_targets_output_lines)
                    build_output = "\n".join(build_output_lines)
                    logger.info("Building all targets completed, will search for binary")
                
                if has_build_errors and not target_built and not target_not_found:
                    logger.error(f"Build completed with return code 0 but contains errors")
                    logger.error(f"Build output:\n{build_output}")
                    if websocket_manager and task_id:
                        await websocket_manager.send_build_progress(
                            task_id=task_id,
                            stage="build",
                            progress=70,
                            message="Build completed but contains errors",
                            log_lines=build_output_lines[-50:]
                        )
                    raise Exception("Build completed but contains errors. Check logs for details.")
                
                logger.info("Build completed successfully")
                logger.info(f"Build output (last 20 lines):\n" + "\n".join(build_output_lines[-20:]))
                if not target_built and not target_not_found:
                    logger.warning("Build output doesn't clearly indicate llama-server was built - will verify binary exists")
                
                # Immediately check if binary exists in common locations
                # Note: For ik_llama.cpp, binary is in clone_dir/bin/ (parent of build_dir)
                clone_dir = os.path.dirname(build_dir) if os.path.basename(build_dir) == "build" else build_dir
                quick_check_paths = [
                    os.path.join(clone_dir, "bin", "llama-server"),  # Common location for ik_llama.cpp
                    os.path.join(build_dir, "bin", "llama-server"),
                    os.path.join(build_dir, "llama-server"),
                ]
                binary_found_quick = False
                for quick_path in quick_check_paths:
                    if os.path.exists(quick_path):
                        logger.info(f"Binary found immediately after build: {quick_path}")
                        binary_found_quick = True
                        break
                
                if not binary_found_quick:
                    logger.warning("Binary not found in common locations immediately after build - will search more thoroughly")
                    if websocket_manager and task_id:
                        await websocket_manager.send_build_progress(
                            task_id=task_id,
                            stage="build",
                            progress=75,
                            message="Build completed, searching for binary...",
                            log_lines=["Binary not found in expected location, searching..."]
                        )
            
            except asyncio.TimeoutError:
                raise Exception("Build timed out")
            
            # Stage 6: Find executable
            if websocket_manager and task_id:
                await websocket_manager.send_build_progress(
                    task_id=task_id,
                    stage="verify",
                    progress=90,
                    message="Verifying build...",
                    log_lines=["Searching for llama-server..."]
                )
            
            # Find llama-server executable
            final_server_path = None
            
            # Check common locations first
            # Note: For ik_llama.cpp and some builds, binary is in clone_dir/bin/ (parent of build_dir)
            clone_dir = os.path.dirname(build_dir) if os.path.basename(build_dir) == "build" else build_dir
            common_paths = [
                os.path.join(clone_dir, "bin", "llama-server"),  # Common location for ik_llama.cpp
                os.path.join(build_dir, "bin", "llama-server"),
                os.path.join(build_dir, "llama-server"),
                os.path.join(build_dir, "server", "llama-server"),
            ]
            
            for path in common_paths:
                if os.path.exists(path):
                    final_server_path = path
                    logger.info(f"Found llama-server at: {final_server_path}")
                    break
            
            # If not found in common locations, search recursively (look for both llama-server and server)
            if not final_server_path:
                logger.warning("llama-server not found in common locations, searching recursively...")
                for root, _, files in os.walk(build_dir):
                    # Check for llama-server first (standard name)
                    if "llama-server" in files:
                        final_server_path = os.path.join(root, "llama-server")
                        logger.info(f"Found llama-server at: {final_server_path}")
                        break
                    # Also check for just "server" (used in examples/server for some forks)
                    if "server" in files and os.path.isfile(os.path.join(root, "server")):
                        # Make sure it's executable and not a directory
                        server_path = os.path.join(root, "server")
                        if os.access(server_path, os.X_OK):
                            final_server_path = server_path
                            logger.info(f"Found server at: {final_server_path}")
                            break
            
            if not final_server_path or not os.path.exists(final_server_path):
                # List what was actually built
                logger.error(f"llama-server executable not found after build in {build_dir}")
                logger.error(f"Repository source: {repo_source_name}")
                logger.error(f"Build directory: {build_dir}")
                logger.error("Searching for any executables in build directory...")
                
                # Also check the clone directory in case build structure is different
                executables_found = []
                search_dirs = [build_dir, clone_dir]
                
                for search_dir in search_dirs:
                    if not os.path.exists(search_dir):
                        continue
                    for root, _, files in os.walk(search_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            # Check if it's an executable (Unix) or has executable extension (Windows)
                            is_executable = (
                                os.access(file_path, os.X_OK) if os.name != 'nt' 
                                else file_path.endswith(('.exe', '.bat', '.cmd'))
                            )
                            if is_executable and os.path.isfile(file_path):
                                rel_path = os.path.relpath(file_path, build_dir)
                                executables_found.append(rel_path)
                
                # Also check for server-related binaries with different names
                server_variants = ["server", "llama_server", "llama-server.exe", "server.exe"]
                for variant in server_variants:
                    for search_dir in search_dirs:
                        if not os.path.exists(search_dir):
                            continue
                        for root, _, files in os.walk(search_dir):
                            if variant in files:
                                variant_path = os.path.join(root, variant)
                                if os.path.exists(variant_path):
                                    logger.warning(f"Found server variant '{variant}' at: {variant_path}")
                                    # Try to use this as the server path
                                    final_server_path = variant_path
                                    break
                        if final_server_path:
                            break
                    if final_server_path:
                        break
                
                if not final_server_path:
                    error_msg = f"llama-server executable not found after build in {build_dir}"
                    if executables_found:
                        logger.error(f"Found executables: {executables_found}")
                        error_msg += f"\n\nFound executables: {', '.join(executables_found[:20])}"
                        error_msg += f"\n\nThis might indicate:\n"
                        error_msg += f"1. The build target name is different for {repo_source_name}\n"
                        error_msg += f"2. The build structure is different\n"
                        error_msg += f"3. The build failed silently\n\n"
                        error_msg += f"Please check the build logs for errors."
                    else:
                        error_msg += f"\n\nNo executables found in build directory. This indicates the build likely failed silently."
                        error_msg += f"\n\nPlease check:\n"
                        error_msg += f"1. Build configuration is correct\n"
                        error_msg += f"2. All dependencies are installed\n"
                        error_msg += f"3. CMake configuration succeeded\n"
                        error_msg += f"4. Build output for errors"
                    
                    # Send detailed error via websocket
                    if websocket_manager and task_id:
                        await websocket_manager.send_build_progress(
                            task_id=task_id,
                            stage="error",
                            progress=0,
                            message="Build completed but binary not found",
                            log_lines=[error_msg] + (executables_found[:10] if executables_found else [])
                        )
                    
                    raise Exception(error_msg)
            
            # Make executable
            os.chmod(final_server_path, 0o755)
            
            # Copy to version directory for easy access
            version_server_path = os.path.join(version_dir, "llama-server")
            shutil.copy2(final_server_path, version_server_path)
            os.chmod(version_server_path, 0o755)
            
            logger.info(f"Build completed, validating binary: {version_server_path}")
            
            # Validate the build
            if websocket_manager and task_id:
                await websocket_manager.send_build_progress(
                    task_id=task_id,
                    stage="validate",
                    progress=95,
                    message="Validating build...",
                    log_lines=["Running validation tests..."]
                )
            
            is_valid = await self.validate_build(version_server_path, websocket_manager, task_id)
            
            if not is_valid:
                logger.warning("Build validation failed")
                if websocket_manager and task_id:
                    await websocket_manager.send_build_progress(
                        task_id=task_id,
                        stage="validate",
                        progress=95,
                        message="Build validation failed - binary may not work correctly",
                        log_lines=["Warning: Build validation failed"]
                    )
            
            logger.info(f"Build completed successfully: {version_server_path}")
            
            if websocket_manager and task_id:
                await websocket_manager.send_build_progress(
                    task_id=task_id,
                    stage="complete",
                    progress=100,
                    message="Build completed successfully!",
                    log_lines=[f"llama-server ready at: {version_server_path}", f"Validation: {'Passed' if is_valid else 'Failed'}"]
                )
            
            return version_server_path
            
        except Exception as e:
            logger.error(f"Build failed: {e}")
            if websocket_manager and task_id:
                try:
                    await websocket_manager.send_build_progress(
                        task_id=task_id,
                        stage="error",
                        progress=0,
                        message=f"Build failed: {str(e)}",
                        log_lines=[f"Error: {str(e)}"]
                    )
                except Exception as ws_error:
                    logger.error(f"Failed to send error to WebSocket: {ws_error}")
            raise Exception(f"Failed to build from source {commit_sha}: {e}")
    
    async def _apply_patch(self, repo_dir: str, patch_url: str):
        """Apply a patch from URL"""
        try:
            if patch_url.startswith("https://github.com/"):
                # GitHub PR URL - convert to patch URL
                if "/pull/" in patch_url:
                    patch_url = patch_url.replace("/pull/", "/pull/").replace("/files", ".patch")
                elif not patch_url.endswith(".patch"):
                    patch_url += ".patch"
            
            # Download patch
            async with aiohttp.ClientSession() as session:
                async with session.get(patch_url) as response:
                    patch_content = await response.text()
            
            # Apply patch
            patch_file = os.path.join(repo_dir, "temp.patch")
            with open(patch_file, 'w') as f:
                f.write(patch_content)
            
            apply_process = await asyncio.create_subprocess_exec(
                "git", "apply", patch_file,
                cwd=repo_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            apply_stdout, apply_stderr = await apply_process.communicate()
            
            if apply_process.returncode != 0:
                raise Exception(f"Failed to apply patch: {apply_stderr.decode()}")
            
            os.remove(patch_file)
            
        except Exception as e:
            raise Exception(f"Failed to apply patch {patch_url}: {e}")
    
    def list_installed_versions(self) -> List[str]:
        """List all installed llama.cpp versions"""
        versions = []
        if os.path.exists(self.llama_dir):
            for item in os.listdir(self.llama_dir):
                version_path = os.path.join(self.llama_dir, item)
                if os.path.isdir(version_path):
                    # Check if it has a server binary
                    binary_path = os.path.join(version_path, "server")
                    if os.path.exists(binary_path) and os.access(binary_path, os.X_OK):
                        versions.append(item)
        return versions
    
    def get_version_path(self, version_name: str) -> Optional[str]:
        """Get the path to a specific version's server binary"""
        version_path = os.path.join(self.llama_dir, version_name)
        if os.path.exists(version_path):
            # Look for server binary in the version directory
            server_path = os.path.join(version_path, "server")
            if os.path.exists(server_path) and os.access(server_path, os.X_OK):
                return server_path
            
            # Look for llama-server binary in subdirectories
            for root, _, files in os.walk(version_path):
                if "llama-server" in files:
                    llama_server_path = os.path.join(root, "llama-server")
                    if os.path.exists(llama_server_path) and os.access(llama_server_path, os.X_OK):
                        return llama_server_path
        return None
    
    def delete_version(self, version_name: str) -> bool:
        """Delete a specific version"""
        version_path = os.path.join(self.llama_dir, version_name)
        if os.path.exists(version_path):
            try:
                shutil.rmtree(version_path)
                return True
            except Exception as e:
                logger.error(f"Failed to delete version {version_name}: {e}")
                return False
        return False
    
    def verify_installation(self, version_name: str) -> Dict[str, bool]:
        """Verify that all required llama.cpp commands are available for a version"""
        version_path = self.get_version_path(version_name)
        if not version_path:
            return {
                "llama-server": False,
                "llama-cli": False,
                "llama-quantize": False
            }
        
        # Check for commands in the same directory
        binary_dir = os.path.dirname(version_path)
        commands = {
            "llama-server": os.path.exists(os.path.join(binary_dir, "llama-server")),
            "llama-cli": os.path.exists(os.path.join(binary_dir, "llama-cli")),
            "llama-quantize": os.path.exists(os.path.join(binary_dir, "llama-quantize"))
        }
        
        return commands
    
    def get_all_commands(self, version_name: str) -> Dict[str, str]:
        """Get all available commands for a specific version with their full paths"""
        version_path = self.get_version_path(version_name)
        if not version_path:
            return {}
        
        binary_dir = os.path.dirname(version_path)
        commands = {}
        
        for cmd in ["llama-server", "llama-cli", "llama-quantize"]:
            cmd_path = os.path.join(binary_dir, cmd)
            if os.path.exists(cmd_path) and os.access(cmd_path, os.X_OK):
                commands[cmd] = cmd_path
        
        return commands
