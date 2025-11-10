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
    def __init__(self):
        self.llama_dir = "data/llama-cpp"
        os.makedirs(self.llama_dir, exist_ok=True)
        self._cached_cuda_architectures: Optional[str] = None

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
    
    async def build_source(self, commit_sha: str, patches: List[str] = None, build_config: BuildConfig = None, websocket_manager=None, task_id: str = None) -> str:
        """Build llama.cpp from source following official documentation - simplified approach"""
        try:
            # Send initial progress
            if websocket_manager and task_id:
                await websocket_manager.send_build_progress(
                    task_id=task_id,
                    stage="init",
                    progress=0,
                    message="Starting simplified build process...",
                    log_lines=[f"Building llama.cpp from {commit_sha}"]
                )
            
            # Create version directory
            version_name = f"source-{commit_sha[:8]}"
            version_dir = os.path.join(self.llama_dir, version_name)
            
            # Clean up existing directory
            if os.path.exists(version_dir):
                logger.info(f"Cleaning up existing directory: {version_dir}")
                shutil.rmtree(version_dir, ignore_errors=True)
                time.sleep(1)
            
            os.makedirs(version_dir, exist_ok=True)
            
            # Stage 1: Clone repository (simplified)
            if websocket_manager and task_id:
                await websocket_manager.send_build_progress(
                    task_id=task_id,
                    stage="clone",
                    progress=20,
                    message="Cloning llama.cpp repository...",
                    log_lines=["Cloning repository..."]
                )
            
            clone_dir = os.path.join(version_dir, "llama.cpp")
            
            # Simple git clone with timeout
            try:
                clone_process = await asyncio.create_subprocess_exec(
                    "git", "clone", "https://github.com/ggerganov/llama.cpp.git", clone_dir,
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
                    # Try master as fallback for "main"
                    if commit_sha == "main":
                        logger.info("Failed to checkout 'main', trying 'master'")
                        master_process = await asyncio.create_subprocess_exec(
                            "git", "checkout", "master",
                            cwd=clone_dir,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        
                        master_stdout, master_stderr = await asyncio.wait_for(
                            master_process.communicate(),
                            timeout=60
                        )
                        
                        if master_process.returncode != 0:
                            raise Exception(f"Failed to checkout both 'main' and 'master': {master_stderr.decode()}")
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
            
            # Build CMake arguments
            cmake_args = ["cmake", ".."]
            
            # Add build type
            cmake_args.append(f"-DCMAKE_BUILD_TYPE={build_config.build_type}")

            def set_flag(flag: str, value: bool):
                state = "ON" if value else "OFF"
                cmake_args.append(f"-D{flag}={state}")
            
            # Add GPU/compute backends
            set_flag("GGML_CUDA", build_config.enable_cuda)
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

                # Ensure CUDA toolchain paths are available when CUDA build requested
                if build_config.enable_cuda:
                    possible_cuda_roots = [
                        env.get("CUDA_PATH"),
                        "/usr/local/cuda",
                        "/usr/local/cuda-12.4",
                        "/usr/local/cuda-12.3",
                        "/usr/local/cuda-12.2",
                    ]
                    cuda_root = next(
                        (root for root in possible_cuda_roots if root and os.path.exists(root)),
                        None,
                    )

                    if cuda_root:
                        nvcc_path = os.path.join(cuda_root, "bin", "nvcc")
                        if os.path.exists(nvcc_path) and not env.get("CUDACXX"):
                            env["CUDACXX"] = nvcc_path
                        # Ensure PATH includes CUDA bin for CMake detection
                        cuda_bin = os.path.join(cuda_root, "bin")
                        current_path = env.get("PATH", "")
                        path_entries = [entry for entry in current_path.split(os.pathsep) if entry]
                        if cuda_bin not in path_entries:
                            env["PATH"] = os.pathsep.join([cuda_bin] + path_entries) if current_path else cuda_bin
                        # Ensure LD_LIBRARY_PATH includes CUDA libs (Linux)
                        for lib_dir in ("lib64", "lib"):
                            full_dir = os.path.join(cuda_root, lib_dir)
                            if os.path.exists(full_dir):
                                current_ld = env.get("LD_LIBRARY_PATH", "")
                                ld_paths = [entry for entry in current_ld.split(os.pathsep) if entry]
                                if full_dir not in ld_paths:
                                    ld_paths.insert(0, full_dir)
                                    env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_paths)
                
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
                    logger.warning(f"CMake configuration failed: {error_msg}")
                    raise Exception(f"CMake configuration failed: {error_msg}")
                
                logger.info("CMake configuration completed successfully")
                
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
                
                build_process = await asyncio.create_subprocess_exec(
                    "cmake", "--build", ".", "--", "-j", str(thread_count),
                    cwd=build_dir,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                build_stdout, build_stderr = await asyncio.wait_for(
                    build_process.communicate(),
                    timeout=1800  # 30 minute timeout for build
                )
                
                if build_process.returncode != 0:
                    error_msg = build_stderr.decode().strip()
                    raise Exception(f"Build failed: {error_msg}")
                
                logger.info("Build completed successfully")
                
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
            for root, _, files in os.walk(build_dir):
                if "llama-server" in files:
                    final_server_path = os.path.join(root, "llama-server")
                    break
            
            if not final_server_path or not os.path.exists(final_server_path):
                raise Exception("llama-server executable not found after build")
            
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
