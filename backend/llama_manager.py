import os
import subprocess
import requests
import json
import shutil
import time
import multiprocessing
from typing import List, Optional, Dict
from dataclasses import dataclass, field, asdict
import asyncio
import aiohttp
import httpx
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
    
    # Advanced options
    custom_cmake_args: str = ""
    cflags: str = ""
    cxxflags: str = ""
    
    # Environment variables
    env_vars: Dict[str, str] = field(default_factory=dict)


class LlamaManager:
    def __init__(self):
        self.llama_dir = "data/llama-cpp"
        os.makedirs(self.llama_dir, exist_ok=True)
    
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
    
    async def install_release(self, tag_name: str, websocket_manager=None, task_id: str = None) -> str:
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
            
            # Get release info
            response = requests.get(f"https://api.github.com/repos/ggerganov/llama.cpp/releases/tags/{tag_name}")
            response.raise_for_status()
            release = response.json()
            
            # Find the appropriate Linux binary asset
            binary_asset = None
            # Try different patterns in order of preference
            patterns = [
                ("ubuntu-x64", ".zip"),  # Most common Linux binary
                ("ubuntu-vulkan-x64", ".zip"),  # Vulkan support
                ("linux-cuda", ".tar.gz"),  # Legacy naming
                ("ubuntu", ".zip"),  # Generic Ubuntu
            ]
            
            for pattern, extension in patterns:
                for asset in release["assets"]:
                    if pattern in asset["name"] and asset["name"].endswith(extension):
                        binary_asset = asset
                        break
                if binary_asset:
                    break
            
            if not binary_asset:
                raise Exception(f"No suitable Linux binary found for release {tag_name}")
            
            logger.info(f"Found binary asset: {binary_asset['name']}")
            
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
            version_name = f"release-{tag_name}"
            version_dir = os.path.join(self.llama_dir, version_name)
            
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
            if binary_asset["name"].endswith(".zip"):
                import zipfile
                with zipfile.ZipFile(download_path, 'r') as zip_ref:
                    zip_ref.extractall(version_dir)
            elif binary_asset["name"].endswith(".tar.gz"):
                import tarfile
                with tarfile.open(download_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(version_dir)
            
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
            
            return final_server_path
            
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
            
            # Build CMake arguments
            cmake_args = ["cmake", ".."]
            
            # Add build type
            cmake_args.append(f"-DCMAKE_BUILD_TYPE={build_config.build_type}")
            
            # Add GPU backends
            if build_config.enable_cuda:
                cmake_args.append("-DGGML_CUBLAS=ON")
            if build_config.enable_vulkan:
                cmake_args.append("-DGGML_VULKAN=ON")
            if build_config.enable_metal:
                cmake_args.append("-DGGML_METAL=ON")
            if build_config.enable_openblas:
                cmake_args.append("-DGGML_OPENBLAS=ON")
            if build_config.enable_flash_attention:
                cmake_args.append("-DGGML_CUDA_FA_ALL_QUANTS=ON")
            
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
