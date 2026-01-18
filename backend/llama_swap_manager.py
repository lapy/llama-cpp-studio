import subprocess
import asyncio
import os
import yaml
import httpx
from typing import Dict, Any, Optional
from backend.llama_swap_config import generate_llama_swap_config
from backend.database import Model
from backend.logging_config import get_logger

logger = get_logger(__name__)

# Global singleton instance
_llama_swap_manager_instance = None


def get_llama_swap_manager() -> "LlamaSwapManager":
    """Get the global llama-swap manager instance"""
    global _llama_swap_manager_instance
    if _llama_swap_manager_instance is None:
        _llama_swap_manager_instance = LlamaSwapManager()
    return _llama_swap_manager_instance


class LlamaSwapManager:
    def __init__(self, proxy_port: int = 2000, config_path: str = None):
        self.proxy_port = proxy_port
        # Use absolute path to avoid permission issues with relative paths
        if config_path is None:
            config_path = "/app/data/llama-swap-config.yaml"
        self.config_path = (
            os.path.abspath(config_path)
            if not os.path.isabs(config_path)
            else config_path
        )
        self.process: Optional[subprocess.Popen] = None
        self.running_models: Dict[str, Dict[str, Any]] = (
            {}
        )  # {proxy_model_name: {model_path, config}}
        self.proxy_url = f"http://localhost:{self.proxy_port}"
        self.admin_url = f"http://localhost:{self.proxy_port}/admin"
        self.monitor_task: Optional[asyncio.Task] = None
        self._should_restart = True  # Flag to control auto-restart

    async def _write_config(self, llama_server_path: str = None):
        """Writes the current running_models to the llama-swap config file."""
        from backend.llama_swap_config import get_active_binary_path_from_db

        # Get binary path from database if not provided
        if not llama_server_path:
            llama_server_path = get_active_binary_path_from_db()
            if not llama_server_path:
                logger.error(
                    "Cannot write config: no llama-server binary path available"
                )
                raise ValueError(
                    "No llama-server binary path provided and none found in database"
                )

        # Load all models from database to include them in config
        from backend.database import get_db, Model

        db = next(get_db())
        try:
            all_models = db.query(Model).all()
            config_content = generate_llama_swap_config(
                self.running_models, llama_server_path, all_models
            )
        finally:
            db.close()

        # Ensure directory exists
        config_dir = os.path.dirname(self.config_path)
        os.makedirs(config_dir, exist_ok=True)

        # Use atomic write: write to temp file first, then rename
        # This avoids permission issues with existing files
        import tempfile

        temp_file = os.path.join(
            config_dir, f".llama-swap-config.yaml.tmp.{os.getpid()}"
        )

        try:
            # Write to temporary file first
            with open(temp_file, "w") as f:
                f.write(config_content)

            # Check if target file exists and is not writable
            if os.path.exists(self.config_path):
                try:
                    # Try to remove existing file
                    os.remove(self.config_path)
                except PermissionError:
                    logger.warning(
                        f"Cannot remove existing config file {self.config_path}, will try to overwrite"
                    )

            # Atomic rename (works even if target file exists and is read-only on some systems)
            os.rename(temp_file, self.config_path)
            logger.debug(f"Successfully wrote config to {self.config_path}")

        except PermissionError as e:
            # Clean up temp file if it exists
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            logger.error(f"Permission denied writing to {self.config_path}: {e}")
            logger.error(
                f"Directory: {config_dir}, exists: {os.path.exists(config_dir)}, writable: {os.access(config_dir, os.W_OK) if os.path.exists(config_dir) else 'N/A'}"
            )
            if os.path.exists(self.config_path):
                try:
                    logger.error(
                        f"File permissions: {oct(os.stat(self.config_path).st_mode)}"
                    )
                except:
                    pass
            raise
        except Exception as e:
            # Clean up temp file if it exists
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            raise

    async def sync_running_models(self):
        """Sync running_models with actual state from llama-swap"""
        from backend.llama_swap_client import LlamaSwapClient

        client = LlamaSwapClient()
        try:
            running_models_data = await client.get_running_models()

            # Clear current running_models
            self.running_models.clear()

            # The response format is {"running": [{"model": "...", "state": "..."}]}
            if (
                isinstance(running_models_data, dict)
                and "running" in running_models_data
            ):
                running_list = running_models_data["running"]
            else:
                running_list = running_models_data

            # Populate with actual running models from llama-swap
            for model_data in running_list:
                if isinstance(model_data, dict):
                    proxy_model_name = model_data.get("model", "")
                    if proxy_model_name:
                        # We don't need to store the full config here since it's in the database
                        self.running_models[proxy_model_name] = {
                            "model_path": "",  # Will be loaded from database when needed
                            "config": {},  # Will be loaded from database when needed
                        }

            logger.info(
                f"Synced running_models with llama-swap state: {len(self.running_models)} models"
            )

        except Exception as e:
            logger.warning(f"Failed to sync running_models with llama-swap: {e}")
            # Keep existing running_models if sync fails

    async def start_proxy(self):
        if self.process and self.process.poll() is None:
            logger.info("llama-swap is already running")
            return

        await self._do_start_proxy()

        # Only start monitoring task on first start, not on restart
        if self.monitor_task is None or self.monitor_task.done():
            self.monitor_task = asyncio.create_task(self._monitor_process())

        # Wait for llama-swap to become ready
        await self._wait_for_proxy_ready()

    async def _do_start_proxy(self):
        """Internal method to actually start the process"""
        # Ensure an initial empty config is written if no models are registered yet
        # This allows llama-swap to start even without models
        await self._write_config()  # Will get path from database

        cmd = [
            "llama-swap",
            "--config",
            self.config_path,
            "--listen",
            f"0.0.0.0:{self.proxy_port}",
            "--watch-config",
        ]
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,
            cwd="/app",
        )

        # Start background task to stream llama-swap logs
        asyncio.create_task(self._stream_llama_swap_logs())

    async def _stream_llama_swap_logs(self):
        """Stream llama-swap stdout/stderr to our logger"""
        if not self.process or not self.process.stdout:
            return

        async def read_loop():
            try:
                while self.process and self.process.poll() is None:
                    # Read a line asynchronously
                    line = await asyncio.to_thread(self.process.stdout.readline)
                    if not line:
                        break
                    line = line.strip()
                    if line:
                        logger.debug(f"[llama-swap] {line}")
            except Exception as e:
                logger.debug(f"Stopped reading llama-swap logs: {e}")

        # Start the reading loop
        asyncio.create_task(read_loop())

    async def _monitor_process(self):
        """Monitor llama-swap process and restart if it dies"""
        try:
            while self._should_restart:
                if self.process:
                    # Check if process is still alive
                    poll_result = self.process.poll()
                    if poll_result is not None:
                        # Process has terminated
                        exit_code = poll_result
                        logger.warning(
                            f"llama-swap process died with exit code {exit_code}"
                        )

                        if self._should_restart:
                            logger.info("Attempting to restart llama-swap...")
                            try:
                                # Clear the dead process
                                self.process = None

                                # Restart it using internal method to avoid re-creating monitor task
                                await self._do_start_proxy()

                                # Wait for it to become ready
                                await self._wait_for_proxy_ready()

                                logger.info("llama-swap restarted successfully")
                            except Exception as e:
                                logger.error(f"Failed to restart llama-swap: {e}")
                                # Wait before retrying
                                await asyncio.sleep(5)

                # Check every 2 seconds
                await asyncio.sleep(2)

        except asyncio.CancelledError:
            logger.debug("Monitor task cancelled")
        except Exception as e:
            logger.error(f"Error in monitor task: {e}")

    async def _wait_for_proxy_ready(self, timeout: int = 30):
        """Waits until the llama-swap proxy is responsive."""
        client = httpx.AsyncClient()
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            try:
                response = await client.get(f"{self.proxy_url}/health", timeout=1)
                if response.status_code == 200:
                    return
            except httpx.ConnectError:
                pass
            await asyncio.sleep(0.5)
        raise Exception("llama-swap proxy did not become ready in time.")

    async def stop_proxy(self):
        """Stops the llama-swap proxy server and all managed models."""
        # Disable auto-restart
        self._should_restart = False

        # Stop the monitor task
        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        if self.process:
            logger.info("Stopping llama-swap proxy...")
            self.process.terminate()
            try:
                await asyncio.wait_for(asyncio.to_thread(self.process.wait), timeout=10)
                logger.info("llama-swap proxy stopped gracefully")
            except asyncio.TimeoutError:
                logger.warning(
                    "llama-swap did not terminate gracefully, killing process"
                )
                self.process.kill()
                await asyncio.to_thread(self.process.wait)
            self.process = None
            self.running_models = {}  # Clear registered models
        else:
            logger.info("llama-swap is not running")

    async def register_model(self, model: Model, config: Dict[str, Any]) -> str:
        """
        Registers a model with llama-swap by storing its configuration.
        Returns the proxy_model_name used by llama-swap.
        Note: This only stores the model info, config is written separately.
        """
        # Use the centralized proxy name from the database
        if not model.proxy_name:
            raise ValueError(f"Model '{model.name}' does not have a proxy_name set")

        proxy_model_name = model.proxy_name

        if proxy_model_name in self.running_models:
            raise ValueError(
                f"Model '{proxy_model_name}' is already registered with llama-swap."
            )

        self.running_models[proxy_model_name] = {
            "model_path": model.file_path,
            "config": config,
        }

        logger.info(
            f"Model '{model.name}' registered as '{proxy_model_name}' with llama-swap"
        )
        return proxy_model_name

    def _detect_correct_binary_path(self, version_dir: str) -> str:
        """
        Automatically detects the correct binary path for llama-server.
        Prioritizes llama-server over server binary for better compatibility.
        """
        import os

        # Priority order: llama-server first (newer, works better), then server (older)
        possible_paths = [
            os.path.join(
                version_dir, "build", "bin", "llama-server"
            ),  # New location (preferred)
            os.path.join(version_dir, "bin", "llama-server"),  # Alternative location
            os.path.join(version_dir, "llama-server"),  # Direct location
            os.path.join(version_dir, "server"),  # Old location (fallback)
        ]

        for path in possible_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                logger.info(f"Found executable llama-server at: {path}")
                return path

        # If no executable found, return the most likely path (new location)
        logger.warning(
            f"No executable llama-server found in {version_dir}, using default path"
        )
        return os.path.join(version_dir, "build", "bin", "llama-server")

    async def _ensure_correct_binary_path(self):
        """
        Ensures the active llama-cpp version has the correct binary path.
        Automatically detects and updates if needed.
        """
        from backend.database import SessionLocal, LlamaVersion

        db = SessionLocal()
        try:
            active_version = (
                db.query(LlamaVersion).filter(LlamaVersion.is_active == True).first()
            )
            if not active_version:
                logger.warning("No active llama-cpp version found")
                return

            # Convert relative path to absolute
            version_dir = active_version.binary_path
            if not os.path.isabs(version_dir):
                version_dir = os.path.join("/app", version_dir)

            # Get the directory containing the binary
            binary_dir = os.path.dirname(version_dir)

            # Detect the correct binary path
            correct_binary_path = self._detect_correct_binary_path(binary_dir)

            # Convert back to relative path for database storage
            relative_path = os.path.relpath(correct_binary_path, "/app")

            # Update database if path has changed
            if active_version.binary_path != relative_path:
                logger.info(
                    f"Updating binary path from '{active_version.binary_path}' to '{relative_path}'"
                )
                active_version.binary_path = relative_path
                db.commit()
                logger.info("Binary path updated successfully")
            else:
                logger.debug(f"Binary path is already correct: {relative_path}")

        except Exception as e:
            logger.error(f"Error ensuring correct binary path: {e}")
        finally:
            db.close()

    async def regenerate_config_with_active_version(self):
        """
        Regenerates the llama-swap config using the currently active llama-cpp version.
        Syncs running_models with actual llama-swap state before regenerating.
        Automatically detects and fixes binary path if needed.
        Ensures llama-swap is running if an active version exists.
        """
        from backend.database import SessionLocal, LlamaVersion

        # First, ensure the binary path is correct
        await self._ensure_correct_binary_path()

        db = SessionLocal()
        try:
            # Get the active version
            active_version = (
                db.query(LlamaVersion).filter(LlamaVersion.is_active == True).first()
            )
            if not active_version:
                logger.warning(
                    "No active llama-cpp version found, skipping config regeneration"
                )
                return

            # Convert to absolute path for existence check
            binary_path = active_version.binary_path
            if not os.path.isabs(binary_path):
                binary_path = os.path.join("/app", binary_path)

            if not os.path.exists(binary_path):
                logger.warning(f"Active version binary not found: {binary_path}")
                return

            # Sync running_models with actual llama-swap state
            await self.sync_running_models()

            # Regenerate config with active version and synced running_models
            await self._write_config(active_version.binary_path)
            logger.info(
                f"Regenerated llama-swap config with active version: {active_version.version} and {len(self.running_models)} running models"
            )

            # Ensure llama-swap is running when we have an active version
            try:
                await self.start_proxy()
                logger.info("Ensured llama-swap is running after config regeneration")
            except Exception as e:
                logger.warning(f"Failed to start llama-swap after config regeneration: {e}")

        except Exception as e:
            logger.error(f"Failed to regenerate config with active version: {e}")
        finally:
            db.close()

    async def unregister_model(self, proxy_model_name: str):
        """
        Unregisters a model from llama-swap by unloading the specific model.
        Works for both app-registered and externally loaded models.
        """
        try:
            logger.info(
                f"unregister_model called with proxy_model_name: {proxy_model_name}"
            )
            logger.info(f"Starting unregister process for model '{proxy_model_name}'")

            # Unload the specific model from llama-swap (works regardless of how it was loaded)
            from backend.llama_swap_client import LlamaSwapClient

            client = LlamaSwapClient()
            try:
                logger.info(f"Calling unload_model for '{proxy_model_name}'...")
                result = await client.unload_model(proxy_model_name)
                logger.info(
                    f"Unloaded model '{proxy_model_name}' from llama-swap, result: {result}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to unload model '{proxy_model_name}' from llama-swap: {e}"
                )
                raise

            # Remove the model from running_models if it exists there
            if proxy_model_name in self.running_models:
                del self.running_models[proxy_model_name]
                logger.info(f"Removed '{proxy_model_name}' from running_models")
            else:
                logger.info(
                    f"Model '{proxy_model_name}' was not in running_models (loaded externally)"
                )

            # Sync with actual llama-swap state to ensure consistency
            await self.sync_running_models()

            logger.info(f"Model '{proxy_model_name}' unregistered from llama-swap")

        except Exception as e:
            logger.error(f"Error in unregister_model: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
