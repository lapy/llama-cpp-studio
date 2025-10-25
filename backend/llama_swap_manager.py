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

def get_llama_swap_manager() -> 'LlamaSwapManager':
    """Get the global llama-swap manager instance"""
    global _llama_swap_manager_instance
    if _llama_swap_manager_instance is None:
        _llama_swap_manager_instance = LlamaSwapManager()
    return _llama_swap_manager_instance

class LlamaSwapManager:
    def __init__(self, proxy_port: int = 2000, config_path: str = "data/llama-swap-config.yaml"):
        self.proxy_port = proxy_port
        self.config_path = config_path
        self.process: Optional[subprocess.Popen] = None
        self.running_models: Dict[str, Dict[str, Any]] = {} # {proxy_model_name: {model_path, config}}
        self.proxy_url = f"http://localhost:{self.proxy_port}"
        self.admin_url = f"http://localhost:{self.proxy_port}/admin"
    
    async def _write_config(self, llama_server_path: str):
        """Writes the current running_models to the llama-swap config file."""
        # Load all models from database to include them in config
        from backend.database import get_db, Model
        db = next(get_db())
        try:
            all_models = db.query(Model).all()
            config_content = generate_llama_swap_config(self.running_models, llama_server_path, all_models)
        finally:
            db.close()
        
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w") as f:
            f.write(config_content)


    async def sync_running_models(self):
        """Sync running_models with actual state from llama-swap"""
        from backend.llama_swap_client import LlamaSwapClient
        
        client = LlamaSwapClient()
        try:
            running_models_data = await client.get_running_models()
            
            # Clear current running_models
            self.running_models.clear()
            
            # The response format is {"running": [{"model": "...", "state": "..."}]}
            if isinstance(running_models_data, dict) and "running" in running_models_data:
                running_list = running_models_data["running"]
            else:
                running_list = running_models_data
            
            # Populate with actual running models from llama-swap
            for model_data in running_list:
                if isinstance(model_data, dict):
                    proxy_model_name = model_data.get('model', '')
                    if proxy_model_name:
                        # We don't need to store the full config here since it's in the database
                        self.running_models[proxy_model_name] = {
                            "model_path": "",  # Will be loaded from database when needed
                            "config": {}  # Will be loaded from database when needed
                        }
            
            logger.info(f"Synced running_models with llama-swap state: {len(self.running_models)} models")
            
        except Exception as e:
            logger.warning(f"Failed to sync running_models with llama-swap: {e}")
            # Keep existing running_models if sync fails

    async def start_proxy(self):
        if self.process and self.process.poll() is None:
            logger.info("llama-swap is already running")
            return

        # Ensure an initial empty config is written if no models are registered yet
        # This allows llama-swap to start even without models
        await self._write_config(llama_server_path="/usr/local/bin/llama-server") # Placeholder path

        cmd = ["llama-swap", "--config", self.config_path, "--listen", f"0.0.0.0:{self.proxy_port}", "--watch-config"]
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd="/app"
        )

        # Wait for llama-swap to become ready
        await self._wait_for_proxy_ready()
    
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
        if self.process:
            logger.info("Stopping llama-swap proxy...")
            self.process.terminate()
            try:
                await asyncio.wait_for(asyncio.to_thread(self.process.wait), timeout=10)
                logger.info("llama-swap proxy stopped gracefully")
            except asyncio.TimeoutError:
                logger.warning("llama-swap did not terminate gracefully, killing process")
                self.process.kill()
                await asyncio.to_thread(self.process.wait)
            self.process = None
            self.running_models = {} # Clear registered models
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
            raise ValueError(f"Model '{proxy_model_name}' is already registered with llama-swap.")

        self.running_models[proxy_model_name] = {
            "model_path": model.file_path,
            "config": config
        }
        
        logger.info(f"Model '{model.name}' registered as '{proxy_model_name}' with llama-swap")
        return proxy_model_name

    def _detect_correct_binary_path(self, version_dir: str) -> str:
        """
        Automatically detects the correct binary path for llama-server.
        Prioritizes llama-server over server binary for better compatibility.
        """
        import os
        
        # Priority order: llama-server first (newer, works better), then server (older)
        possible_paths = [
            os.path.join(version_dir, "build", "bin", "llama-server"),  # New location (preferred)
            os.path.join(version_dir, "bin", "llama-server"),  # Alternative location
            os.path.join(version_dir, "llama-server"),  # Direct location
            os.path.join(version_dir, "server"),  # Old location (fallback)
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                logger.info(f"Found executable llama-server at: {path}")
                return path
        
        # If no executable found, return the most likely path (new location)
        logger.warning(f"No executable llama-server found in {version_dir}, using default path")
        return os.path.join(version_dir, "build", "bin", "llama-server")

    async def _ensure_correct_binary_path(self):
        """
        Ensures the active llama-cpp version has the correct binary path.
        Automatically detects and updates if needed.
        """
        from backend.database import SessionLocal, LlamaVersion
        
        db = SessionLocal()
        try:
            active_version = db.query(LlamaVersion).filter(LlamaVersion.is_active == True).first()
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
                logger.info(f"Updating binary path from '{active_version.binary_path}' to '{relative_path}'")
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
        """
        from backend.database import SessionLocal, LlamaVersion
        
        # First, ensure the binary path is correct
        await self._ensure_correct_binary_path()
        
        db = SessionLocal()
        try:
            # Get the active version
            active_version = db.query(LlamaVersion).filter(LlamaVersion.is_active == True).first()
            if not active_version:
                logger.warning("No active llama-cpp version found, skipping config regeneration")
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
            logger.info(f"Regenerated llama-swap config with active version: {active_version.version} and {len(self.running_models)} running models")
            
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
            logger.info(f"unregister_model called with proxy_model_name: {proxy_model_name}")
            logger.info(f"Starting unregister process for model '{proxy_model_name}'")

            # Unload the specific model from llama-swap (works regardless of how it was loaded)
            from backend.llama_swap_client import LlamaSwapClient
            client = LlamaSwapClient()
            try:
                logger.info(f"Calling unload_model for '{proxy_model_name}'...")
                result = await client.unload_model(proxy_model_name)
                logger.info(f"Unloaded model '{proxy_model_name}' from llama-swap, result: {result}")
            except Exception as e:
                logger.error(f"Failed to unload model '{proxy_model_name}' from llama-swap: {e}")
                raise

            # Remove the model from running_models if it exists there
            if proxy_model_name in self.running_models:
                del self.running_models[proxy_model_name]
                logger.info(f"Removed '{proxy_model_name}' from running_models")
            else:
                logger.info(f"Model '{proxy_model_name}' was not in running_models (loaded externally)")
            
            # Sync with actual llama-swap state to ensure consistency
            await self.sync_running_models()
            
            logger.info(f"Model '{proxy_model_name}' unregistered from llama-swap")
            
        except Exception as e:
            logger.error(f"Error in unregister_model: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

