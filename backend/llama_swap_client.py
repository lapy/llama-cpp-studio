import httpx
import asyncio
import json
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from backend.logging_config import get_logger

logger = get_logger(__name__)


class LlamaSwapClient:
    def __init__(self, base_url: str = "http://localhost:2000"):
        self.base_url = base_url
        # Track models that are currently loading to avoid spamming 503 errors
        self._loading_models: Set[str] = set()
        self._last_health_status: bool = True
        self._consecutive_failures: int = 0
    
    def mark_model_loading(self, model_name: str):
        """Mark a model as currently loading"""
        self._loading_models.add(model_name)
        logger.debug(f"Model '{model_name}' marked as loading")
    
    def mark_model_ready(self, model_name: str):
        """Mark a model as ready (no longer loading)"""
        self._loading_models.discard(model_name)
        logger.debug(f"Model '{model_name}' marked as ready")
    
    def is_model_loading(self, model_name: str) -> bool:
        """Check if a model is currently loading"""
        return model_name in self._loading_models
    
    def get_loading_models(self) -> Set[str]:
        """Get set of currently loading models"""
        return self._loading_models.copy()
    
    def clear_loading_state(self, model_name: str):
        """Clear loading state for a model (e.g., on stop or error)"""
        self._loading_models.discard(model_name)
    
    async def get_running_models(self) -> List[Dict[str, Any]]:
        """Get currently running models from /running endpoint.
        
        The /running endpoint returns model states including 'loading' state.
        We use this to track loading models and avoid polling during load.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/running", timeout=5)
                response.raise_for_status()
                data = response.json()
                
                # Update loading states from response
                # Format: {"running": [{"model": "name", "state": "running|loading|..."}]}
                if isinstance(data, dict) and 'running' in data:
                    running_list = data['running']
                    for model_info in running_list:
                        if isinstance(model_info, dict):
                            model_name = model_info.get('model', '')
                            state = model_info.get('state', '')
                            if model_name:
                                if state == 'loading':
                                    self._loading_models.add(model_name)
                                elif state in ('running', 'ready'):
                                    self._loading_models.discard(model_name)
                
                self._consecutive_failures = 0
                return data
        except Exception as e:
            self._consecutive_failures += 1
            # Only log at debug level to avoid spam
            if self._consecutive_failures <= 3:
                logger.debug(f"Failed to get running models (attempt {self._consecutive_failures}): {e}")
            return []
    
    async def unload_model(self, model_name: str):
        """Unload a specific model via /api/models/unload/{model_name} endpoint"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{self.base_url}/api/models/unload/{model_name}", timeout=10)
                response.raise_for_status()
                return response.text
        except Exception as e:
            logger.error(f"Failed to unload model {model_name}: {e}")
            raise

    async def unload_all_models(self):
        """Unload all models via /unload endpoint"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/unload", timeout=10)
                response.raise_for_status()
                return response.text  # Return text instead of JSON since endpoint returns "OK"
        except Exception as e:
            logger.error(f"Failed to unload all models: {e}")
            raise
    
    async def check_health(self) -> Dict[str, Any]:
        """Check if llama-swap proxy is responsive.
        
        Returns a dict with health status and loading info:
        {
            "healthy": bool,
            "loading_models": list of model names currently loading,
            "status_code": int or None
        }
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health", timeout=2)
                is_healthy = response.status_code == 200
                self._last_health_status = is_healthy
                if is_healthy:
                    self._consecutive_failures = 0
                return {
                    "healthy": is_healthy,
                    "loading_models": list(self._loading_models),
                    "status_code": response.status_code
                }
        except Exception as e:
            self._consecutive_failures += 1
            return {
                "healthy": False,
                "loading_models": list(self._loading_models),
                "status_code": None,
                "error": str(e) if self._consecutive_failures <= 3 else None
            }
    
    async def check_health_simple(self) -> bool:
        """Simple health check returning just a boolean"""
        result = await self.check_health()
        return result.get("healthy", False)
    
    async def get_model_info(self, model_id: str, upstream_path: str = "v1/models"):
        """Get model info via /upstream/:model_id/* endpoint"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/upstream/{model_id}/{upstream_path}",
                    timeout=5
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get model info for {model_id}: {e}")
            raise