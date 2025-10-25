import httpx
import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from backend.logging_config import get_logger

logger = get_logger(__name__)

class LlamaSwapClient:
    def __init__(self, base_url: str = "http://localhost:2000"):
        self.base_url = base_url
    
    async def get_running_models(self) -> List[Dict[str, Any]]:
        """Get currently running models from /running endpoint"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/running", timeout=5)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get running models: {e}")
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
    
    async def check_health(self) -> bool:
        """Check if llama-swap proxy is responsive"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health", timeout=2)
                return response.status_code == 200
        except:
            return False
    
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