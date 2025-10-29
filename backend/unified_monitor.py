import asyncio
import psutil
import time
import yaml
import os
from collections import deque
from datetime import datetime
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session

from backend.websocket_manager import websocket_manager
from backend.gpu_detector import get_gpu_info
from backend.llama_swap_client import LlamaSwapClient
from backend.database import SessionLocal, RunningInstance, Model
from backend.logging_config import get_logger

logger = get_logger(__name__)


class UnifiedMonitor:
    """Simple, performant unified monitoring service with single stream"""
    
    def __init__(self):
        self.is_running = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.update_interval = 2.0  # Update every 2 seconds
        
        # Data storage
        self.recent_logs = deque(maxlen=100)  # Keep last 100 log entries
        
        # llama-swap integration
        self.llama_swap_client = LlamaSwapClient()
        
        # Model mapping cache
        self.model_mapping = {}
        self._load_model_mapping()
    
    def _load_model_mapping(self):
        """Load model mapping from llama-swap configuration"""
        try:
            config_path = "data/llama-swap-config.yaml" if os.path.exists("data") else "/app/data/llama-swap-config.yaml"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Extract model mappings from the config
                models_config = config.get('models', {})
                for llama_swap_name, model_config in models_config.items():
                    cmd = model_config.get('cmd', '')
                    # Extract the actual model file path from the command
                    if '--model' in cmd:
                        parts = cmd.split('--model')
                        if len(parts) > 1:
                            model_path = parts[1].strip().split()[0]
                            # Extract just the filename without path and extension
                            filename = os.path.basename(model_path).replace('.gguf', '')
                            self.model_mapping[llama_swap_name] = {
                                'filename': filename,
                                'full_path': model_path
                            }
                
                logger.info(f"Loaded {len(self.model_mapping)} model mappings from llama-swap config")
                logger.debug(f"Model mappings: {self.model_mapping}")
            else:
                logger.warning(f"llama-swap config not found at {config_path}")
        except Exception as e:
            logger.error(f"Failed to load model mapping: {e}")
    
    def add_log(self, log_event: Dict[str, Any]):
        """Add a log event to the buffer"""
        self.recent_logs.append(log_event)
    
    async def start_monitoring(self):
        """Start the unified monitoring background task"""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        
        logger.info("Unified monitoring started")
    
    async def stop_monitoring(self):
        """Stop the unified monitoring background task"""
        self.is_running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
        
        logger.info("Unified monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop that collects all metrics and sends unified stream"""
        while self.is_running:
            try:
                await self._collect_and_send_unified_data()
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Unified monitoring error: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _collect_and_send_unified_data(self):
        """Collect all monitoring data and send as single unified message"""
        try:
            # 1. System metrics
            cpu_percent = psutil.cpu_percent(interval=0)
            memory = psutil.virtual_memory()
            # Use data directory at project root or /app/data for Docker
            data_dir = 'data' if os.path.exists('data') else '/app/data'
            try:
                disk = psutil.disk_usage(data_dir)
            except FileNotFoundError:
                disk = psutil.disk_usage('/')
            
            # 2. Running instances from database
            db = SessionLocal()
            try:
                running_instances = db.query(RunningInstance).all()
                active_instances = []
                for instance in running_instances:
                    active_instances.append({
                        "id": instance.id,
                        "model_id": instance.model_id,
                        "port": instance.port,
                        "proxy_model_name": instance.proxy_model_name,
                        "started_at": instance.started_at.isoformat() if instance.started_at else None
                    })
            finally:
                db.close()
            
            # 3. Running models from external sources (llama-swap)
            try:
                external_response = await self.llama_swap_client.get_running_models()
                logger.debug(f"Raw external response: {external_response}")
                
                # Extract the running models array from the response
                if isinstance(external_response, dict) and 'running' in external_response:
                    external_models = external_response['running']
                elif isinstance(external_response, list):
                    external_models = external_response
                else:
                    external_models = []
                
                # Enhance external models with mapping information
                enhanced_external_models = []
                for model in external_models:
                    model_name = model.get('model', '')
                    enhanced_model = {
                        'model': model_name,
                        'state': model.get('state', 'unknown'),
                        'mapping': self.model_mapping.get(model_name, {})
                    }
                    enhanced_external_models.append(enhanced_model)
                
                logger.debug(f"Enhanced {len(enhanced_external_models)} external models with mapping: {enhanced_external_models}")
                
                # Sync database with external running models
                await self._sync_database_with_external_models(enhanced_external_models)
                
            except Exception as e:
                logger.error(f"Failed to get external running models: {e}")
                enhanced_external_models = []
            
            # 4. GPU info
            try:
                gpu_info = await get_gpu_info()
                vram_data = None
                if not gpu_info.get("cpu_only_mode", True):
                    vram_data = await self._get_vram_data(gpu_info)
            except Exception as e:
                logger.error(f"Failed to get GPU info: {e}")
                gpu_info = {"cpu_only_mode": True, "device_count": 0}
                vram_data = None
            
            # 5. Create unified monitoring data
            unified_data = {
                "type": "unified_monitoring",
                "timestamp": datetime.utcnow().isoformat(),
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory": {
                        "total": memory.total,
                        "available": memory.available,
                        "percent": memory.percent,
                        "used": memory.used,
                        "free": memory.free,
                        "cached": getattr(memory, 'cached', 0),
                        "buffers": getattr(memory, 'buffers', 0),
                        "swap_total": psutil.swap_memory().total,
                        "swap_used": psutil.swap_memory().used
                    },
                    "disk": {
                        "total": disk.total,
                        "used": disk.used,
                        "free": disk.free,
                        "percent": (disk.used / disk.total) * 100
                    }
                },
                "gpu": {
                    "cpu_only_mode": gpu_info.get("cpu_only_mode", True),
                    "device_count": gpu_info.get("device_count", 0),
                    "total_vram": gpu_info.get("total_vram", 0),
                    "available_vram": gpu_info.get("available_vram", 0),
                    "vram_data": vram_data
                },
                "models": {
                    "running_instances": active_instances
                },
                "proxy_status": {
                    "enabled": True,
                    "port": 2000,
                    "endpoint": "http://localhost:2000/v1/chat/completions"
                },
                "logs": list(self.recent_logs)[-20:]  # Last 20 logs
            }
            
            # 6. Send unified data to all WebSocket connections
            logger.debug(f"Broadcasting unified monitoring data: {unified_data}")
            await websocket_manager.broadcast(unified_data)
            
        except Exception as e:
            logger.error(f"Error collecting unified monitoring data: {e}")
    
    
    async def _get_vram_data(self, gpu_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get current VRAM usage data"""
        try:
            import nvidia_ml_py3 as nvml
            nvml.nvmlInit()
            
            device_count = gpu_info.get("device_count", 0)
            total_vram = 0
            used_vram = 0
            gpu_details = []
            
            for i in range(device_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                
                gpu_total = mem_info.total
                gpu_used = mem_info.used
                gpu_free = mem_info.free
                
                total_vram += gpu_total
                used_vram += gpu_used
                
                gpu_details.append({
                    "device_id": i,
                    "total": gpu_total,
                    "used": gpu_used,
                    "free": gpu_free,
                    "utilization": utilization.gpu,
                    "memory_utilization": utilization.memory
                })
            
            return {
                "total": total_vram,
                "used": used_vram,
                "free": total_vram - used_vram,
                "percent": (used_vram / total_vram * 100) if total_vram > 0 else 0,
                "gpus": gpu_details,
                "cuda_version": gpu_info.get("cuda_version", "N/A"),
                "device_count": gpu_info.get("device_count", 0),
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Failed to get VRAM data: {e}")
            return {
                "total": 0,
                "used": 0,
                "free": 0,
                "percent": 0,
                "gpus": [],
                "cuda_version": "N/A",
                "device_count": 0,
                "timestamp": time.time()
            }
    
    async def _sync_database_with_external_models(self, external_models: List[Dict[str, Any]]):
        """Sync database RunningInstance records with external running models"""
        try:
            db: Session = SessionLocal()
            try:
                # Get all current running instances from database
                current_instances = db.query(RunningInstance).all()
                current_proxy_names = {instance.proxy_model_name for instance in current_instances if instance.proxy_model_name}
                
                # Get external model names
                external_names = {model['model'] for model in external_models}
                
                # Find models that are running externally but not in database
                missing_in_db = external_names - current_proxy_names
                
                # Find models that are in database but not running externally
                missing_in_external = current_proxy_names - external_names
                
                # Add missing models to database
                for proxy_name in missing_in_db:
                    # Find the corresponding model in the database by matching the proxy name
                    model = self._find_model_by_proxy_name(db, proxy_name)
                    if model:
                        # Create a new RunningInstance
                        new_instance = RunningInstance(
                            model_id=model.id,
                            proxy_model_name=proxy_name,
                            port='N/A',  # llama-swap doesn't provide port info
                            started_at=datetime.utcnow()
                        )
                        db.add(new_instance)
                        logger.info(f"Added missing model '{proxy_name}' to database")
                        
                        # Update model.is_active
                        model.is_active = True
                
                # Remove models that are no longer running externally
                for proxy_name in missing_in_external:
                    instances_to_remove = db.query(RunningInstance).filter(
                        RunningInstance.proxy_model_name == proxy_name
                    ).all()
                    
                    for instance in instances_to_remove:
                        # Update model.is_active
                        model = db.query(Model).filter(Model.id == instance.model_id).first()
                        if model:
                            model.is_active = False
                        
                        # Remove the RunningInstance
                        db.delete(instance)
                        logger.info(f"Removed stopped model '{proxy_name}' from database")
                
                db.commit()
                logger.debug(f"Database sync completed. Added: {len(missing_in_db)}, Removed: {len(missing_in_external)}")
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error syncing database with external models: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _find_model_by_proxy_name(self, db: Session, proxy_name: str) -> Optional[Model]:
        """Find a model in the database by matching the proxy name"""
        try:
            # Use the stored proxy_name field for direct lookup
            model = db.query(Model).filter(Model.proxy_name == proxy_name).first()
            
            if model:
                return model
            
            logger.warning(f"Could not find model for proxy name: {proxy_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error finding model by proxy name '{proxy_name}': {e}")
            return None
    
    # API methods for HTTP endpoints
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status (for HTTP API)"""
        cpu_percent = psutil.cpu_percent(interval=0)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/app/data')
        
        db = SessionLocal()
        try:
            running_instances = db.query(RunningInstance).all()
            active_instances = []
            for instance in running_instances:
                active_instances.append({
                    "id": instance.id,
                    "model_id": instance.model_id,
                    "port": instance.port,
                    "proxy_model_name": instance.proxy_model_name,
                    "started_at": instance.started_at
                })
        finally:
            db.close()
        
        return {
            "system": {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                    "free": memory.free
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                }
            },
            "running_instances": active_instances,
            "proxy_status": {
                "enabled": True,
                "port": 2000,
                "endpoint": "http://localhost:2000/v1/chat/completions"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_running_models(self) -> List[Dict[str, Any]]:
        """Get currently running models from llama-swap"""
        try:
            return await self.llama_swap_client.get_running_models()
        except Exception as e:
            logger.debug(f"Failed to get running models from llama-swap: {e}")
            return []
    
    async def unload_all_models(self) -> Dict[str, Any]:
        """Unload all models via llama-swap"""
        try:
            return await self.llama_swap_client.unload_all_models()
        except Exception as e:
            logger.error(f"Failed to unload all models: {e}")
            return {"error": str(e)}
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get llama-swap and system health status"""
        try:
            llama_swap_healthy = await self.llama_swap_client.check_health()
        except Exception as e:
            logger.error(f"Failed to check llama-swap health: {e}")
            llama_swap_healthy = False
        
        return {
            'llama_swap_proxy': 'healthy' if llama_swap_healthy else 'unhealthy',
            'monitoring_active': self.is_running,
            'active_connections': len(websocket_manager.active_connections)
        }
    
    def get_recent_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent logs from monitor buffer"""
        logs = list(self.recent_logs)
        return logs[-limit:]
    
    def add_log(self, log_event: Dict[str, Any]):
        """Add a log event to the buffer"""
        self.recent_logs.append(log_event)


# Global unified monitor instance
unified_monitor = UnifiedMonitor()