from fastapi import APIRouter
import subprocess
import json
import os
import multiprocessing
from typing import Dict, List, Optional

router = APIRouter()


@router.get("/gpu-info")
async def get_gpu_info():
    """Detect GPU information and CPU capabilities"""
    gpu_info = {
        "cuda_version": "N/A",
        "device_count": 0,
        "gpus": [],
        "total_vram": 0,
        "available_vram": 0,
        "cpu_only_mode": True,
        "cpu_threads": multiprocessing.cpu_count()
    }
    
    # Try to detect NVIDIA GPUs
    try:
        import pynvml
        pynvml.nvmlInit()
        
        device_count = pynvml.nvmlDeviceGetCount()
        gpus = []
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Get basic info
            name_bytes = pynvml.nvmlDeviceGetName(handle)
            # Handle both bytes and string return types from pynvml
            if isinstance(name_bytes, bytes):
                name = name_bytes.decode('utf-8')
            else:
                name = name_bytes
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Get compute capability
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            compute_capability = f"{major}.{minor}"
            
            # Get temperature
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temperature = None
            
            # Get utilization
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
                memory_util = utilization.memory
            except:
                gpu_util = None
                memory_util = None
            
            gpu_info_item = {
                "index": i,
                "name": name,
                "memory": {
                    "total": memory_info.total,
                    "free": memory_info.free,
                    "used": memory_info.used
                },
                "compute_capability": compute_capability,
                "temperature": temperature,
                "utilization": {
                    "gpu": gpu_util,
                    "memory": memory_util
                }
            }
            
            gpus.append(gpu_info_item)
        
        # Get CUDA version
        try:
            cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
            cuda_version_str = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
        except:
            cuda_version_str = "Unknown"
        
        gpu_info.update({
            "cuda_version": cuda_version_str,
            "device_count": device_count,
            "gpus": gpus,
            "total_vram": sum(gpu["memory"]["total"] for gpu in gpus),
            "available_vram": sum(gpu["memory"]["free"] for gpu in gpus),
            "cpu_only_mode": device_count == 0
        })
        
    except ImportError:
        # pynvml not available, try nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu,utilization.memory,compute_cap", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True
            )
            
            gpus = []
            lines = result.stdout.strip().split('\n')
            
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 9:
                    gpu_info_item = {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory": {
                            "total": int(parts[2]) * 1024 * 1024 * 1024,  # Convert GB to bytes
                            "free": int(parts[3]) * 1024 * 1024 * 1024,
                            "used": int(parts[4]) * 1024 * 1024 * 1024
                        },
                        "compute_capability": parts[8] if len(parts) > 8 else "Unknown",
                        "temperature": int(parts[5]) if parts[5] != "N/A" else None,
                        "utilization": {
                            "gpu": int(parts[6]) if parts[6] != "N/A" else None,
                            "memory": int(parts[7]) if parts[7] != "N/A" else None
                        }
                    }
                    gpus.append(gpu_info_item)
            
            gpu_info.update({
                "device_count": len(gpus),
                "gpus": gpus,
                "total_vram": sum(gpu["memory"]["total"] for gpu in gpus),
                "available_vram": sum(gpu["memory"]["free"] for gpu in gpus),
                "cpu_only_mode": len(gpus) == 0
            })
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            # No NVIDIA GPU detected, CPU-only mode
            pass
    
    except Exception:
        # Any other error, assume CPU-only mode
        pass
    
    # Check for NVLink topology if GPUs are available
    if gpu_info["device_count"] > 1:
        try:
            result = subprocess.run(
                ["nvidia-smi", "nvlink", "--query-topology", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse NVLink topology
            nvlink_info = {
                "has_nvlink": True,
                "topology": result.stdout.strip(),
                "recommended_strategy": "Multi-GPU with NVLink" if "NV" in result.stdout else "Multi-GPU without NVLink"
            }
            gpu_info["nvlink_topology"] = nvlink_info
            
        except:
            gpu_info["nvlink_topology"] = {
                "has_nvlink": False,
                "recommended_strategy": "Multi-GPU without NVLink"
            }
    
    return gpu_info