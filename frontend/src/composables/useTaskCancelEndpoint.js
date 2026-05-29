/** Resolve the manager-specific cancel API for a progress task. */

const MANAGER_CANCEL_ENDPOINTS = {
  cuda: '/api/llama-versions/cuda/cancel',
  lmdeploy: '/api/lmdeploy/cancel',
  onecat_vllm: '/api/1cat-vllm/cancel',
}

const TYPE_CANCEL_ENDPOINTS = {
  build: '/api/llama-versions/build-cancel',
  download: '/api/models/downloads/cancel',
}

export function cancelEndpointForTask(task) {
  if (!task) return null
  const manager = task.metadata?.manager
  if (manager && MANAGER_CANCEL_ENDPOINTS[manager]) {
    return MANAGER_CANCEL_ENDPOINTS[manager]
  }
  if (task.type && TYPE_CANCEL_ENDPOINTS[task.type]) {
    return TYPE_CANCEL_ENDPOINTS[task.type]
  }
  return null
}
