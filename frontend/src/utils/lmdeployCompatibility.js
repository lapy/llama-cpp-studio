export function lmdeployCompatibilityFor(gpuInfo) {
  const gpus = Array.isArray(gpuInfo?.gpus) ? gpuInfo.gpus : []
  const sm70Devices = gpus.filter((gpu) =>
    String(gpu?.compute_capability || '').startsWith('7.0'),
  )
  if (!sm70Devices.length) return { show: false, options: [] }

  const names = [...new Set(sm70Devices.map((gpu) => gpu.name).filter(Boolean))]
  return {
    show: true,
    note: `Detected SM70 hardware (${names.join(', ') || 'compute capability 7.0'}); this install option supplies Volta kernels. Backend and eager mode are runtime parameters, not install settings.`,
    options: [
      {
        key: 'compat_sm70',
        label: 'Use CUDA 12 PyTorch with sm70 kernels',
        hint: 'The default CUDA 13 torch wheel has no Volta kernels. Installs the declared CUDA 12 torch build into the LMDeploy venv. TurboMind and PyTorch remain runtime choices.',
      },
    ],
  }
}
