/** Options for binding a llama-family source build to an installed CUDA toolkit. */
export function cudaToolkitOptions(cudaStatus) {
  const seen = new Set([''])
  const options = [{ label: 'Active Studio toolkit', value: '' }]
  for (const item of cudaStatus?.installed_versions || []) {
    const version = String(item?.version || '').trim()
    if (!version || seen.has(version)) continue
    seen.add(version)
    options.push({
      label: `CUDA ${version}${item?.is_current ? ' (active)' : ''}`,
      value: version,
    })
  }
  return options
}
