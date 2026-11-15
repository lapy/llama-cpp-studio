/**
 * Conservative CUDA toolkit support window by NVIDIA compute capability.
 *
 * CUDA 11.x still targets Kepler (3.5); CUDA 12.x dropped Kepler and needs
 * Maxwell (5.0) or newer; CUDA 13.x dropped Maxwell, Pascal, and Volta and
 * needs Turing (7.5) or newer. Volta (compute capability 7.0)
 * therefore cannot use a CUDA 13.x toolkit.
 *
 * These are guidance-only checks: the backend remains free to install or
 * activate anything, and an unknown capability never produces a warning.
 */

/** Lowest compute capability each documented CUDA release line can target. */
const MINIMUM_COMPUTE_CAPABILITY = {
  11: 3.5,
  12: 5.0,
  13: 7.5,
}

/** Architecture name for the release line, used in the guidance message. */
const ARCHITECTURE_LABEL = {
  11: 'Kepler',
  12: 'Maxwell',
  13: 'Turing',
}

/** Parse a release line ("13") or concrete patch version ("13.5.0"). */
function releaseMajor(version) {
  if (version == null) return null
  const match = String(version)
    .trim()
    .match(/^(\d+)(?:\.|$)/)
  return match ? Number(match[1]) : null
}

/** Numeric [major, minor, patch] tuple for deterministic version ordering. */
function versionParts(version) {
  const match = String(version ?? '')
    .trim()
    .match(/^(\d+)(?:\.(\d+))?(?:\.(\d+))?/)
  if (!match) return null
  return [Number(match[1]), Number(match[2] ?? 0), Number(match[3] ?? 0)]
}

function compareVersionParts(left, right) {
  for (let index = 0; index < 3; index += 1) {
    if (left[index] !== right[index]) return left[index] - right[index]
  }
  return 0
}

function formatCapability(value) {
  return Number(value).toFixed(1)
}

/**
 * Lowest compute capability a CUDA version can target, or null for a version
 * outside the documented 11.x/12.x/13.x release lines.
 */
export function minimumComputeCapability(version) {
  const major = releaseMajor(version)
  return major != null ? (MINIMUM_COMPUTE_CAPABILITY[major] ?? null) : null
}

/**
 * Compute capability recorded by the GPU detector ("7.0"), or null when the
 * backend did not report it (for example a partial nvidia-smi fallback).
 */
export function gpuComputeCapability(gpu) {
  const raw = gpu?.compute_capability
  if (raw == null || raw === '') return null
  const value = typeof raw === 'number' ? raw : Number(String(raw).trim())
  return Number.isFinite(value) && value > 0 ? value : null
}

function gpuName(gpu, index) {
  const name = typeof gpu?.name === 'string' ? gpu.name.trim() : ''
  return name || `GPU ${index + 1}`
}

function describeGpus(gpus) {
  if (gpus.length === 1) {
    return `${gpus[0].name} is ${formatCapability(gpus[0].capability)}`
  }
  const details = gpus.map((gpu) => `${gpu.name} (${formatCapability(gpu.capability)})`).join(', ')
  return `${details} are below that`
}

function newestCompatibleVersion(versions, capabilityFloor) {
  let best = null
  let bestParts = null
  for (const entry of versions) {
    if (entry.minimum > capabilityFloor) continue
    const parts = versionParts(entry.version)
    if (!parts) continue
    if (!bestParts || compareVersionParts(parts, bestParts) > 0) {
      best = entry.version
      bestParts = parts
    }
  }
  return best
}

/**
 * Guidance for CUDA versions that cannot target one or more detected GPUs.
 *
 * Returns one `{ version, minimum, gpus, message }` per incompatible version,
 * newest version first. Versions with an unknown release line, GPUs with an
 * unknown capability, and fleets with no GPUs produce no warnings. Whether a
 * mixed fleet is incompatible is judged GPU by GPU, so one Volta card is
 * enough to flag CUDA 13.x even next to a newer card.
 */
export function cudaCompatibilityWarnings(versions, gpus) {
  const knownGpus = (Array.isArray(gpus) ? gpus : [])
    .map((gpu, index) => ({ name: gpuName(gpu, index), capability: gpuComputeCapability(gpu) }))
    .filter((gpu) => gpu.capability != null)
  if (!knownGpus.length) return []

  const candidates = (Array.isArray(versions) ? versions : [])
    .map((version) => {
      const normalized = version == null ? '' : String(version).trim()
      return {
        version: normalized,
        minimum: minimumComputeCapability(normalized),
      }
    })
    .filter((entry) => entry.version && entry.minimum != null)
  if (!candidates.length) return []

  // A recommendation must run on the oldest detected GPU in the fleet.
  const capabilityFloor = Math.min(...knownGpus.map((gpu) => gpu.capability))
  const recommendation = newestCompatibleVersion(candidates, capabilityFloor)

  const seen = new Set()
  const warnings = []
  for (const entry of candidates) {
    if (seen.has(entry.version)) continue
    seen.add(entry.version)
    const incompatible = knownGpus.filter((gpu) => gpu.capability < entry.minimum)
    if (!incompatible.length) continue
    const major = releaseMajor(entry.version)
    const needs = `CUDA ${major}.x needs ${ARCHITECTURE_LABEL[major]} (${formatCapability(entry.minimum)})+`
    const advice = recommendation ? ` Choose ${recommendation} or older.` : ''
    warnings.push({
      version: entry.version,
      minimum: entry.minimum,
      gpus: incompatible.map((gpu) => gpu.name),
      message: `${needs}; ${describeGpus(incompatible)}.${advice}`,
    })
  }

  return warnings.sort((left, right) =>
    compareVersionParts(versionParts(right.version), versionParts(left.version)),
  )
}
