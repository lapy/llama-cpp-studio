/**
 * R90: ONE audio-package predicate for every model surface.
 *
 * ModelRow, ModelLibrary, ModelWarmupButton and AudioWorkspace each carried the
 * same engine-or-format check (five copies in four files). Keeping it here means
 * the rule is stated once, and the drift surface shrinks accordingly.
 */

/** The engine/format marker that identifies an audio package. */
const AUDIO_PACKAGE = 'audio_cpp'

export function isAudioPackage(model) {
  if (!model) return false
  const engine = model.config?.engine || model.engine || ''
  const format = model.format || model.artifact?.format || ''
  return engine === AUDIO_PACKAGE || format === AUDIO_PACKAGE
}
