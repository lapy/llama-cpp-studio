/**
 * Build-settings fallbacks — GENERATED, do not hand-edit.
 *
 * Source of truth: the backend build-option catalogs (the llama-family engine
 * module and the audio.cpp engine module). Regenerate with
 * `python scripts/export-build-defaults.py`; `--check` runs in the gates so a
 * backend default change cannot silently drift from the frontend fallback.
 */

import generated from './build-defaults.generated.json'

export const FALLBACK_BUILD_DEFAULTS = Object.freeze({ ...generated.defaults })

export const FALLBACK_BUILD_TYPES = Object.freeze([...generated.build_types])
