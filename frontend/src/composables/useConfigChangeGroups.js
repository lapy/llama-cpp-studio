import { computed } from 'vue'

import { buildConfigChanges } from '@/utils/configChanges'
import { formatLabel } from '@/utils/modelPresentation'

/**
 * Human-readable parameter diffs between the live form and the last saved
 * config. Extracted from ModelConfig so the view keeps its line budget and the
 * diff snapshot rules live next to the change builder.
 */
export function useConfigChangeGroups({
  config,
  savedConfig,
  catalogParamByKey,
  engineOptions,
  hasUnsavedChanges,
  loading,
  buildEngineStashFromForm,
}) {
  return computed(() => {
    if (!hasUnsavedChanges.value || loading.value) return []
    try {
      const paramLabels = {}
      for (const [key, param] of catalogParamByKey.value.entries()) {
        paramLabels[key] = param.label || formatLabel(key)
      }
      const engineLabel =
        engineOptions.value.find((option) => option.value === config.value.engine)?.label ||
        config.value.engine
      // Snapshot without mutating the live form (buildPersistedPayload syncs UI
      // rows into config and must stay outside a render computation).
      const snapshot = (source, fromStoredSection = false) => {
        const engines =
          source.engines && typeof source.engines === 'object'
            ? JSON.parse(JSON.stringify(source.engines))
            : {}
        engines[source.engine] = fromStoredSection
          ? JSON.parse(JSON.stringify(source.engines?.[source.engine] || {}))
          : buildEngineStashFromForm(source)
        return { engine: source.engine, engines }
      }
      return buildConfigChanges(snapshot(config.value), snapshot(savedConfig.value, true), {
        paramLabels,
        engineLabel,
      })
    } catch {
      return []
    }
  })
}
