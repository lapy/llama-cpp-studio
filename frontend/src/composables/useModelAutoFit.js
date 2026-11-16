import { computed } from 'vue'

import { buildAutoFitState } from '@/utils/modelAutoFit'

/**
 * Auto-fit state for ModelConfig.
 *
 * Detection is capability-based: the engine advertises `fit` in its scanned
 * parameter list, so no engine ids appear here. Restoration only touches the
 * draft and only the three settings that override auto-fit; persistence stays with
 * the action bar.
 */
export function useModelAutoFit({
  enabled,
  parameters,
  config,
  activeParamKeys,
  catalogParamByKey,
  toast,
}) {
  const state = computed(() =>
    buildAutoFitState({
      parameters: enabled?.value ? parameters?.value : [],
      config: config?.value,
    }),
  )

  /**
   * Restore the engine's auto-fit defaults on the draft:
   * - fit -> "on" when the control offers it, otherwise clear the override
   * - ctx_size -> 0 (explicitly "loaded/fitted by the engine")
   * - n_gpu_layers -> clear the override (the numeric control cannot hold "auto")
   *
   * Returns the number of settings restored.
   */
  function restore() {
    if (!state.value.supported || !state.value.overrides.length) return 0
    let restored = 0
    for (const override of state.value.overrides) {
      if (!catalogParamByKey?.value?.has(override.key)) continue
      if (override.restore.mode === 'clear') {
        // Deleting the key is the view's existing "use engine default" action.
        delete config.value[override.key]
      } else {
        config.value[override.key] = override.restore.value
        if (!activeParamKeys.value.includes(override.key)) {
          activeParamKeys.value = [...activeParamKeys.value, override.key]
        }
      }
      restored += 1
    }
    if (!restored) return 0
    toast?.add({
      severity: 'success',
      summary: 'Auto settings restored',
      detail: `${restored} explicit ${restored === 1 ? 'override was' : 'overrides were'} cleared from the draft. Save changes to keep them; nothing has been applied yet.`,
      life: 4000,
    })
    return restored
  }

  return { state, restore }
}
