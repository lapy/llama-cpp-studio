import { computed, ref, watch } from 'vue'

import { formatLabel, modelRuntimeState } from '@/utils/modelPresentation'

/**
 * Action-bar state for the model configuration page.
 *
 * The page owns persistence and runtime-plan loading; this composable turns
 * those signals into the bar's status pill, save/start enablement, and the
 * start action, so the bar and the overview review block share one truth.
 */
export function useModelConfigActionBar({
  router,
  toast,
  modelStore,
  enginesStore,
  model,
  config,
  saving,
  hasUnsavedChanges,
  modelIsRunning,
  modelRuntimeQuality,
  showApplyRuntime,
  publishedStepLabel,
  runtimeStateDetail,
  configSaveHeld,
  requestAliasesValid,
  persistenceNotice,
  persistenceAction,
  unrecognizedSavedKeys,
  modelLaunchPlan,
  engineOptions,
  hasUsableEngine,
  modelFormat,
  refreshModelFromStore,
  formatError,
}) {
  const startingModelRequest = ref(false)
  const startModelError = ref('')

  const modelIdentityLabel = computed(
    () => model.value?.display_name || model.value?.base_model_name || 'This model',
  )
  const modelFormatLabel = computed(() => formatLabel(modelFormat.value))
  const currentEngineLabel = computed(() => {
    const engine = engineOptions.value.find((option) => option.value === config.value.engine)
    return engine?.label || config.value.engine || ''
  })

  const observedStarting = computed(() => modelRuntimeState(model.value).key === 'loading')
  const isStartingModel = computed(() => startingModelRequest.value || observedStarting.value)

  watch(hasUnsavedChanges, (dirty, wasDirty) => {
    if (dirty || (wasDirty && !dirty)) startModelError.value = ''
  })
  watch(modelIsRunning, (running) => {
    if (running) startModelError.value = ''
  })

  const pageRequiresSave = computed(
    () =>
      Boolean(persistenceNotice.value?.retry && persistenceAction.value === 'save') ||
      unrecognizedSavedKeys.value.length > 0,
  )
  const canSaveConfig = computed(
    () =>
      (hasUnsavedChanges.value || pageRequiresSave.value) && !saving.value && !configSaveHeld.value,
  )
  const saveConfigHint = computed(() => {
    if (configSaveHeld.value) {
      return requestAliasesValid.value
        ? 'Saving is paused until the reported persistence problem is resolved.'
        : 'Fix the request aliases before saving.'
    }
    if (saving.value) return 'Saving changes…'
    if (!hasUnsavedChanges.value && pageRequiresSave.value) {
      return 'Save to drop stored keys this page no longer recognizes.'
    }
    if (!hasUnsavedChanges.value) return 'No changes to save.'
    return 'Save this form. Apply settings separately when you want the model to use it.'
  })

  const startModelBlockedReason = computed(() => {
    if (configSaveHeld.value)
      return 'Resolve the configuration problem reported on this page before starting the model.'
    if (hasUnsavedChanges.value) return 'Save or discard these edits before starting the model.'
    if (modelLaunchPlan.value?.action === 'invalid')
      return 'Fix the configuration errors in Review before starting the model.'
    if (showApplyRuntime.value)
      return 'Apply the saved settings before starting so the model starts with them.'
    if (!hasUsableEngine.value) return 'Install or activate an engine before starting this model.'
    if (isStartingModel.value) return 'The model is already starting.'
    return ''
  })
  const canStartModel = computed(
    () => Boolean(model.value) && !modelIsRunning.value && !startModelBlockedReason.value,
  )

  const configurationStatus = computed(() => {
    if (isStartingModel.value) {
      return {
        key: 'starting',
        label: 'Starting',
        severity: 'warn',
        detail: 'The model is starting. This bar updates when it reports running.',
      }
    }
    if (startModelError.value) {
      return {
        key: 'failed',
        label: 'Failed',
        severity: 'danger',
        detail: `${startModelError.value} Fix the problem, then try Start model again.`,
      }
    }
    if (hasUnsavedChanges.value) {
      return {
        key: 'dirty',
        label: 'Unsaved changes',
        severity: 'warn',
        detail: runtimeStateDetail.value,
      }
    }
    if (showApplyRuntime.value) {
      return {
        key: 'pending',
        label: 'Saved · needs apply',
        severity: 'warn',
        detail: runtimeStateDetail.value,
      }
    }
    if (modelIsRunning.value) {
      return {
        key: 'running',
        label: 'Running',
        severity: 'success',
        detail: runtimeStateDetail.value,
      }
    }
    if (publishedStepLabel.value === 'Status unknown') {
      return {
        key: 'unknown',
        label: 'Status unknown',
        severity: 'warn',
        detail: runtimeStateDetail.value,
      }
    }
    if (modelRuntimeQuality.value === 'stale') {
      return {
        key: 'stale',
        label: publishedStepLabel.value,
        severity: 'warn',
        detail: runtimeStateDetail.value,
      }
    }
    return {
      key: 'applied',
      label: 'Applied',
      severity: 'success',
      detail: runtimeStateDetail.value,
    }
  })

  async function refreshRuntimeAfterStart() {
    refreshModelFromStore()
    await Promise.allSettled([
      enginesStore.fetchRuntimePlan?.(),
      enginesStore.fetchRuntimePlanState?.(),
    ])
  }

  /**
   * Start uses the catalog quantization id, exactly like the library's Start
   * action. The stable proxy id is for launch/routing only.
   */
  async function startModelFromConfig() {
    if (!canStartModel.value || startingModelRequest.value) return
    const modelId = model.value?.id
    if (!modelId) return
    startingModelRequest.value = true
    startModelError.value = ''
    try {
      await modelStore.startModel(modelId)
      await refreshRuntimeAfterStart()
      toast.add(
        modelIsRunning.value
          ? { severity: 'success', summary: 'Model started', life: 3000 }
          : {
              severity: 'info',
              summary: 'Model starting',
              detail: 'The page updates when this model reports running.',
              life: 4000,
            },
      )
    } catch (e) {
      startModelError.value = formatError(e) || 'The model did not start.'
      toast.add({
        severity: 'error',
        summary: 'Failed to start',
        detail: startModelError.value,
        life: 5000,
      })
    } finally {
      startingModelRequest.value = false
    }
  }

  function openModelInLibrary() {
    router.push('/models')
  }

  return {
    modelIdentityLabel,
    modelFormatLabel,
    currentEngineLabel,
    isStartingModel,
    canSaveConfig,
    saveConfigHint,
    startModelBlockedReason,
    canStartModel,
    configurationStatus,
    startModelFromConfig,
    openModelInLibrary,
  }
}
