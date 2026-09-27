import { onBeforeUnmount, onMounted, ref } from 'vue'
import { onBeforeRouteLeave } from 'vue-router'

const DRAFT_PREFIX = 'llama-studio.config-draft.'

export function draftStorageKey(modelId, engine) {
  return `${DRAFT_PREFIX}${modelId}.${engine || 'default'}`
}

export function readDraft(modelId, engine) {
  if (typeof sessionStorage === 'undefined' || !modelId) return null
  try {
    const raw = sessionStorage.getItem(draftStorageKey(modelId, engine))
    if (!raw) return null
    const parsed = JSON.parse(raw)
    if (!parsed || typeof parsed !== 'object' || !parsed.config) return null
    return parsed
  } catch {
    return null
  }
}

export function writeDraft(modelId, engine, payload) {
  if (typeof sessionStorage === 'undefined' || !modelId) return
  sessionStorage.setItem(draftStorageKey(modelId, engine), JSON.stringify(payload))
}

export function clearDraft(modelId, engine) {
  if (typeof sessionStorage === 'undefined' || !modelId) return
  sessionStorage.removeItem(draftStorageKey(modelId, engine))
}

/**
 * Blocks in-app navigation and browser unload while `isDirty` is true.
 * `isDirty` is read at navigation time so it can close over a computed.
 */
export function useDraftGuard(isDirty) {
  const leavePromptVisible = ref(false)
  let resolveLeave = null

  onBeforeRouteLeave(() => {
    if (!isDirty()) return true
    leavePromptVisible.value = true
    return new Promise((resolve) => {
      resolveLeave = resolve
    })
  })

  function finishLeave(allow) {
    leavePromptVisible.value = false
    const resolve = resolveLeave
    resolveLeave = null
    if (resolve) resolve(Boolean(allow))
  }

  function onBeforeUnload(event) {
    if (!isDirty()) return
    event.preventDefault()
    event.returnValue = ''
  }

  onMounted(() => window.addEventListener('beforeunload', onBeforeUnload))
  onBeforeUnmount(() => {
    window.removeEventListener('beforeunload', onBeforeUnload)
    if (resolveLeave) {
      resolveLeave(false)
      resolveLeave = null
    }
  })

  return { leavePromptVisible, finishLeave }
}
