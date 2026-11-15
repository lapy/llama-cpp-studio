import { isAudioPackage } from '@/utils/audioModel'

/**
 * Announce a completed download so the library leads the new model with Set up.
 *
 * The library refreshes its catalog before announcing, so this only reads the
 * refreshed model. One toast per model per library mount keeps a poll or a
 * repeated completion event from stacking the same guidance.
 */
export function createDownloadSetupAnnouncer({ modelStore, toast }) {
  const announced = new Set()

  return function announceDownloadedSetup(payload) {
    if (payload?.status !== 'completed' || !payload?.model_id) return
    const model = (modelStore.allQuantizations || []).find(
      (quant) => String(quant?.id) === String(payload.model_id),
    )
    if (!model || model.config_reviewed !== false || isAudioPackage(model)) return
    const key = String(model.id)
    if (announced.has(key)) return
    announced.add(key)
    toast.add({
      severity: 'success',
      summary: 'Model downloaded',
      detail: 'Set it up to start.',
      life: 10000,
      group: 'model-library',
      configureModelId: model.id,
    })
  }
}
