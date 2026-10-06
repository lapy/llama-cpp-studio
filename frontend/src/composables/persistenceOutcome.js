// Classifies a failed save or recovery response.
//
// committed true means the server observed the file replacement. It does not
// mean the directory entry was made crash-durable. committed false means the
// previous document is unchanged. "unknown" means the outcome could not be
// established. Nothing here retries a request by itself.

export function classifyPersistenceError(error) {
  const data = error?.response?.data
  if (!data || typeof data !== 'object') {
    return {
      outcome: 'unknown',
      code: null,
      committed: 'unknown',
      summary: 'Save outcome unknown',
      detail: 'The save outcome could not be established. Your edits are still here. Refresh before trying again.',
      preserveEdits: true,
      refresh: true,
      retry: false,
      success: false,
    }
  }
  if (data.code === 'STORE_QUEUE_FULL') {
    return {
      outcome: 'rejected',
      code: 'STORE_QUEUE_FULL',
      committed: false,
      summary: 'Save paused',
      detail: 'Persistence is busy, so this save was not stored. Your edits are still here. Try again when you are ready.',
      preserveEdits: true,
      refresh: false,
      retry: true,
      success: false,
    }
  }
  if (data.code === 'STORE_WRITE_FAILED' && data.committed === true) {
    return {
      outcome: 'replaced',
      code: 'STORE_WRITE_FAILED',
      committed: true,
      summary: 'Saved, acknowledgement failed',
      detail: 'The document was replaced, but acknowledgement failed. Refresh before trying again.',
      preserveEdits: true,
      refresh: true,
      retry: false,
      success: false,
    }
  }
  if (data.code === 'STORE_WRITE_FAILED' && data.committed === false) {
    return {
      outcome: 'not_saved',
      code: 'STORE_WRITE_FAILED',
      committed: false,
      summary: 'Not saved',
      detail: 'The save was not stored. The previous state is unchanged. If this was the first save, no document was written. Your edits are still here.',
      preserveEdits: true,
      refresh: false,
      retry: true,
      success: false,
    }
  }
  if (data.code === 'STORE_WRITE_FAILED' || data.code === 'RECONCILE_UNKNOWN' || data.committed === 'unknown' || data.outcome === 'unknown') {
    return {
      outcome: 'unknown',
      code: data.code || null,
      committed: 'unknown',
      summary: 'Outcome unknown',
      detail: 'The outcome could not be established. Your edits are still here. Refresh before trying again.',
      preserveEdits: true,
      refresh: true,
      retry: false,
      success: false,
    }
  }
  return null
}

export function noteDocumentSaveFailure(toast, error) {
  const outcome = classifyPersistenceError(error)
  if (!outcome) return null
  toast.add({
    severity: outcome.committed === true ? 'warn' : 'error',
    summary: outcome.summary,
    detail: outcome.detail,
    life: 6000,
  })
  return outcome
}

export function saveHeldForRefresh(notice) {
  if (!notice?.refresh) return false
  return notice.committed === true || notice.committed === 'unknown'
}
