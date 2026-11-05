/** Confirmation pair from a 409 that withholds a repeat of earlier work. */
export function withheldConfirmation(error) {
  const detail = error?.response?.data?.detail
  if (!detail || typeof detail !== 'object' || Array.isArray(detail)) return null
  if (detail.code !== 'ACTION_RETRY_WITHHELD') return null
  const operationId = detail.operation_id
  const stateToken = detail.state_token
  if (!operationId || !stateToken) return null
  return {
    confirm_operation_id: String(operationId),
    confirm_state: String(stateToken),
    message: typeof detail.message === 'string' && detail.message
      ? detail.message
      : 'Prior work may already have happened. Confirm this operation and its current state before trying again.',
  }
}

/** Next delete request after a retained launch generation, or a withheld retry. */
export function versionDeleteRetry(error) {
  const detail = error?.response?.data?.detail
  if (!detail || typeof detail !== 'object' || Array.isArray(detail)) return null
  if (detail.code === 'RETAINED_LAUNCH_REFERENCE') {
    const params = { retire_launch_references: true }
    if (detail.confirm_operation_id && detail.state_token) {
      params.confirm_operation_id = String(detail.confirm_operation_id)
      params.confirm_state = String(detail.state_token)
    }
    return {
      message: detail.message
        || 'This install is still referenced by a published or retained launch generation. Confirm to retire those generations and delete the version.',
      params,
    }
  }
  const held = withheldConfirmation(error)
  if (!held) return null
  return {
    message: held.message,
    params: {
      retire_launch_references: true,
      confirm_operation_id: held.confirm_operation_id,
      confirm_state: held.confirm_state,
    },
  }
}

export function versionDeleteErrorText(error) {
  const detail = error?.response?.data?.detail
  if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
    return detail.message || detail.code || 'Delete failed'
  }
  return detail || error?.message || 'Delete failed'
}
