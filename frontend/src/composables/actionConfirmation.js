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
