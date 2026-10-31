/** Copy only on an explicit user action. Never read the user's clipboard. */
export async function copyText(text, { container } = {}) {
  const value = String(text ?? '')
  if (!value) throw new Error('There is no text to copy.')
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value)
      return
    }
  } catch {
    // Permission denial and HTTP-only deployments still need a checked fallback.
  }

  const active = document.activeElement
  const selection = window.getSelection()
  const ranges = selection
    ? Array.from({ length: selection.rangeCount }, (_, i) => selection.getRangeAt(i).cloneRange())
    : []
  const start = active?.selectionStart
  const end = active?.selectionEnd
  const direction = active?.selectionDirection
  // Keep the transfer field inside an open dialog's focus trap.
  const parent =
    container || active?.closest?.('[role="dialog"], [aria-modal="true"]') || document.body
  const area = document.createElement('textarea')
  area.value = value
  area.readOnly = true
  area.tabIndex = -1
  area.setAttribute('aria-label', 'Clipboard transfer')
  Object.assign(area.style, {
    position: 'fixed',
    left: '0',
    top: '0',
    width: '1px',
    height: '1px',
    opacity: '0',
    padding: '0',
    border: '0',
  })
  try {
    parent.appendChild(area)
    area.focus({ preventScroll: true })
    area.select()
    area.setSelectionRange(0, value.length)
    // execCommand is a legacy compatibility path for insecure LAN origins.
    // It may fail silently: a false result must never become a success message.
    if (typeof document.execCommand !== 'function' || document.execCommand('copy') !== true) {
      throw new Error('Automatic copying is unavailable. Select the text and copy it manually.')
    }
  } finally {
    area.remove()
    if (active?.isConnected) {
      active.focus?.({ preventScroll: true })
      if (typeof start === 'number' && typeof end === 'number')
        active.setSelectionRange?.(start, end, direction)
    }
    if (selection) {
      selection.removeAllRanges()
      for (const range of ranges) {
        if (range.commonAncestorContainer.isConnected) selection.addRange(range)
      }
    }
  }
}
