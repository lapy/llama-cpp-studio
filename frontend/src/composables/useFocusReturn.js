import { nextTick, watch } from 'vue'

const MAIN_CONTENT_ID = 'main-content'

/**
 * Remember the control that opened a dialog and return focus there when it closes.
 * If that control is gone, focus the main landmark.
 */
export function useFocusReturn() {
  let opener = null

  function remember(target) {
    const node = target || (typeof document !== 'undefined' ? document.activeElement : null)
    opener = node instanceof HTMLElement ? node : null
  }

  function restore() {
    const current = opener
    opener = null
    const stillThere = current && typeof document !== 'undefined' && document.contains(current)
    const target = stillThere
      ? current
      : (typeof document !== 'undefined' ? document.getElementById(MAIN_CONTENT_ID) : null)
    target?.focus?.()
  }

  return { remember, restore }
}

/**
 * Capture focus before a dialog opens and restore it after the dialog closes.
 * The watcher runs before the dialog takes focus.
 */
export function watchDialogFocus(visible) {
  const focus = useFocusReturn()
  watch(visible, (open, wasOpen) => {
    if (open && !wasOpen) focus.remember()
    if (!open && wasOpen) nextTick(() => focus.restore())
  })
  return focus
}
