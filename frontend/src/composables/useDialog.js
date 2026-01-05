import { ref } from 'vue'

/**
 * Composable for managing dialog visibility state
 * @param {boolean} initialVisible - Initial visibility state
 * @returns {Object} Dialog state and methods
 */
export function useDialog(initialVisible = false) {
  const visible = ref(initialVisible)

  const open = () => {
    visible.value = true
  }

  const close = () => {
    visible.value = false
  }

  const toggle = () => {
    visible.value = !visible.value
  }

  return {
    visible,
    open,
    close,
    toggle
  }
}

