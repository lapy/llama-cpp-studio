import { onMounted, onUnmounted, ref } from 'vue'

/**
 * A ref that follows the wall clock on an interval and stops on unmount.
 * Used for relative ages ("updated 2s ago") without leaking timers.
 */
export function useNow(intervalMs = 1000) {
  const now = ref(Date.now())
  let timer = null

  onMounted(() => {
    now.value = Date.now()
    timer = setInterval(() => {
      now.value = Date.now()
    }, intervalMs)
  })

  onUnmounted(() => {
    if (timer) clearInterval(timer)
    timer = null
  })

  return now
}
