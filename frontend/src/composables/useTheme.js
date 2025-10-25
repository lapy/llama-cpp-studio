import { ref, computed, watch } from 'vue'

export function useTheme() {
  const theme = ref(localStorage.getItem('theme') || 'dark')
  
  const isDark = computed(() => theme.value === 'dark')
  const isLight = computed(() => theme.value === 'light')
  
  const setTheme = (newTheme) => {
    if (newTheme === 'dark' || newTheme === 'light') {
      theme.value = newTheme
      localStorage.setItem('theme', newTheme)
      document.documentElement.setAttribute('data-theme', newTheme)
    }
  }
  
  const toggleTheme = () => {
    setTheme(isDark.value ? 'light' : 'dark')
  }
  
  // Initialize theme on mount
  const initTheme = () => {
    document.documentElement.setAttribute('data-theme', theme.value)
  }
  
  // Watch for theme changes
  watch(theme, (newTheme) => {
    document.documentElement.setAttribute('data-theme', newTheme)
  }, { immediate: true })
  
  return {
    theme,
    isDark,
    isLight,
    setTheme,
    toggleTheme,
    initTheme
  }
}
