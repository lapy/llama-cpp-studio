import { nextTick, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'

/**
 * The identity-first landing section. Every engine adds it ahead of its own
 * tabs so `#overview` is a stable deep link on text and audio pages alike.
 */
export const MODEL_CONFIG_OVERVIEW_TAB = Object.freeze({
  id: 'overview',
  label: 'Overview',
  icon: 'pi pi-th-large',
})

function normalizeHashSection(hash) {
  const raw = String(hash ?? '')
    .trim()
    .replace(/^#/, '')
  if (!raw) return ''
  try {
    return decodeURIComponent(raw).toLowerCase()
  } catch {
    return raw.toLowerCase()
  }
}

/** Stable URL hash for a tab, based on the label the user sees. */
export function sectionHashForTab(tab) {
  if (!tab) return ''
  const label = String(tab.label ?? '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
  return label || String(tab.id ?? '')
}

/** Resolve `#runtime` (label) and `#server` (id) to the active tab id. */
export function tabIdForHash(hash, tabs) {
  const wanted = normalizeHashSection(hash)
  if (!wanted || !Array.isArray(tabs)) return ''
  const byId = tabs.find((tab) => String(tab?.id ?? '').toLowerCase() === wanted)
  if (byId) return byId.id
  return tabs.find((tab) => sectionHashForTab(tab) === wanted)?.id || ''
}

/**
 * Keeps ModelConfig's tab selection in sync with the route hash.
 * The tab is switched before scrolling so panels rendered inside a tab exist.
 */
export function useModelConfigSections({ tabs, activeTab, ready, panelId }) {
  const route = useRoute()
  const router = useRouter()
  let handledSection = ''

  async function scrollToSection(sectionId) {
    if (typeof document === 'undefined') return
    await nextTick()
    const panel = document.getElementById(panelId(sectionId))
    panel?.scrollIntoView?.({ block: 'start' })
  }

  watch(
    [() => route.hash, tabs, ready],
    ([hash, availableTabs, isReady]) => {
      if (!isReady) return
      if (!hash) {
        if (handledSection) {
          activeTab.value = availableTabs?.[0]?.id || activeTab.value
          handledSection = ''
        }
        return
      }
      const sectionId = tabIdForHash(hash, availableTabs)
      if (!sectionId) return
      const sectionKey = `${route.path}:${hash}:${sectionId}`
      const changed = activeTab.value !== sectionId
      activeTab.value = sectionId
      if (!changed && handledSection === sectionKey) return
      handledSection = sectionKey
      void scrollToSection(sectionId)
    },
    { immediate: true, flush: 'post' },
  )

  function selectTab(tab) {
    if (!tab?.id) return
    activeTab.value = tab.id
    const hash = `#${sectionHashForTab(tab)}`
    if (route.hash === hash) {
      handledSection = `${route.path}:${hash}:${tab.id}`
      void scrollToSection(tab.id)
      return
    }
    const navigation = router.push({ path: route.path, query: route.query, hash })
    if (navigation && typeof navigation.catch === 'function') navigation.catch(() => {})
  }

  return { selectTab }
}
