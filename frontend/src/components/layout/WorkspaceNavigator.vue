<template>
  <button
    class="workspace-search"
    type="button"
    :aria-label="`Search workspace ${shortcut}`"
    aria-haspopup="dialog"
    @click="open = true"
  >
    <i class="pi pi-search" aria-hidden="true" /><span>Search workspace</span
    ><kbd aria-hidden="true">{{ ' ' + shortcut }}</kbd>
  </button>
  <Dialog
    v-model:visible="open"
    modal
    header="Go to workspace"
    class="workspace-navigator"
    :style="{ width: '34rem', maxWidth: 'calc(100vw - 2rem)' }"
    @show="focusSearch"
  >
    <div class="navigator-input">
      <i class="pi pi-search" aria-hidden="true" /><input
        ref="searchInput"
        v-model="query"
        role="combobox"
        aria-label="Find a page or tool"
        aria-autocomplete="list"
        :aria-expanded="true"
        aria-controls="workspace-destinations"
        :aria-activedescendant="matches.length ? `workspace-destination-${selected}` : undefined"
        placeholder="Find a page or tool…"
        autocomplete="off"
        @keydown="onKey"
      />
    </div>
    <div
      id="workspace-destinations"
      class="navigator-results"
      role="listbox"
      aria-label="Workspace destinations"
    >
      <button
        v-for="(item, index) in matches"
        :id="`workspace-destination-${index}`"
        :key="item.to"
        type="button"
        role="option"
        :aria-selected="selected === index"
        tabindex="-1"
        @mousemove="selected = index"
        @click="go(item)"
      >
        <i :class="['pi', item.icon]" aria-hidden="true" /><span
          ><strong>{{ item.label }}</strong
          ><small>{{ item.description }}</small></span
        ><i class="pi pi-arrow-up-right navigator-arrow" aria-hidden="true" />
      </button>
    </div>
    <p v-if="!matches.length" class="navigator-empty" role="status">
      No destinations match “{{ query }}”.
    </p>
    <div class="navigator-help">
      <span><kbd>↑</kbd><kbd>↓</kbd> to navigate</span><span><kbd>Enter</kbd> to open</span
      ><span><kbd>Esc</kbd> to close</span>
    </div>
  </Dialog>
</template>
<script setup>
import { nextTick, onMounted, onUnmounted, ref, computed, watch } from 'vue'
import { useRouter } from 'vue-router'
import Dialog from 'primevue/dialog'
const router = useRouter()
const open = ref(false),
  query = ref(''),
  selected = ref(0),
  searchInput = ref(null)
const shortcut =
  typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform) ? '⌘ K' : 'Ctrl K'
const destinations = [
  {
    label: 'Models',
    description: 'Library, configuration, and model lifecycle',
    icon: 'pi-database',
    to: '/models',
    keywords: 'start stop warmup inference',
  },
  {
    label: 'Proxy',
    description: 'Live traffic, routing, requests, and keys',
    icon: 'pi-share-alt',
    to: '/proxy',
    keywords: 'gateway logs tokens errors',
  },
  {
    label: 'Engines',
    description: 'Install and manage inference engines',
    icon: 'pi-cog',
    to: '/engines',
    keywords: 'llama vllm sglang versions',
  },
  {
    label: 'Audio',
    description: 'Speech, transcription, and audio workspace',
    icon: 'pi-volume-up',
    to: '/audio',
    keywords: 'tts asr voice',
  },
  {
    label: 'Search models',
    description: 'Discover and download models',
    icon: 'pi-search',
    to: '/search',
    keywords: 'huggingface repository',
  },
  {
    label: 'System diagnostics',
    description: 'Service health and diagnostic bundle',
    icon: 'pi-sliders-h',
    to: '/system',
    keywords: 'debug persistence',
  },
  {
    label: 'Backup & restore',
    description: 'Export or restore saved configuration',
    icon: 'pi-download',
    to: '/system#config-backup',
    keywords: 'import settings',
  },
  {
    label: 'Change history',
    description: 'Inspect and restore configuration revisions',
    icon: 'pi-history',
    to: '/system#history',
    keywords: 'undo revisions',
  },
]
const matches = computed(() =>
  destinations.filter((item) =>
    `${item.label} ${item.description} ${item.keywords}`
      .toLowerCase()
      .includes(query.value.trim().toLowerCase()),
  ),
)
watch(query, () => {
  selected.value = 0
})
watch(open, (value) => {
  if (value) {
    query.value = ''
    selected.value = 0
  }
})
async function focusSearch() {
  await nextTick()
  searchInput.value?.focus()
}
async function go(item) {
  if (!item) return
  open.value = false
  await router.push(item.to)
  await nextTick()
  document.getElementById('main-content')?.focus({ preventScroll: true })
}
function onKey(event) {
  if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
    event.preventDefault()
    selected.value = matches.value.length
      ? (selected.value + (event.key === 'ArrowDown' ? 1 : -1) + matches.value.length) %
        matches.value.length
      : 0
    document
      .getElementById(`workspace-destination-${selected.value}`)
      ?.scrollIntoView?.({ block: 'nearest' })
  }
  if (event.key === 'Enter') {
    event.preventDefault()
    go(matches.value[selected.value])
  }
}
function onShortcut(event) {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault()
    open.value = !open.value
  }
}
onMounted(() => window.addEventListener('keydown', onShortcut))
onUnmounted(() => window.removeEventListener('keydown', onShortcut))
</script>
<style scoped>
.workspace-search {
  display: inline-flex;
  align-items: center;
  gap: 0.65rem;
  min-height: 2rem;
  padding: 0.4rem 0.65rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-secondary);
  font-size: 0.75rem;
  cursor: pointer;
}
.workspace-search:hover {
  background: var(--hover-bg);
  color: var(--text-primary);
}
kbd {
  padding: 0.05rem 0.25rem;
  border: 1px solid var(--border-primary);
  border-radius: 0.2rem;
  font: inherit;
  font-size: 0.68rem;
}
.navigator-input {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem;
  border: 1px solid var(--border-secondary);
  border-radius: var(--radius-md);
  color: var(--text-secondary);
}
.navigator-input:focus-within {
  border-color: var(--accent-primary);
}
.navigator-input input {
  width: 100%;
  min-width: 0;
  border: 0;
  background: transparent;
  color: var(--text-primary);
  outline: none;
}
.navigator-results {
  margin-top: 0.5rem;
  max-height: min(50dvh, 26rem);
  overflow: auto;
}
.navigator-results button {
  display: flex;
  align-items: center;
  gap: 0.85rem;
  width: 100%;
  padding: 0.75rem;
  border: 1px solid transparent;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-secondary);
  text-align: left;
  cursor: pointer;
}
.navigator-results button[aria-selected='true'] {
  background: var(--nav-active-bg);
  border-color: var(--border-primary);
  color: var(--accent-primary);
}
.navigator-results button > span {
  display: grid;
  gap: 0.15rem;
}
.navigator-results strong {
  color: var(--text-primary);
  font-size: 0.85rem;
  font-weight: 550;
}
.navigator-results small {
  font-size: 0.73rem;
  color: var(--text-secondary);
}
.navigator-arrow {
  margin-left: auto;
  font-size: 0.75rem;
}
.navigator-help {
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 0.5rem;
  padding-top: 0.75rem;
  margin-top: 0.5rem;
  border-top: 1px solid var(--border-primary);
  color: var(--text-secondary);
  font-size: 0.68rem;
}
.navigator-help kbd {
  margin-right: 0.2rem;
}
.navigator-empty {
  padding: 1.5rem 0.5rem;
  text-align: center;
  color: var(--text-secondary);
}

@media (max-width: 600px) {
  .workspace-search > span,
  .workspace-search > kbd {
    display: none;
  }
  .workspace-search {
    min-width: 2.25rem;
    min-height: 2.25rem;
    justify-content: center;
  }
}
</style>
