<template>
  <div class="log-viewer task-logs" :style="height ? { height } : undefined">
    <div v-if="showToolbar" class="log-toolbar" role="toolbar" aria-label="Log tools">
      <label class="log-search">
        <i class="pi pi-search" aria-hidden="true" />
        <input
          v-model="query"
          type="search"
          :placeholder="searchPlaceholder"
          aria-label="Search log"
          :disabled="!lines.length"
          autocomplete="off"
          spellcheck="false"
        />
        <button
          v-if="query"
          type="button"
          class="log-search__clear"
          aria-label="Clear log search"
          @click="query = ''"
        >
          <i class="pi pi-times" aria-hidden="true" />
        </button>
      </label>

      <div v-if="levels.length" class="log-levels" role="group" aria-label="Log severity">
        <button
          v-for="option in levels"
          :key="option.value"
          type="button"
          class="log-level"
          :class="{ 'log-level--active': level === option.value }"
          :aria-pressed="level === option.value ? 'true' : 'false'"
          @click="setLevel(option.value)"
        >
          {{ option.label }}
        </button>
      </div>

      <span v-if="countLabel" class="log-count" aria-live="polite">{{ countLabel }}</span>
      <span v-if="live" class="log-live" title="Streaming live">
        <span class="log-live__dot" aria-hidden="true" />Live
      </span>

      <button
        type="button"
        class="log-action"
        :aria-pressed="following ? 'true' : 'false'"
        :aria-label="following ? 'Follow log output' : 'Resume following log output'"
        :title="following ? 'Following new output' : 'Follow new output'"
        @click="resumeFollow"
      >
        <i :class="following ? 'pi pi-angle-double-down' : 'pi pi-angle-down'" aria-hidden="true" />
        <span class="log-action__label">{{ following ? 'Following' : 'Follow' }}</span>
      </button>

      <button
        type="button"
        class="log-action"
        :aria-label="copyLabel"
        :title="copyLabel"
        :disabled="!filtered.length"
        @click="copyLogs"
      >
        <i :class="copyIcon" aria-hidden="true" />
        <span class="log-action__label">{{ copyLabel }}</span>
      </button>

      <button
        v-if="downloadable"
        type="button"
        class="log-action log-action--icon"
        aria-label="Download log"
        title="Download log"
        :disabled="!filtered.length"
        @click="downloadLogs"
      >
        <i class="pi pi-download" aria-hidden="true" />
      </button>
      <button
        v-if="clearable"
        type="button"
        class="log-action log-action--icon"
        aria-label="Clear log"
        title="Clear captured log"
        @click="$emit('clear')"
      >
        <i class="pi pi-trash" aria-hidden="true" />
      </button>
      <slot name="actions" />
    </div>

    <div ref="scroller" class="log-body" tabindex="0" :aria-label="ariaLabel" @scroll="onScroll">
      <p v-if="!lines.length" class="log-empty">
        {{ loading ? 'Loading saved log…' : emptyMessage }}
      </p>
      <p v-else-if="!filtered.length" class="log-empty">
        No log lines match “{{ query }}”{{ level ? ' at this severity' : '' }}.
      </p>
      <template v-else>
        <LogLine
          v-for="entry in filtered"
          :key="entry.key"
          :line="entry.text"
          :stream="entry.stream"
          :query="query"
        />
      </template>
    </div>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { copyText } from '@/utils/clipboard'
import { stripAnsi } from '@/utils/ansi'
import LogLine from './LogLine.vue'

const props = defineProps({
  lines: { type: Array, default: () => [] },
  emptyMessage: { type: String, default: 'Waiting for output…' },
  loading: { type: Boolean, default: false },
  levels: { type: Array, default: () => [] },
  streamAware: { type: Boolean, default: false },
  downloadable: { type: Boolean, default: false },
  clearable: { type: Boolean, default: false },
  live: { type: Boolean, default: false },
  showToolbar: { type: Boolean, default: true },
  searchPlaceholder: { type: String, default: 'Search log…' },
  height: { type: String, default: '' },
  ariaLabel: { type: String, default: 'Engine log output' },
})

const emit = defineEmits(['download', 'clear'])

const scroller = ref(null)
const query = ref('')
const level = ref('')
const following = ref(true)
const copied = ref('')
let copyTimer = 0
let lastScrollTop = 0

const STREAM_PREFIX = /^(stdout|stderr):\s?/
const normalizeCache = new Map()

function normalizeLine(raw) {
  const cacheKey = typeof raw === 'string' ? raw : JSON.stringify(raw)
  const cached = normalizeCache.get(cacheKey)
  if (cached) return cached
  let text = typeof raw === 'string' ? raw : String(raw?.line ?? raw?.text ?? '')
  let stream = ''
  if (props.streamAware) {
    const match = STREAM_PREFIX.exec(text)
    if (match) {
      stream = match[1]
      text = text.slice(match[0].length)
    }
  }
  // Carriage returns render in place: a terminal keeps only the last write.
  if (text.includes('\r')) text = text.slice(text.lastIndexOf('\r') + 1)
  const normalized = { text, stream }
  if (normalizeCache.size > 40_000) normalizeCache.clear()
  normalizeCache.set(cacheKey, normalized)
  return normalized
}

const entries = computed(() =>
  props.lines.map((raw, index) => {
    const { text, stream } = normalizeLine(raw)
    return { key: index, index, text, stream }
  }),
)

const plainCache = new Map()
function plainText(text) {
  let plain = plainCache.get(text)
  if (plain === undefined) {
    plain = stripAnsi(text)
    if (plainCache.size > 40_000) plainCache.clear()
    plainCache.set(text, plain)
  }
  return plain
}

const filtered = computed(() => {
  const needle = query.value.trim().toLowerCase()
  const severity = level.value
  if (!needle && !severity) return entries.value
  return entries.value.filter((entry) => {
    const plain = plainText(entry.text).toLowerCase()
    if (needle && !plain.includes(needle)) return false
    if (!severity) return true
    const error = entry.stream === 'stderr' || /error|fatal|traceback|exception|failed/.test(plain)
    if (severity === 'error') return error
    if (severity === 'warn') return error || /warn|deprecat/.test(plain)
    return true
  })
})

const countLabel = computed(() => {
  if (!props.lines.length) return ''
  if (!query.value.trim() && !level.value) return ''
  return `${filtered.value.length} / ${entries.value.length}`
})

const copyLabel = computed(() => {
  if (copied.value === 'ok') return 'Copied'
  if (copied.value === 'failed') return 'Copy failed'
  return 'Copy'
})
const copyIcon = computed(() => {
  if (copied.value === 'ok') return 'pi pi-check'
  if (copied.value === 'failed') return 'pi pi-exclamation-triangle'
  return 'pi pi-copy'
})

function setLevel(value) {
  level.value = value
}

function displayLines() {
  return filtered.value.map((entry) => plainText(entry.text))
}

function scrollToBottom() {
  const element = scroller.value
  if (!element) return
  element.scrollTop = element.scrollHeight
  lastScrollTop = element.scrollTop
}

function scheduleFollow() {
  nextTick(() => {
    scrollToBottom()
    // Content-visibility rows settle after their first paint; re-pin once so
    // wrapped lines still end exactly at the newest output.
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(scrollToBottom)
  })
}

function resumeFollow() {
  following.value = true
  scrollToBottom()
}

function onScroll() {
  const element = scroller.value
  if (!element) return
  const top = element.scrollTop
  const movedUp = top < lastScrollTop - 4
  lastScrollTop = top
  const distance = element.scrollHeight - top - element.clientHeight
  if (following.value) {
    // Content growth moves scrollHeight, not scrollTop. Only an actual upward
    // scroll releases follow; otherwise re-pin to the newest output.
    if (movedUp) following.value = false
    else if (distance > 32) scrollToBottom()
  } else if (distance < 32) {
    following.value = true
  }
}

async function copyLogs() {
  const text = displayLines().join('\n')
  if (!text) return
  try {
    await copyText(text)
    copied.value = 'ok'
  } catch {
    copied.value = 'failed'
  }
  window.clearTimeout(copyTimer)
  copyTimer = window.setTimeout(() => {
    copied.value = ''
  }, 1600)
}

function downloadLogs() {
  const lines = displayLines()
  if (lines.length) emit('download', lines)
}

watch(
  () => props.lines,
  () => {
    if (!props.lines.length) following.value = true
    if (following.value) scheduleFollow()
  },
  { deep: false },
)
watch(
  () => props.lines.length,
  () => {
    if (following.value) scheduleFollow()
  },
)
watch(filtered, () => {
  if (following.value) scheduleFollow()
})
watch([query, level], () => {
  following.value = false
  nextTick(() => {
    if (scroller.value) scroller.value.scrollTop = 0
    lastScrollTop = 0
  })
})

onMounted(() => {
  scheduleFollow()
})

onBeforeUnmount(() => {
  window.clearTimeout(copyTimer)
})
</script>

<style scoped>
.log-viewer {
  /* Terminal palette (dark defaults). Scoped custom properties inherit into
     every rendered LogLine, so ANSI colours stay consistent everywhere. */
  --ansi-fg: #d7dde8;
  --ansi-bg: #0d1117;
  --ansi-fg-black: #6b7280;
  --ansi-fg-red: #f87171;
  --ansi-fg-green: #4ade80;
  --ansi-fg-yellow: #facc15;
  --ansi-fg-blue: #60a5fa;
  --ansi-fg-magenta: #e879f9;
  --ansi-fg-cyan: #22d3ee;
  --ansi-fg-white: #e5e7eb;
  --ansi-fg-bright-black: #9ca3af;
  --ansi-fg-bright-red: #fca5a5;
  --ansi-fg-bright-green: #86efac;
  --ansi-fg-bright-yellow: #fde047;
  --ansi-fg-bright-blue: #93c5fd;
  --ansi-fg-bright-magenta: #f0abfc;
  --ansi-fg-bright-cyan: #67e8f9;
  --ansi-fg-bright-white: #f8fafc;
  --ansi-bg-black: #1f2937;
  --ansi-bg-red: #7f1d1d;
  --ansi-bg-green: #14532d;
  --ansi-bg-yellow: #713f12;
  --ansi-bg-blue: #1e3a8a;
  --ansi-bg-magenta: #701a75;
  --ansi-bg-cyan: #164e63;
  --ansi-bg-white: #374151;
  --ansi-bg-bright-black: #374151;
  --ansi-bg-bright-red: #991b1b;
  --ansi-bg-bright-green: #166534;
  --ansi-bg-bright-yellow: #a16207;
  --ansi-bg-bright-blue: #1d4ed8;
  --ansi-bg-bright-magenta: #a21caf;
  --ansi-bg-bright-cyan: #0e7490;
  --ansi-bg-bright-white: #4b5563;

  display: flex;
  flex-direction: column;
  height: 22rem;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
  border: 1px solid var(--border-primary, #2a2f45);
  border-radius: var(--radius-md, 0.5rem);
  background: color-mix(in srgb, var(--bg-primary, #111214) 94%, black);
}

:global([data-theme='light']) .log-viewer {
  --ansi-fg: #1f2937;
  --ansi-bg: #f8fafc;
  --ansi-fg-black: #111827;
  --ansi-fg-red: #b91c1c;
  --ansi-fg-green: #15803d;
  --ansi-fg-yellow: #a16207;
  --ansi-fg-blue: #1d4ed8;
  --ansi-fg-magenta: #a21caf;
  --ansi-fg-cyan: #0e7490;
  --ansi-fg-white: #475569;
  --ansi-fg-bright-black: #4b5563;
  --ansi-fg-bright-red: #dc2626;
  --ansi-fg-bright-green: #16a34a;
  --ansi-fg-bright-yellow: #ca8a04;
  --ansi-fg-bright-blue: #2563eb;
  --ansi-fg-bright-magenta: #c026d3;
  --ansi-fg-bright-cyan: #0891b2;
  --ansi-fg-bright-white: #1e293b;
  --ansi-bg-black: #e5e7eb;
  --ansi-bg-red: #fecaca;
  --ansi-bg-green: #bbf7d0;
  --ansi-bg-yellow: #fef08a;
  --ansi-bg-blue: #bfdbfe;
  --ansi-bg-magenta: #f5d0fe;
  --ansi-bg-cyan: #a5f3fc;
  --ansi-bg-white: #e2e8f0;
  --ansi-bg-bright-black: #cbd5e1;
  --ansi-bg-bright-red: #fca5a5;
  --ansi-bg-bright-green: #86efac;
  --ansi-bg-bright-yellow: #fde047;
  --ansi-bg-bright-blue: #93c5fd;
  --ansi-bg-bright-magenta: #f0abfc;
  --ansi-bg-bright-cyan: #67e8f9;
  --ansi-bg-bright-white: #cbd5e1;
}

.log-toolbar {
  display: flex;
  flex: 0 0 auto;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.3rem;
  row-gap: 0.25rem;
  min-width: 0;
  padding: 0.28rem 0.35rem;
  border-bottom: 1px solid var(--border-primary, #2a2f45);
  background: color-mix(in srgb, var(--bg-secondary, #17181b) 88%, transparent);
}

.log-search {
  display: flex;
  flex: 1 1 7rem;
  align-items: center;
  gap: 0.25rem;
  min-width: 4.5rem;
  padding: 0 0.4rem;
  border: 1px solid var(--border-primary, #2a2f45);
  border-radius: 999px;
  color: var(--text-secondary);
}

.log-search:focus-within {
  border-color: var(--accent-primary, #a5b4fc);
  box-shadow: 0 0 0 2px var(--focus-ring, rgb(129 140 248 / 30%));
}

.log-search i {
  font-size: 0.65rem;
  flex: 0 0 auto;
}

.log-search input {
  width: 100%;
  min-width: 0;
  height: 1.5rem;
  padding: 0;
  border: 0;
  outline: 0;
  background: transparent;
  color: var(--text-primary);
  font: inherit;
  font-size: 0.72rem;
}

.log-search input::placeholder {
  color: var(--text-secondary);
  opacity: 0.8;
}

.log-search input::-webkit-search-cancel-button {
  display: none;
}

.log-search__clear {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0;
  border: 0;
  background: transparent;
  color: var(--text-secondary);
  cursor: pointer;
}

.log-search__clear .pi {
  font-size: 0.65rem;
}

.log-levels {
  display: inline-flex;
  flex: 0 0 auto;
  overflow: hidden;
  border: 1px solid var(--border-primary, #2a2f45);
  border-radius: 999px;
}

.log-level {
  height: 1.55rem;
  padding: 0 0.4rem;
  border: 0;
  border-right: 1px solid var(--border-primary, #2a2f45);
  background: transparent;
  color: var(--text-secondary);
  font: inherit;
  font-size: 0.68rem;
  font-weight: 600;
  cursor: pointer;
}

.log-level:last-child {
  border-right: 0;
}

.log-level:hover {
  color: var(--text-primary);
  background: var(--bg-card-hover, rgb(255 255 255 / 6%));
}

.log-level--active {
  color: var(--nav-active-fg, #f8fafc);
  background: var(--nav-active-bg, rgb(129 140 248 / 20%));
}

.log-count,
.log-live {
  display: inline-flex;
  flex: 0 0 auto;
  align-items: center;
  gap: 0.25rem;
  color: var(--text-secondary);
  font-size: 0.66rem;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.log-live__dot {
  width: 0.4rem;
  height: 0.4rem;
  border-radius: 50%;
  background: var(--status-success, #10b981);
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--status-success, #10b981) 22%, transparent);
}

.log-action {
  display: inline-flex;
  flex: 0 0 auto;
  align-items: center;
  gap: 0.22rem;
  height: 1.55rem;
  padding: 0 0.42rem;
  border: 1px solid var(--border-primary, #2a2f45);
  border-radius: 999px;
  background: transparent;
  color: var(--text-secondary);
  font: inherit;
  font-size: 0.68rem;
  font-weight: 650;
  cursor: pointer;
}

.log-action .pi {
  font-size: 0.68rem;
}

.log-action:hover {
  color: var(--text-primary);
  border-color: var(--border-secondary, #3b4261);
  background: var(--bg-card-hover, rgb(255 255 255 / 6%));
}

.log-action:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}

.log-action:disabled:hover {
  color: var(--text-secondary);
  border-color: var(--border-primary, #2a2f45);
  background: transparent;
}

.log-action[aria-pressed='true'] {
  color: var(--text-primary);
  border-color: var(--nav-active-bg, #0e7490);
  background: var(--nav-active-bg, rgb(129 140 248 / 14%));
}

.log-action--icon {
  width: 1.55rem;
  justify-content: center;
  padding: 0;
}

.log-body {
  flex: 1 1 auto;
  min-height: 0;
  overflow: auto;
  padding: 0.4rem 0.2rem 0.55rem;
  background: color-mix(in srgb, var(--ansi-bg) 96%, transparent);
  color: var(--ansi-fg);
  overscroll-behavior: contain;
  scrollbar-gutter: stable;
}

.log-body:focus-visible {
  outline: 2px solid var(--accent-primary, #a5b4fc);
  outline-offset: -2px;
}

.log-empty {
  margin: 0.4rem 0.6rem;
  color: var(--ansi-fg);
  opacity: 0.7;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 0.72rem;
}

@media (max-width: 640px) {
  .log-viewer {
    height: 17rem;
  }

  .log-action__label {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }

  .log-action {
    width: 1.55rem;
    justify-content: center;
    padding: 0;
  }

  .log-live {
    display: none;
  }
}
</style>
