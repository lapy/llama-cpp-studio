<template>
  <div
    class="log-line"
    :class="[
      stream === 'stderr' ? 'log-line--stderr' : 'log-line--stdout',
      { 'log-line--wrap': wrap },
    ]"
    :data-stream="stream || undefined"
  >
    <span v-if="!segments.length" aria-hidden="true">&nbsp;</span>
    <span
      v-for="(segment, index) in segments"
      :key="index"
      :class="segment.className"
      :style="segment.style"
      >{{ segment.text }}</span
    >
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { renderAnsi } from '@/utils/ansi'

const props = defineProps({
  line: { type: String, default: '' },
  stream: { type: String, default: '' },
  query: { type: String, default: '' },
  wrap: { type: Boolean, default: false },
})

const segments = computed(() => renderAnsi(props.line, props.query))
</script>

<style scoped>
.log-line {
  width: max-content;
  min-width: 100%;
  min-height: 1.45em;
  padding: 0 0.2rem;
  color: var(--ansi-fg);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace;
  font-size: 0.75rem;
  font-variant-ligatures: none;
  line-height: 1.45;
  white-space: pre;
  overflow-wrap: normal;
}

.log-line--wrap {
  width: auto;
  min-width: 0;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.log-line--stderr {
  padding-left: 0.45rem;
  border-left: 2px solid color-mix(in srgb, var(--ansi-fg) 16%, transparent);
  background: transparent;
  color: var(--ansi-fg);
}

.ansi-bold {
  font-weight: 700;
}

.ansi-dim {
  opacity: 0.68;
}

.ansi-italic {
  font-style: italic;
}

.ansi-underline {
  text-decoration: underline;
  text-underline-offset: 2px;
}

.ansi-reverse {
  background: var(--ansi-fg);
  color: var(--ansi-bg);
}

.ansi-fg-black {
  color: var(--ansi-fg-black);
}
.ansi-fg-red {
  color: var(--ansi-fg-red);
}
.ansi-fg-green {
  color: var(--ansi-fg-green);
}
.ansi-fg-yellow {
  color: var(--ansi-fg-yellow);
}
.ansi-fg-blue {
  color: var(--ansi-fg-blue);
}
.ansi-fg-magenta {
  color: var(--ansi-fg-magenta);
}
.ansi-fg-cyan {
  color: var(--ansi-fg-cyan);
}
.ansi-fg-white {
  color: var(--ansi-fg-white);
}
.ansi-fg-bright-black {
  color: var(--ansi-fg-bright-black);
}
.ansi-fg-bright-red {
  color: var(--ansi-fg-bright-red);
}
.ansi-fg-bright-green {
  color: var(--ansi-fg-bright-green);
}
.ansi-fg-bright-yellow {
  color: var(--ansi-fg-bright-yellow);
}
.ansi-fg-bright-blue {
  color: var(--ansi-fg-bright-blue);
}
.ansi-fg-bright-magenta {
  color: var(--ansi-fg-bright-magenta);
}
.ansi-fg-bright-cyan {
  color: var(--ansi-fg-bright-cyan);
}
.ansi-fg-bright-white {
  color: var(--ansi-fg-bright-white);
}

.ansi-bg-black {
  background-color: var(--ansi-bg-black);
}
.ansi-bg-red {
  background-color: var(--ansi-bg-red);
}
.ansi-bg-green {
  background-color: var(--ansi-bg-green);
}
.ansi-bg-yellow {
  background-color: var(--ansi-bg-yellow);
}
.ansi-bg-blue {
  background-color: var(--ansi-bg-blue);
}
.ansi-bg-magenta {
  background-color: var(--ansi-bg-magenta);
}
.ansi-bg-cyan {
  background-color: var(--ansi-bg-cyan);
}
.ansi-bg-white {
  background-color: var(--ansi-bg-white);
}
.ansi-bg-bright-black {
  background-color: var(--ansi-bg-bright-black);
}
.ansi-bg-bright-red {
  background-color: var(--ansi-bg-bright-red);
}
.ansi-bg-bright-green {
  background-color: var(--ansi-bg-bright-green);
}
.ansi-bg-bright-yellow {
  background-color: var(--ansi-bg-bright-yellow);
}
.ansi-bg-bright-blue {
  background-color: var(--ansi-bg-bright-blue);
}
.ansi-bg-bright-magenta {
  background-color: var(--ansi-bg-bright-magenta);
}
.ansi-bg-bright-cyan {
  background-color: var(--ansi-bg-bright-cyan);
}
.ansi-bg-bright-white {
  background-color: var(--ansi-bg-bright-white);
}

.log-match {
  border-radius: 2px;
  background: #fde68a;
  color: #1f2937;
  box-shadow: 0 0 0 1px rgb(180 83 9 / 45%);
}
</style>
