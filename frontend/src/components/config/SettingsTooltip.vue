<template>
  <span class="settings-tooltip-wrapper">
    <i 
      class="pi pi-info-circle tooltip-icon" 
      v-tooltip.top.right="{ value: tooltipContent, escape: false, fitContent: true }"
    ></i>
  </span>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  title: {
    type: String,
    required: true
  },
  description: {
    type: String,
    required: true
  },
  whenToAdjust: {
    type: String,
    default: null
  },
  tradeoffs: {
    type: Array,
    default: () => []
  },
  recommended: {
    type: String,
    default: null
  },
  ranges: {
    type: Array,
    default: () => []
  }
})

const tooltipContent = computed(() => {
  let content = `<div style="max-width: 380px; text-align: left; padding: 0;">`
  content += `<strong style="font-size: 1rem; font-weight: 600; color: var(--accent-cyan); display: block; margin-bottom: 0.5rem;">${props.title}</strong>`
  content += `<div style="margin-bottom: 0.75rem; line-height: 1.5;">`
  content += `<strong style="font-weight: 600;">What it does:</strong> ${props.description}`
  content += `</div>`
  
  if (props.whenToAdjust) {
    content += `<div style="margin-bottom: 0.75rem; line-height: 1.5;">`
    content += `<strong style="font-weight: 600;">When to adjust:</strong> ${props.whenToAdjust}`
    content += `</div>`
  }
  
  if (props.tradeoffs && props.tradeoffs.length > 0) {
    content += `<div style="margin-bottom: 0.75rem;">`
    content += `<strong style="font-weight: 600; display: block; margin-bottom: 0.25rem;">Trade-offs:</strong>`
    content += `<ul style="margin: 0.25rem 0; padding-left: 1.25rem; list-style: disc; line-height: 1.5;">`
    props.tradeoffs.forEach(tradeoff => {
      content += `<li style="margin: 0.125rem 0;">${tradeoff}</li>`
    })
    content += `</ul></div>`
  }
  
  if (props.ranges && props.ranges.length > 0) {
    content += `<div style="margin-bottom: 0.75rem;">`
    content += `<strong style="font-weight: 600; display: block; margin-bottom: 0.25rem;">Recommended ranges:</strong>`
    content += `<ul style="margin: 0.25rem 0; padding-left: 1.25rem; list-style: disc; line-height: 1.5;">`
    props.ranges.forEach(range => {
      content += `<li style="margin: 0.125rem 0;">${range}</li>`
    })
    content += `</ul></div>`
  }
  
  if (props.recommended) {
    content += `<div style="margin-top: 0.5rem; padding-top: 0.75rem; border-top: 1px solid var(--border-primary); line-height: 1.5;">`
    content += `<strong style="font-weight: 600; color: var(--accent-green);">For this model:</strong> ${props.recommended}`
    content += `</div>`
  }
  
  content += `</div>`
  return content
})
</script>

<style scoped>
.settings-tooltip-wrapper {
  display: inline-flex;
  align-items: center;
  margin-left: var(--spacing-xs);
}

.tooltip-icon {
  color: var(--accent-cyan);
  font-size: 0.9rem;
  cursor: help;
  transition: all var(--transition-normal);
}

.tooltip-icon:hover {
  color: var(--accent-primary);
  transform: scale(1.1);
}
</style>

<style>
/* Override tooltip positioning to prevent overflow */
.p-tooltip {
  max-width: min(380px, calc(100vw - 20px)) !important;
  word-wrap: break-word !important;
  overflow-wrap: break-word !important;
}

/* Ensure tooltips stay within viewport */
.p-tooltip-top {
  margin-top: 0.5rem !important;
}

.p-tooltip-right {
  margin-left: 0.5rem !important;
}

.p-tooltip-left {
  margin-right: 0.5rem !important;
}

.p-tooltip-bottom {
  margin-bottom: 0.5rem !important;
}
</style>

