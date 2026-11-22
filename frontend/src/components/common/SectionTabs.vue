<template>
  <nav v-if="isLinks" :class="tabsClass" :aria-label="ariaLabel">
    <RouterLink
      v-for="item in items"
      :key="item.id"
      :to="item.to"
      class="section-tabs__tab"
      :aria-current="item.id === modelValue ? 'page' : undefined"
    >
      <i v-if="item.icon" :class="['pi', item.icon]" aria-hidden="true" />
      <span>{{ item.label }}</span>
      <span v-if="item.count" class="section-tabs__count">{{ item.count }}</span>
    </RouterLink>
  </nav>
  <div v-else :class="tabsClass" role="tablist" :aria-label="ariaLabel">
    <button
      v-for="item in items"
      :id="tabId(item)"
      :key="item.id"
      type="button"
      role="tab"
      class="section-tabs__tab"
      :aria-selected="item.id === modelValue"
      :aria-controls="panelId(item)"
      :tabindex="item.id === modelValue ? 0 : -1"
      :disabled="item.disabled"
      @click="select(item)"
      @keydown="onRovingTabKeydown"
    >
      <i v-if="item.icon" :class="['pi', item.icon]" aria-hidden="true" />
      <span>{{ item.label }}</span>
      <span v-if="item.count" class="section-tabs__count">{{ item.count }}</span>
    </button>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { RouterLink } from 'vue-router'
import { onRovingTabKeydown } from '@/composables/useRovingTabs'

const props = defineProps({
  /**
   * [{ id, label, icon?, count?, disabled?, to? }]. An item with `to`
   * switches the strip to router-link navigation (System-style section nav);
   * otherwise it renders an ARIA tablist with roving focus and arrow keys.
   */
  items: {
    type: Array,
    required: true,
  },
  /** Selected tab id, or the current section id in links mode. */
  modelValue: {
    type: String,
    default: '',
  },
  ariaLabel: {
    type: String,
    required: true,
  },
  /**
   * When set, tabs get `${idPrefix}-tab-<id>` ids and point aria-controls at
   * `${idPrefix}-panel-<id>`. Leave unset for a tab strip without panels.
   */
  idPrefix: {
    type: String,
    default: '',
  },
  variant: {
    type: String,
    default: 'underline',
    validator: (value) => ['underline', 'pills'].includes(value),
  },
  /** Collapse to a two-column grid on narrow screens instead of scrolling. */
  responsiveGrid: {
    type: Boolean,
    default: false,
  },
})

const emit = defineEmits(['update:modelValue'])

const isLinks = computed(() => props.items.some((item) => item.to))
const tabsClass = computed(() => [
  'section-tabs',
  props.variant === 'pills' ? 'section-tabs--pills' : '',
  props.responsiveGrid ? 'section-tabs--responsive-grid' : '',
])

function tabId(item) {
  return props.idPrefix ? `${props.idPrefix}-tab-${item.id}` : undefined
}

function panelId(item) {
  return props.idPrefix ? `${props.idPrefix}-panel-${item.id}` : undefined
}

function select(item) {
  if (item.disabled || item.id === props.modelValue) return
  emit('update:modelValue', item.id)
}
</script>
