<template>
  <section
    :id="id || undefined"
    :class="sectionClass"
    :aria-labelledby="title ? headingId : undefined"
  >
    <header v-if="hasHeader" class="page-section__header">
      <div class="page-section__heading-group">
        <slot name="title">
          <component :is="headingTag" v-if="title" :id="headingId" class="page-section__title">{{
            title
          }}</component>
        </slot>
        <p v-if="description" class="page-section__description">{{ description }}</p>
        <slot name="description" />
      </div>
      <div v-if="$slots.meta" class="page-section__meta">
        <slot name="meta" />
      </div>
      <div v-if="$slots.actions" class="page-section__actions">
        <slot name="actions" />
      </div>
    </header>
    <div class="page-section__body">
      <slot />
    </div>
  </section>
</template>

<script setup>
import { computed, useId, useSlots } from 'vue'

const slots = useSlots()

const props = defineProps({
  title: {
    type: String,
    default: '',
  },
  description: {
    type: String,
    default: '',
  },
  /** Section heading level; page h1 stays with PageHeader. */
  headingTag: {
    type: String,
    default: 'h2',
    validator: (value) => /^h[1-6]$/.test(value),
  },
  /** Anchor target for deep links such as #storage. */
  id: {
    type: String,
    default: '',
  },
  variant: {
    type: String,
    default: 'default',
    validator: (value) => ['default', 'plain', 'compact'].includes(value),
  },
})

const autoId = useId()
const resolvedId = computed(() => props.id || `section-${autoId}`)
const headingId = computed(() => `${resolvedId.value}-title`)
const sectionClass = computed(() =>
  ['page-section', props.variant === 'default' ? '' : `page-section--${props.variant}`].filter(
    Boolean,
  ),
)
const hasHeader = computed(
  () =>
    Boolean(props.title || props.description) ||
    Boolean(slots.title || slots.description || slots.meta || slots.actions),
)
</script>
