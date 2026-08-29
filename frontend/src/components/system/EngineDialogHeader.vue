<template>
  <div class="engine-dialog-header">
    <div class="engine-dialog-header__main">
      <span v-if="$slots.leading" class="engine-dialog-header__leading">
        <slot name="leading" />
      </span>
      <span class="engine-dialog-header__title">{{ title }}</span>
      <span v-if="$slots.tags" class="engine-dialog-header__tags">
        <slot name="tags" />
      </span>
    </div>
    <div v-if="$slots.actions || $slots.more" class="engine-dialog-header__actions">
      <slot name="actions" />
      <div
        v-if="$slots.more"
        ref="moreRoot"
        class="engine-dialog-header__more"
        :class="{ 'is-open': moreOpen }"
      >
        <Button
          class="engine-dialog-header__more-toggle"
          icon="pi pi-ellipsis-v"
          text
          severity="secondary"
          size="small"
          aria-label="More actions"
          aria-haspopup="true"
          :aria-expanded="moreOpen"
          @click.stop="moreOpen = !moreOpen"
        />
        <div class="engine-dialog-header__more-items" @click="onMoreItemClick">
          <slot name="more" />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted, ref } from 'vue'
import Button from 'primevue/button'

defineProps({
  title: { type: String, required: true },
})

const moreOpen = ref(false)
const moreRoot = ref(null)

function onMoreItemClick(event) {
  if (event.target.closest('button')) {
    moreOpen.value = false
  }
}

function onDocPointerDown(event) {
  if (!moreOpen.value) return
  if (moreRoot.value && !moreRoot.value.contains(event.target)) {
    moreOpen.value = false
  }
}

function onKeydown(event) {
  if (event.key === 'Escape') moreOpen.value = false
}

function onViewportChange() {
  if (window.matchMedia('(min-width: 641px)').matches) {
    moreOpen.value = false
  }
}

onMounted(() => {
  document.addEventListener('pointerdown', onDocPointerDown)
  document.addEventListener('keydown', onKeydown)
  window.addEventListener('resize', onViewportChange)
})

onUnmounted(() => {
  document.removeEventListener('pointerdown', onDocPointerDown)
  document.removeEventListener('keydown', onKeydown)
  window.removeEventListener('resize', onViewportChange)
})
</script>

<style scoped>
/* Fills the header slot so actions sit just before PrimeVue’s maximize/close icons. */
.engine-dialog-header {
  display: flex;
  align-items: center;
  gap: 0.35rem 0.5rem;
  flex: 1;
  min-width: 0;
}

.engine-dialog-header__main {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  flex: 1;
  min-width: 0;
}

.engine-dialog-header__leading {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.engine-dialog-header__leading :deep(.pi) {
  font-size: 1.25rem;
  color: var(--accent-cyan);
}

.engine-dialog-header__title {
  font-size: 1.25rem;
  font-weight: 600;
  line-height: 1.3;
  margin: 0;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.engine-dialog-header__tags {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  min-width: 0;
  max-width: 12rem;
}

.engine-dialog-header__tags :deep(.engine-dialog-tag-clip) {
  min-width: 0;
  max-width: 100%;
}

.engine-dialog-header__actions {
  display: flex;
  align-items: center;
  gap: 0.15rem;
  flex-shrink: 0;
  /* Visually group with PrimeVue maximize/close (they use margin-right: 0.5rem between icons). */
  margin-right: 0.5rem;
}

.engine-dialog-header__actions :deep(.p-button),
.engine-dialog-header__actions :deep(.p-button .p-button-icon) {
  color: var(--text-secondary, rgba(255, 255, 255, 0.6));
}

.engine-dialog-header__actions :deep(.p-button:enabled:hover),
.engine-dialog-header__actions :deep(.p-button:enabled:hover .p-button-icon) {
  color: var(--text-primary, rgba(255, 255, 255, 0.87));
}

.engine-dialog-header__more {
  display: flex;
  align-items: center;
}

.engine-dialog-header__more-toggle {
  display: none !important;
}

.engine-dialog-header__more-items {
  display: flex;
  align-items: center;
  gap: 0.15rem;
}

/* Wide header: keep overflow actions as icon-only, same size as the cog. */
.engine-dialog-header__more-items :deep(.p-button) {
  width: 2rem;
  height: 2rem;
  min-width: 2rem;
  padding: 0;
  border: none;
  border-radius: 50%;
}

.engine-dialog-header__more-items :deep(.p-button-icon) {
  margin: 0;
  font-size: 1rem;
  line-height: 1;
}

.engine-dialog-header__more-items :deep(.p-button-label) {
  display: none;
}

@media (max-width: 768px) {
  .engine-dialog-header__title {
    font-size: 1.05rem;
  }

  .engine-dialog-header__tags {
    display: none;
  }
}

@media (max-width: 640px) {
  .engine-dialog-header__actions {
    margin-right: 0.15rem;
  }

  .engine-dialog-header__more {
    position: relative;
  }

  .engine-dialog-header__more-toggle {
    display: inline-flex !important;
  }

  .engine-dialog-header__more-items {
    display: none;
    position: absolute;
    top: calc(100% + 0.3rem);
    right: 0;
    z-index: 40;
    flex-direction: column;
    align-items: stretch;
    min-width: 12.5rem;
    padding: 0.3rem;
    gap: 0.1rem;
    border-radius: var(--radius-md);
    border: 1px solid var(--border-primary);
    background: var(--bg-card, var(--surface-0));
    box-shadow: var(--shadow-lg);
  }

  .engine-dialog-header__more.is-open .engine-dialog-header__more-items {
    display: flex;
  }

  .engine-dialog-header__more.is-open .engine-dialog-header__more-items :deep(.p-button) {
    width: 100%;
    min-width: 13.5rem;
    height: auto;
    justify-content: flex-start;
    gap: 0.5rem;
    padding: 0.5rem 0.65rem;
  }

  .engine-dialog-header__more.is-open .engine-dialog-header__more-items :deep(.p-button-label) {
    display: inline;
    font-size: 0.8125rem;
    font-weight: 500;
    line-height: 1.3;
    text-align: left;
    white-space: normal;
  }
}
</style>

<style>
@media (max-width: 640px) {
  .engine-dialog-header__more.is-open .engine-dialog-header__more-items .p-button {
    width: 100% !important;
    min-width: 13.5rem;
    height: auto !important;
  }
}
</style>
