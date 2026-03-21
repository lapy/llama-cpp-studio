<template>
  <div v-if="visible" class="engine-build-settings-hint" role="status">
    <div class="engine-build-settings-hint__row">
      <i class="pi pi-sliders-h engine-build-settings-hint__icon" aria-hidden="true" />
      <div class="engine-build-settings-hint__copy">
        <span class="engine-build-settings-hint__title">Set build settings first</span>
        <p class="engine-build-settings-hint__text">
          Choose CUDA, server binary, and other CMake options before you install or build. You can reopen
          <strong>Build settings</strong> from the dialog header anytime.
        </p>
      </div>
    </div>
    <div class="engine-build-settings-hint__actions">
      <Button
        label="Open build settings"
        icon="pi pi-external-link"
        size="small"
        severity="info"
        @click="onOpenSettings"
      />
      <Button label="Got it" text size="small" severity="secondary" @click="dismiss" />
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import Button from 'primevue/button'

const props = defineProps({
  /** localStorage slot: llama_cpp or ik_llama */
  engineKey: {
    type: String,
    required: true,
  },
})

const emit = defineEmits(['open-settings'])

const LS_KEY = 'lcs.engine.buildSettingsHintDismissed.v1'

const visible = ref(true)

function loadDismissed() {
  try {
    const raw = localStorage.getItem(LS_KEY)
    const o = raw ? JSON.parse(raw) : {}
    return Boolean(o[props.engineKey])
  } catch {
    return false
  }
}

function persistDismissed() {
  try {
    const raw = localStorage.getItem(LS_KEY)
    const o = raw ? JSON.parse(raw) : {}
    o[props.engineKey] = true
    localStorage.setItem(LS_KEY, JSON.stringify(o))
  } catch {
    /* ignore quota / private mode */
  }
}

function dismiss() {
  visible.value = false
  persistDismissed()
}

function onOpenSettings() {
  emit('open-settings')
  dismiss()
}

onMounted(() => {
  if (loadDismissed()) {
    visible.value = false
  }
})
</script>

<style scoped>
.engine-build-settings-hint {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  padding: 0.85rem 1rem;
  margin-bottom: 0.75rem;
  border-radius: var(--radius-lg);
  border: 1px solid rgba(96, 165, 250, 0.35);
  background: rgba(96, 165, 250, 0.07);
}

.engine-build-settings-hint__row {
  display: flex;
  gap: 0.65rem;
  align-items: flex-start;
  min-width: 0;
}

.engine-build-settings-hint__icon {
  font-size: 1.15rem;
  color: var(--accent-cyan, #60a5fa);
  flex-shrink: 0;
  margin-top: 0.1rem;
}

.engine-build-settings-hint__copy {
  min-width: 0;
}

.engine-build-settings-hint__title {
  display: block;
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 0.25rem;
}

.engine-build-settings-hint__text {
  margin: 0;
  font-size: 0.8125rem;
  line-height: 1.45;
  color: var(--text-secondary);
}

.engine-build-settings-hint__actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.35rem;
  padding-left: 0.1rem;
}

@media (min-width: 480px) {
  .engine-build-settings-hint__actions {
    padding-left: 1.85rem;
  }
}
</style>
