<template>
  <StateBanner v-if="visible" class="engine-build-settings-hint" severity="info">
    <span class="engine-build-settings-hint__title">Set build settings first</span>
    <p class="engine-build-settings-hint__text">{{ description }}</p>
    <template #actions>
      <Button
        label="Open build settings"
        icon="pi pi-external-link"
        size="small"
        severity="info"
        outlined
        @click="onOpenSettings"
      />
      <Button label="Got it" text size="small" severity="secondary" @click="dismiss" />
    </template>
  </StateBanner>
</template>

<script setup>
import { computed, ref, onMounted } from 'vue'
import Button from 'primevue/button'
import StateBanner from '@/components/common/StateBanner.vue'
import { engineBuildHint } from '../../engines/registry.js'

const props = defineProps({
  /** localStorage slot / engine id */
  engineKey: {
    type: String,
    required: true,
  },
  /** Override the default per-engine description. */
  description: {
    type: String,
    default: '',
  },
})

const emit = defineEmits(['open-settings'])

const LS_KEY = 'lcs.engine.buildSettingsHintDismissed.v1'

const visible = ref(true)

const description = computed(
  () =>
    props.description ||
    engineBuildHint(props.engineKey) ||
    'Configure build or install defaults before you proceed. Reopen Build settings from the dialog header anytime.',
)

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
  margin-bottom: 0;
}

.engine-build-settings-hint__title {
  display: block;
  margin-bottom: 0.15rem;
  color: var(--text-primary);
  font-size: var(--text-sm);
  font-weight: 650;
}

.engine-build-settings-hint__text {
  margin: 0;
  color: var(--text-secondary);
  font-size: var(--text-sm);
  line-height: 1.45;
}
</style>
