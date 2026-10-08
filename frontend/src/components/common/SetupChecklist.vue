<template>
  <section v-if="visible" class="setup-checklist" aria-labelledby="setup-checklist-title">
    <div class="setup-checklist__summary">
      <div class="setup-checklist__copy">
        <h2 id="setup-checklist-title">{{ currentStep?.label }}</h2>
        <p class="setup-checklist__detail">{{ currentStep?.detail }}</p>
      </div>
      <div class="setup-checklist__actions">
        <button
          v-if="currentStep?.to"
          type="button"
          class="setup-checklist__link"
          @click="router.push(currentStep.to)"
        >
          {{ currentStep.action }}
        </button>
        <button
          type="button"
          class="setup-checklist__expand"
          :aria-expanded="expanded"
          aria-controls="setup-steps"
          @click="expanded = !expanded"
        >
          {{ expanded ? 'Hide steps' : 'Show all steps' }}
        </button>
        <button type="button" class="setup-checklist__dismiss" @click="dismiss">Dismiss</button>
      </div>
    </div>
    <ol v-if="expanded" id="setup-steps" class="setup-checklist__steps">
      <li
        v-for="step in steps"
        :key="step.id"
        :class="{ 'is-done': step.done, 'is-current': step.current }"
      >
        <span class="setup-checklist__mark" aria-hidden="true">{{
          step.done ? '✓' : step.current ? '→' : '○'
        }}</span>
        <div>
          <div class="setup-checklist__label">{{ step.label }}</div>
          <p class="setup-checklist__detail">{{ step.detail }}</p>
        </div>
      </li>
    </ol>
  </section>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useModelStore } from '@/stores/models'
import { useEnginesStore } from '@/stores/engines'
import { runnableEngineIds } from '@/composables/engineReadiness'

const router = useRouter()

const DISMISS_KEY = 'llama-studio.setup-checklist.dismissed'

const modelStore = useModelStore()
const enginesStore = useEnginesStore()
const dismissed = ref(
  typeof localStorage !== 'undefined' && localStorage.getItem(DISMISS_KEY) === '1',
)
const expanded = ref(false)
const descriptorsReady = ref(false)

const quants = computed(() =>
  (modelStore.models || []).flatMap((group) => group.quantizations || []),
)

const engineReady = computed(() => runnableEngineIds(enginesStore.engineDescriptors).length > 0)

const hasModels = computed(() => quants.value.length > 0)

const configReviewed = computed(() => quants.value.some((quant) => quant?.config_reviewed))

const hasVerifiedRunning = computed(() =>
  quants.value.some(
    (quant) =>
      quant?.is_active &&
      quant?.runtime_quality !== 'unreachable' &&
      quant?.runtime_quality !== 'stale',
  ),
)

const proxyKnown = computed(() => enginesStore.systemStatus?.proxy_status != null)
const proxyHealthy = computed(() => Boolean(enginesStore.systemStatus?.proxy_status?.healthy))

const runnableNames = computed(() =>
  (enginesStore.engineDescriptors || [])
    .filter((descriptor) => descriptor?.runnable)
    .map((descriptor) => descriptor.label || descriptor.id)
    .join(', '),
)

const steps = computed(() => {
  const items = [
    {
      id: 'engine',
      label: 'Set up an engine',
      detail: engineReady.value
        ? `${runnableNames.value} can launch a model.`
        : 'Install and activate an engine before starting a model.',
      done: engineReady.value,
      to: '/engines',
      action: engineReady.value ? 'Manage engines' : 'Install an engine',
    },
    {
      id: 'download',
      label: 'Download a model',
      detail: 'Search for a model compatible with your engine.',
      done: hasModels.value,
      to: '/search',
      action: 'Search and download',
    },
    {
      id: 'config',
      label: 'Review configuration',
      detail: 'Open the model, check the engine and context, then save.',
      done: configReviewed.value,
      to: null,
      action: '',
    },
    {
      id: 'start',
      label: 'Start the model',
      detail: 'Start a configured model from the library.',
      done: hasVerifiedRunning.value,
      to: null,
      action: '',
    },
    {
      id: 'connect',
      label: 'Connect a client',
      detail:
        proxyKnown.value && !proxyHealthy.value
          ? 'llama-swap is offline, so clients cannot reach a running model yet.'
          : 'Open Connect on a running model to copy its endpoint and a sample request.',
      done: hasVerifiedRunning.value && proxyHealthy.value,
      to: null,
      action: '',
    },
  ]
  const currentId = items.find((step) => !step.done)?.id
  return items.map((step) => ({ ...step, current: step.id === currentId }))
})

const currentStep = computed(() => steps.value.find((step) => step.current) || null)
const verifiedSuccess = computed(
  () =>
    descriptorsReady.value && engineReady.value && hasVerifiedRunning.value && proxyHealthy.value,
)

const visible = computed(
  () =>
    descriptorsReady.value &&
    !dismissed.value &&
    !verifiedSuccess.value &&
    steps.value.some((step) => !step.done),
)

onMounted(async () => {
  try {
    await enginesStore.fetchEngineDescriptors()
  } catch {
    /* The engine step stays incomplete when descriptors cannot be loaded. */
  } finally {
    descriptorsReady.value = true
  }
})

function dismiss() {
  dismissed.value = true
  try {
    localStorage.setItem(DISMISS_KEY, '1')
  } catch {
    /* ignore private-mode storage failures */
  }
}
</script>

<style scoped>
.setup-checklist {
  padding: 1rem 1.25rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  background: var(--bg-card);
}
.setup-checklist__summary {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.75rem 1.5rem;
}
.setup-checklist__copy {
  flex: 1 1 18rem;
}
.setup-checklist h2 {
  margin: 0;
  font-size: 0.875rem;
  font-weight: 600;
}
.setup-checklist__detail {
  margin: 0.25rem 0 0;
  font-size: 0.8125rem;
  line-height: 1.6;
}
.setup-checklist__actions {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.25rem;
}
.setup-checklist__actions button {
  min-height: 2.75rem;
  padding: 0.5rem 0.625rem;
  border: 0;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-secondary);
  font: inherit;
  font-size: 0.8125rem;
  cursor: pointer;
}
.setup-checklist__actions .setup-checklist__link {
  color: var(--accent-primary);
  font-weight: 600;
}
.setup-checklist__actions button:hover {
  background: var(--hover-bg);
}
.setup-checklist__steps {
  list-style: none;
  display: grid;
  gap: 1rem;
  margin: 1rem 0 0;
  padding: 1rem 0 0;
  border-top: 1px solid var(--border-primary);
}
.setup-checklist__steps li {
  display: flex;
  gap: 0.75rem;
  color: var(--text-secondary);
}
.setup-checklist__steps li.is-current {
  color: var(--text-primary);
}
.setup-checklist__mark {
  width: 1.25rem;
  flex: none;
  color: var(--accent-primary);
}
.setup-checklist__label {
  font-size: 0.875rem;
  font-weight: 600;
}
</style>
