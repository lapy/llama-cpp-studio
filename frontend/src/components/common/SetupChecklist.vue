<template>
  <section v-if="visible" class="setup-checklist" aria-labelledby="setup-checklist-title">
    <div class="setup-checklist__head">
      <div>
        <span class="setup-checklist__eyebrow"
          >Getting started · Step {{ currentStepNumber }} of {{ steps.length }}</span
        >
        <h2 id="setup-checklist-title">Next: {{ currentStep?.label }}</h2>
      </div>
      <button type="button" class="setup-checklist__dismiss" @click="dismiss">Dismiss</button>
    </div>
    <div class="setup-checklist__progress" aria-hidden="true">
      <span :style="{ width: `${completionPercent}%` }" />
    </div>
    <p class="setup-checklist__detail">{{ currentStep?.detail }}</p>
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
        @click="expanded = !expanded"
      >
        {{ expanded ? 'Hide steps' : 'Show all steps' }}
      </button>
    </div>
    <ol v-if="expanded" class="setup-checklist__steps">
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
      label: 'Prepare a runnable engine',
      detail: engineReady.value
        ? `${runnableNames.value} can launch a model.`
        : 'Install an engine and activate a working version. A broken or inactive install is not ready. This includes Unsloth llama.cpp.',
      done: engineReady.value,
      to: '/engines',
      action: engineReady.value ? 'Manage engines' : 'Install an engine',
    },
    {
      id: 'download',
      label: 'Download a model',
      detail: 'Saved models appear in this library. Downloading does not review the configuration.',
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
      detail: 'Starting loads weights. A restart is required after some saved changes.',
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
          : 'Use Connect on the running model for a request that matches its capability.',
      done: hasVerifiedRunning.value && proxyHealthy.value,
      to: null,
      action: '',
    },
  ]
  const currentId = items.find((step) => !step.done)?.id
  return items.map((step) => ({ ...step, current: step.id === currentId }))
})

const currentStep = computed(() => steps.value.find((step) => step.current) || null)
const currentStepNumber = computed(() => {
  const index = steps.value.findIndex((step) => step.current)
  return index < 0 ? steps.value.length : index + 1
})
const completionPercent = computed(() => {
  const completed = steps.value.filter((step) => step.done).length
  return Math.round((completed / steps.value.length) * 100)
})

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
  position: relative;
  overflow: hidden;
  margin-bottom: 0.5rem;
  padding: 1rem 1.1rem;
  border: 1px solid color-mix(in srgb, var(--accent-cyan) 22%, var(--border-primary));
  border-radius: var(--radius-xl);
  background:
    linear-gradient(
      110deg,
      color-mix(in srgb, var(--accent-cyan) 8%, transparent),
      transparent 45%
    ),
    var(--bg-card);
  box-shadow: var(--shadow-sm);
}

.setup-checklist__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
}

.setup-checklist__head h2 {
  margin: 0.15rem 0 0;
  font-size: 1rem;
}

.setup-checklist__eyebrow {
  color: var(--accent-cyan);
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.07em;
  text-transform: uppercase;
}

.setup-checklist__progress {
  height: 0.22rem;
  margin: 0.8rem 0 0.65rem;
  overflow: hidden;
  border-radius: 999px;
  background: var(--border-primary);
}

.setup-checklist__progress span {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: var(--gradient-primary);
  transition: width var(--transition-normal);
}

.setup-checklist__dismiss,
.setup-checklist__expand {
  border: none;
  background: transparent;
  color: var(--text-secondary);
  cursor: pointer;
  font: inherit;
  font-size: 0.78rem;
}

.setup-checklist__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  align-items: center;
}

.setup-checklist__steps {
  list-style: none;
  display: grid;
  gap: 0.65rem;
  margin: 0.75rem 0 0;
  padding: 0;
}

.setup-checklist__steps li {
  display: flex;
  gap: 0.6rem;
  color: var(--text-secondary);
}

.setup-checklist__steps li.is-current {
  color: var(--text-primary);
}

.setup-checklist__mark {
  width: 1.25rem;
  flex: none;
  font-weight: 700;
  color: var(--accent-cyan);
}

.setup-checklist__label {
  font-weight: 650;
}

.setup-checklist__detail {
  margin: 0.2rem 0 0.45rem;
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.setup-checklist__link {
  display: inline-block;
  padding: 0.38rem 0.7rem;
  border: 1px solid color-mix(in srgb, var(--accent-cyan) 28%, transparent);
  border-radius: var(--radius-md);
  background: var(--accent-cyan-soft);
  color: var(--accent-cyan);
  font: inherit;
  font-size: 0.85rem;
  font-weight: 650;
  cursor: pointer;
}
</style>
