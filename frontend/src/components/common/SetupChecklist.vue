<template>
  <section v-if="visible" class="setup-checklist" aria-labelledby="setup-checklist-title">
    <div class="setup-checklist__summary">
      <div class="setup-checklist__copy">
        <h2 id="setup-checklist-title">{{ currentStep?.label }}</h2>
        <p class="setup-checklist__detail">{{ currentStep?.detail }}</p>
      </div>
      <div class="setup-checklist__actions">
        <button
          v-if="currentStep?.to || currentStep?.id === 'start'"
          type="button"
          class="setup-checklist__link"
          :disabled="starting"
          @click="activateStep"
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
    <p v-if="setupError" class="setup-checklist__detail" role="alert">{{ setupError }}</p>
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
  <button v-else-if="dismissed && !verifiedSuccess" type="button" class="setup-reopen" @click="reopen">Show setup guide</button>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useModelStore } from '@/stores/models'
import { useEnginesStore } from '@/stores/engines'
import { modelRuntimeState } from '@/utils/modelPresentation'
import { runnableEngineIds } from '@/composables/engineReadiness'

const router = useRouter()

const DISMISS_KEY = 'llama-studio.setup-checklist.dismissed'

const modelStore = useModelStore()
const enginesStore = useEnginesStore()
const dismissed = ref(
  typeof localStorage !== 'undefined' && localStorage.getItem(DISMISS_KEY) === '1',
)
const expanded = ref(false)
const starting = ref(false)
const setupError = ref('')
const descriptorsReady = ref(false)

const quants = computed(() =>
  (modelStore.models || []).flatMap((group) => group.quantizations || []),
)

const engineReady = computed(() => runnableEngineIds(enginesStore.engineDescriptors).length > 0)

const hasModels = computed(() => quants.value.length > 0)

const configReviewed = computed(() => quants.value.some((quant) => quant?.config_reviewed))

const hasVerifiedRunning = computed(() => quants.value.some((quant) => modelRuntimeState(quant).actionable))

const setupModel = computed(() => quants.value.find((q) => !q.config_reviewed) || quants.value[0])
const runningModel = computed(() => quants.value.find((q) => q.is_active && !['stale', 'unreachable', 'unknown'].includes(q.runtime_quality)))
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
      to: setupModel.value ? `/models/${encodeURIComponent(setupModel.value.id)}/config` : '/models',
      action: 'Configure model',
    },
    {
      id: 'start',
      label: 'Start the model',
      detail: 'Start a configured model from the library.',
      done: hasVerifiedRunning.value,
      to: null,
      action: 'Start configured model',
    },
    {
      id: 'connect',
      label: 'Ready to connect',
      detail:
        proxyKnown.value && !proxyHealthy.value
          ? 'The inference proxy is offline. Start it before connecting a client.'
          : 'Open Connect on a running model to copy its endpoint and a sample request.',
      done: hasVerifiedRunning.value && proxyHealthy.value,
      to: proxyHealthy.value && runningModel.value ? '/models' : '/proxy',
      action: proxyHealthy.value ? 'Open model library' : 'Open proxy',
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

async function activateStep() {
  setupError.value = ''
  if (currentStep.value?.id !== 'start') {
    if (currentStep.value?.to) router.push(currentStep.value.to)
    return
  }
  const model = quants.value.find((quant) => quant.config_reviewed && !quant.is_active) || setupModel.value
  if (!model || starting.value) return
  starting.value = true
  try {
    await modelStore.startModel(model.id)
    await modelStore.fetchModels()
  } catch {
    setupError.value = 'Could not start the model. Open its configuration to check the engine, then retry.'
  } finally { starting.value = false }
}

function reopen() {
  dismissed.value = false
  expanded.value = true
  try { localStorage.removeItem(DISMISS_KEY) } catch { /* Session-only preference. */ }
}

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

@media (max-width: 480px) {
  .setup-checklist {
    padding: 0.75rem 1rem;
  }

  .setup-checklist__actions {
    gap: 0.1rem;
  }

  .setup-checklist__actions button {
    padding: 0.4rem 0.35rem;
    font-size: 0.75rem;
  }
}

.setup-reopen {
  align-self: flex-start;
  padding: 0.5rem 0.75rem;
  background: transparent;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  cursor: pointer;
}
</style>
