<template>
  <PageSection v-if="visible" class="setup-checklist" :title="currentStep?.label" variant="compact">
    <template #meta>
      <Tag :value="progressLabel" severity="info" />
    </template>
    <template #actions>
      <Button
        v-if="currentStep?.id === 'start'"
        class="setup-checklist__link"
        :label="currentStep.action"
        icon="pi pi-play"
        size="small"
        severity="secondary"
        outlined
        :loading="starting"
        @click="activateStep"
      />
      <RouterLink
        v-else-if="currentStep?.to"
        class="page-action-link setup-checklist__link"
        :to="currentStep.to"
      >
        {{ currentStep.action }}
      </RouterLink>
      <Button
        class="setup-checklist__expand"
        :label="expanded ? 'Hide steps' : 'Show all steps'"
        size="small"
        text
        severity="secondary"
        :aria-expanded="expanded"
        aria-controls="setup-steps"
        @click="expanded = !expanded"
      />
      <Button
        class="setup-checklist__dismiss"
        label="Dismiss"
        size="small"
        text
        severity="secondary"
        @click="dismiss"
      />
    </template>

    <p class="setup-checklist__detail">{{ currentStep?.detail }}</p>

    <div class="setup-checklist__progress-row">
      <progress
        class="setup-checklist__progress"
        :value="completedCount"
        :max="steps.length"
        :aria-label="`Setup progress: ${completedCount} of ${steps.length} steps complete`"
      />
      <span class="setup-checklist__progress-label">{{ progressLabel }}</span>
    </div>

    <StateBanner v-if="setupError" severity="error" :message="setupError" />

    <ol v-if="expanded" id="setup-steps" class="setup-checklist__steps">
      <li
        v-for="step in steps"
        :key="step.id"
        :class="{ 'is-done': step.done, 'is-current': step.current }"
      >
        <span class="setup-checklist__mark" aria-hidden="true">{{
          step.done ? '✓' : step.current ? '→' : '○'
        }}</span>
        <div class="setup-checklist__step-copy">
          <div class="setup-checklist__label">{{ step.label }}</div>
          <p class="setup-checklist__detail">{{ step.detail }}</p>
          <RouterLink
            v-if="step.to"
            class="setup-checklist__step-link"
            :to="step.to"
            @click="expanded = false"
          >
            {{ step.action }}
          </RouterLink>
        </div>
      </li>
    </ol>
  </PageSection>
  <Button
    v-else-if="descriptorsReady && dismissed && !verifiedSuccess"
    class="setup-reopen"
    label="Show setup guide"
    size="small"
    severity="secondary"
    outlined
    @click="reopen"
  />
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { RouterLink, useRouter } from 'vue-router'
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import PageSection from '@/components/common/PageSection.vue'
import StateBanner from '@/components/common/StateBanner.vue'
import { useModelStore } from '@/stores/models'
import { useEnginesStore } from '@/stores/engines'
import { modelRuntimeState } from '@/utils/modelPresentation'
import { engineSetupState } from '@/composables/engineReadiness'

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
const descriptorError = ref(false)

const quants = computed(() =>
  (modelStore.models || []).flatMap((group) => group.quantizations || []),
)

const hasModels = computed(() => quants.value.length > 0)

const configReviewed = computed(() => quants.value.some((quant) => quant?.config_reviewed))

const hasVerifiedRunning = computed(() =>
  quants.value.some((quant) => modelRuntimeState(quant).actionable),
)

function isStartingState(state) {
  return state.key === 'starting' || state.key === 'loading'
}

const hasActiveModel = computed(() =>
  quants.value.some((quant) => quant?.is_active && !isStartingState(modelRuntimeState(quant))),
)

const hasLoadingModel = computed(() =>
  quants.value.some((quant) => isStartingState(modelRuntimeState(quant))),
)

const verifiedRunningModel = computed(
  () => quants.value.find((quant) => modelRuntimeState(quant).actionable) || null,
)

const activeModel = computed(() => quants.value.find((quant) => quant?.is_active) || null)

const setupModel = computed(() => quants.value.find((q) => !q.config_reviewed) || quants.value[0])

const startableModel = computed(
  () => quants.value.find((quant) => quant.config_reviewed && !quant.is_active) || null,
)

const proxyKnown = computed(() => enginesStore.systemStatus?.proxy_status != null)
const proxyHealthy = computed(() => Boolean(enginesStore.systemStatus?.proxy_status?.healthy))
const proxyOffline = computed(() => proxyKnown.value && !proxyHealthy.value)

// A failed descriptor request must not look like a fresh install. The store
// keeps the last successful list, so only an empty list plus a failed request
// becomes the explicit "unknown" state.
const engine = computed(() => {
  const state = engineSetupState(enginesStore.engineDescriptors)
  if (state.key === 'missing' && descriptorError.value) return { key: 'unknown', names: [] }
  return state
})

const engineReady = computed(() => engine.value.key === 'ready')

const engineStep = computed(() => {
  const names = engine.value.names.join(', ')
  if (engine.value.key === 'ready') {
    return {
      label: 'Engine ready',
      detail: names ? `${names} can launch a model.` : 'An engine can launch a model.',
      action: 'Manage engines',
    }
  }
  if (engine.value.key === 'inactive') {
    return {
      label: 'Activate an engine',
      detail: names
        ? `Installed but not active: ${names}. Activate a version to start a model.`
        : 'An engine is installed but not active. Activate a version to start a model.',
      action: 'Activate an engine',
    }
  }
  if (engine.value.key === 'unknown') {
    return {
      label: 'Check your engine',
      detail:
        'Engine status could not be loaded. Open Engines to verify an engine is installed and active.',
      action: 'Open engines',
    }
  }
  return {
    label: 'Install an engine',
    detail: 'Install and activate an engine before starting a model.',
    action: 'Install an engine',
  }
})

const connectTarget = computed(() =>
  proxyHealthy.value && verifiedRunningModel.value ? '/models' : '/proxy',
)

const connectAction = computed(() =>
  connectTarget.value === '/models' ? 'Open model library' : 'Open proxy',
)

const connectDetail = computed(() => {
  if (proxyOffline.value) {
    return activeModel.value
      ? 'The inference proxy is offline. Start the proxy to reach the running model, then connect a client.'
      : 'The inference proxy is offline. Start it before connecting a client.'
  }
  if (!proxyKnown.value) {
    return 'Proxy status is unavailable. Open Proxy to check it before connecting a client.'
  }
  return 'Open Connect on the running model to copy its endpoint and a sample request.'
})

const steps = computed(() => {
  const items = [
    {
      id: 'model',
      label: hasModels.value ? 'Model in library' : 'Find a model',
      detail: hasModels.value
        ? `${quants.value.length} model${quants.value.length === 1 ? '' : 's'} in your library.`
        : 'Search for a GGUF model that fits your engine and download a quantization.',
      done: hasModels.value,
      to: '/search',
      action: hasModels.value ? 'Find more models' : 'Search models',
      actionable: true,
    },
    {
      id: 'engine',
      ...engineStep.value,
      done: engineReady.value,
      to: '/engines',
      actionable: true,
    },
    {
      id: 'config',
      label: 'Review configuration',
      detail: 'Open the model, check the engine and context, then save.',
      done: configReviewed.value,
      to: setupModel.value
        ? `/models/${encodeURIComponent(setupModel.value.id)}/config`
        : '/models',
      action: 'Configure model',
      actionable: Boolean(setupModel.value),
    },
    {
      id: 'start',
      label: 'Start the model',
      detail: hasVerifiedRunning.value
        ? 'A verified model is running.'
        : hasActiveModel.value
          ? 'A model is active, but the proxy has not verified it yet.'
          : 'Start a configured model from the library.',
      // A model that is already active has finished this step. If the proxy
      // cannot verify it, the next actionable step is the proxy — not a
      // second Start that would be rejected.
      done: hasVerifiedRunning.value || hasActiveModel.value,
      to: null,
      action: 'Start configured model',
      actionable: Boolean(startableModel.value),
    },
    {
      id: 'connect',
      label: 'Connect a client',
      detail: connectDetail.value,
      done: hasVerifiedRunning.value && proxyHealthy.value,
      to: connectTarget.value,
      action: connectAction.value,
      actionable: true,
    },
  ]
  const currentId = items.find((step) => !step.done && step.actionable !== false)?.id
  return items.map((step) => ({ ...step, current: step.id === currentId }))
})

const currentStep = computed(() => steps.value.find((step) => step.current) || null)
const completedCount = computed(() => steps.value.filter((step) => step.done).length)
const progressLabel = computed(
  () => `${completedCount.value} of ${steps.value.length} steps complete`,
)

// A verified running model proves an engine is ready, so a transient
// descriptor failure must not bring the guide back after it was dismissed by
// success. Regression (stopped model or unhealthy proxy) still reopens it.
const verifiedSuccess = computed(
  () => descriptorsReady.value && hasVerifiedRunning.value && proxyHealthy.value,
)

const visible = computed(
  () =>
    descriptorsReady.value &&
    !dismissed.value &&
    !verifiedSuccess.value &&
    // A loading model is already progressing; there is no corrective action
    // to offer until it becomes ready or fails.
    !hasLoadingModel.value &&
    Boolean(currentStep.value),
)

onMounted(async () => {
  try {
    await enginesStore.fetchEngineDescriptors()
    descriptorError.value = false
  } catch {
    descriptorError.value = true
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
  const model = startableModel.value
  if (!model || starting.value) return
  starting.value = true
  try {
    await modelStore.startModel(model.id)
    await modelStore.fetchModels()
  } catch {
    setupError.value =
      'Could not start the model. Open its configuration to check the engine, then retry.'
  } finally {
    starting.value = false
  }
}

function reopen() {
  dismissed.value = false
  expanded.value = true
  try {
    localStorage.removeItem(DISMISS_KEY)
  } catch {
    /* Session-only preference. */
  }
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
.setup-checklist__detail {
  margin: 0;
  color: var(--text-secondary);
  font-size: var(--text-sm);
  line-height: 1.6;
}

.setup-checklist__progress-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem 0.75rem;
}

.setup-checklist__progress {
  flex: 1 1 12rem;
  height: 0.4rem;
  overflow: hidden;
  border: 0;
  border-radius: 999px;
  background: var(--bg-tertiary);
}

.setup-checklist__progress::-webkit-progress-bar {
  border-radius: 999px;
  background: var(--bg-tertiary);
}

.setup-checklist__progress::-webkit-progress-value {
  border-radius: 999px;
  background: var(--accent-primary);
}

.setup-checklist__progress::-moz-progress-bar {
  border-radius: 999px;
  background: var(--accent-primary);
}

.setup-checklist__progress-label {
  color: var(--text-secondary);
  font-size: var(--text-xs);
  font-variant-numeric: tabular-nums;
}

.setup-checklist__steps {
  list-style: none;
  display: grid;
  gap: 0.75rem;
  margin: 0;
  padding: 0.75rem 0 0;
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
  flex: none;
  width: 1.25rem;
  color: var(--accent-primary);
}

.setup-checklist__step-copy {
  min-width: 0;
}

.setup-checklist__label {
  font-size: var(--text-sm);
  font-weight: 600;
}

.setup-checklist__step-link {
  display: inline-flex;
  align-items: center;
  min-height: 2rem;
  color: var(--accent-primary);
  font-size: var(--text-sm);
  font-weight: 600;
}

.setup-reopen {
  align-self: flex-start;
}

@media (max-width: 480px) {
  .setup-checklist__progress-row {
    align-items: stretch;
  }
}
</style>
