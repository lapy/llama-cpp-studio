<template>
  <section v-if="visible" class="setup-checklist" aria-labelledby="setup-checklist-title">
    <div class="setup-checklist__head">
      <h2 id="setup-checklist-title">Get to a running model</h2>
      <button type="button" class="setup-checklist__dismiss" @click="dismiss">Dismiss</button>
    </div>
    <ol class="setup-checklist__steps">
      <li v-for="step in steps" :key="step.id" :class="{ 'is-done': step.done, 'is-current': step.current }">
        <span class="setup-checklist__mark" aria-hidden="true">{{ step.done ? '✓' : step.current ? '→' : '○' }}</span>
        <div>
          <div class="setup-checklist__label">{{ step.label }}</div>
          <p class="setup-checklist__detail">{{ step.detail }}</p>
          <button v-if="step.to && !step.done" type="button" class="setup-checklist__link" @click="router.push(step.to)">{{ step.action }}</button>
        </div>
      </li>
    </ol>
  </section>
</template>

<script setup>
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useModelStore } from '@/stores/models'
import { useEnginesStore } from '@/stores/engines'

const router = useRouter()

const DISMISS_KEY = 'llama-studio.setup-checklist.dismissed'

const modelStore = useModelStore()
const enginesStore = useEnginesStore()
const dismissed = ref(typeof localStorage !== 'undefined' && localStorage.getItem(DISMISS_KEY) === '1')

const engineReady = computed(() => {
  const lists = [
    enginesStore.llamaVersions,
    enginesStore.ikLlamaVersions,
    enginesStore.lmdeployVersions,
    enginesStore.onecatVllmVersions,
    enginesStore.sglangVersions,
    enginesStore.sglangV100Versions,
    enginesStore.vllmVersions,
    enginesStore.audioCppVersions,
  ]
  return lists.some((list) => Array.isArray(list) && list.length > 0)
})

const hasModels = computed(() => (modelStore.models || []).some((group) => (group.quantizations || []).length > 0))

const hasRunning = computed(() =>
  (modelStore.models || []).some((group) =>
    (group.quantizations || []).some((quant) => quant?.is_active),
  ),
)

const proxyKnown = computed(() => enginesStore.systemStatus?.proxy_status != null)
const proxyHealthy = computed(() => Boolean(enginesStore.systemStatus?.proxy_status?.healthy))

const steps = computed(() => {
  const items = [
    {
      id: 'task',
      label: 'Choose a task',
      detail: 'Search by what you want to run: text, vision, or audio.',
      done: hasModels.value,
      to: '/search',
      action: 'Discover models',
    },
    {
      id: 'engine',
      label: 'Prepare a compatible engine',
      detail: engineReady.value
        ? 'An engine is installed. Activate one if it is not already in use.'
        : 'Install an engine before the first launch. GGUF models need llama.cpp or ik_llama.cpp.',
      done: engineReady.value,
      to: '/engines',
      action: engineReady.value ? 'Review engines' : 'Install an engine',
    },
    {
      id: 'download',
      label: 'Download a model',
      detail: 'Saved models appear in this library.',
      done: hasModels.value,
      to: '/search',
      action: 'Search and download',
    },
    {
      id: 'config',
      label: 'Review configuration',
      detail: 'Check context size, GPU placement, and the engine, then save.',
      done: hasModels.value,
      to: hasModels.value ? null : '/search',
      action: 'Open a model',
    },
    {
      id: 'start',
      label: 'Start the model',
      detail: 'Starting loads weights. A restart is required after some saved changes.',
      done: hasRunning.value,
      to: null,
      action: '',
    },
    {
      id: 'connect',
      label: 'Connect a client',
      detail: proxyKnown.value && !proxyHealthy.value
        ? 'llama-swap is offline, so clients cannot reach a running model yet.'
        : 'Use the llama-swap link in the header once the model is running.',
      done: hasRunning.value && proxyHealthy.value,
      to: null,
      action: '',
    },
  ]
  const currentId = items.find((step) => !step.done)?.id
  return items.map((step) => ({ ...step, current: step.id === currentId }))
})

const visible = computed(() => !dismissed.value && steps.value.some((step) => !step.done))

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
  margin-bottom: 1rem;
  padding: 0.9rem 1rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  background: var(--bg-secondary);
}

.setup-checklist__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  margin-bottom: 0.75rem;
}

.setup-checklist__head h2 {
  margin: 0;
  font-size: 1rem;
}

.setup-checklist__dismiss {
  border: none;
  background: transparent;
  color: var(--text-secondary);
  cursor: pointer;
  font: inherit;
}

.setup-checklist__steps {
  list-style: none;
  display: grid;
  gap: 0.65rem;
  margin: 0;
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

.setup-checklist__steps li.is-done .setup-checklist__label {
  color: var(--text-secondary);
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
  margin: 0.15rem 0 0;
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.setup-checklist__link {
  display: inline-block;
  margin-top: 0.2rem;
  padding: 0;
  border: none;
  background: none;
  color: var(--accent-cyan);
  font: inherit;
  font-size: 0.85rem;
  font-weight: 650;
  cursor: pointer;
}
</style>
