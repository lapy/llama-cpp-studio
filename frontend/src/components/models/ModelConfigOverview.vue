<template>
  <div class="config-overview">
    <PageSection
      title="Overview"
      description="Confirm what this model is, which engine runs it, and what to do next before editing parameters."
      variant="plain"
    >
      <PageSection
        id="config-overview-identity"
        :title="modelLabel"
        heading-tag="h3"
        variant="compact"
      >
        <template #meta>
          <Tag :value="format" severity="info" />
          <Tag v-if="model.quantization" :value="model.quantization" severity="secondary" />
          <span v-if="sizeLabel" class="config-overview__size">{{ sizeLabel }}</span>
          <span v-else class="config-overview__size">Size not reported</span>
          <Tag
            :value="status.label"
            :severity="status.severity"
            aria-label="Configuration status"
            data-overview-status
          />
        </template>

        <div class="config-overview__identity-copy">
          <a
            v-if="model.huggingface_id"
            :href="`https://huggingface.co/${model.huggingface_id}`"
            target="_blank"
            rel="noopener noreferrer"
            class="config-overview__repo"
          >
            <i class="pi pi-external-link" aria-hidden="true" />
            {{ model.huggingface_id }}
          </a>
          <p v-else class="config-muted-hint">Repository not reported for this model.</p>
        </div>

        <div class="config-overview__engine">
          <h3 class="page-subheading">Engine</h3>
          <div
            class="engine-selector"
            role="radiogroup"
            aria-label="Engine"
            @keydown="onEngineKeydown"
          >
            <button
              v-for="eng in engineOptions"
              :key="eng.value"
              type="button"
              class="engine-option"
              role="radio"
              :class="{
                selected: config.engine === eng.value,
                disabled: eng.disabled,
              }"
              :aria-checked="config.engine === eng.value ? 'true' : 'false'"
              :aria-disabled="eng.disabled ? 'true' : 'false'"
              :disabled="eng.disabled"
              :tabindex="engineTabIndex(eng)"
              v-tooltip.bottom="
                eng.disabledReason ||
                (eng.runnable || !eng.descriptor
                  ? ''
                  : 'No active version. Activate one from Engines.')
              "
              @click="$emit('select-engine', eng.value)"
            >
              <div class="engine-option-label">
                <template v-if="engineMark(eng.value)">
                  <span
                    v-if="engineMark(eng.value).text"
                    :class="['engine-mark', engineMark(eng.value).className]"
                    aria-hidden="true"
                    >{{ engineMark(eng.value).text }}</span
                  >
                  <i v-else :class="engineMark(eng.value).icon" aria-hidden="true" />
                </template>
                <span class="engine-name">{{ eng.label }}</span>
              </div>
              <small v-if="eng.disabledReason" class="engine-disabled-reason">
                {{ eng.disabledReason }}
              </small>
            </button>
          </div>
          <p v-if="!engineOptions.length" class="config-muted-hint">
            No installed engine is compatible with this model.
            <Button
              label="Open Engines"
              icon="pi pi-arrow-right"
              size="small"
              link
              @click="$emit('open-engines')"
            />
          </p>
          <p v-else-if="unavailableSavedEngine" class="config-muted-hint">
            {{ unavailableSavedEngine.label }} is saved for this model. It is hidden because it is
            not installed or not compatible.
          </p>
          <div v-if="isSglangEngine" class="page-form__field sglang-runtime-selector">
            <label class="page-form__label" for="sglang-runtime">SGLang runtime</label>
            <Select
              input-id="sglang-runtime"
              :model-value="sglangRuntime"
              :options="sglangRuntimeOptions"
              option-label="label"
              option-value="value"
              @update:model-value="$emit('update:sglang-runtime', $event)"
            />
            <small class="page-form__hint"
              >Each runtime has its own CLI parameters. Saved values are kept when switching.</small
            >
          </div>
        </div>
      </PageSection>

      <PageSection title="At a glance" heading-tag="h3" variant="compact">
        <dl class="config-overview__fact-grid">
          <div v-for="fact in facts" :key="fact.key" class="config-overview__fact">
            <dt>{{ fact.label }}</dt>
            <dd>
              <span class="config-overview__fact-value">{{ fact.value }}</span>
              <small v-if="fact.hint" class="config-overview__fact-hint">{{ fact.hint }}</small>
            </dd>
          </div>
        </dl>
      </PageSection>

      <ModelConfigAutoFit
        v-if="autoFit.supported"
        :active="autoFit.active"
        :overrides="autoFit.overrides"
        :defaults="autoFit.defaults"
        @restore="$emit('restore-auto-fit')"
        @review="$emit('review-auto-fit')"
      />

      <PageSection title="Review" heading-tag="h3" variant="compact">
        <template #meta>
          <p class="config-overview__review-summary">{{ reviewSummary }}</p>
        </template>
        <ul v-if="reviewItems.length" class="config-overview__review-list">
          <li
            v-for="item in reviewItems"
            :key="item.key"
            :class="['config-overview__review-item', `config-overview__review-item--${item.tone}`]"
          >
            <i :class="item.icon" aria-hidden="true" />
            <div class="config-overview__review-body">
              <strong>{{ item.title }}</strong>
              <p>{{ item.detail }}</p>
              <div v-if="item.section || item.engines" class="config-overview__review-links">
                <Button
                  v-if="item.section"
                  :label="item.sectionLabel"
                  size="small"
                  text
                  @click="$emit('go-to-section', item.section)"
                />
                <Button
                  v-if="item.engines"
                  label="Open Engines"
                  size="small"
                  text
                  @click="$emit('open-engines')"
                />
              </div>
            </div>
          </li>
        </ul>
        <p v-else class="config-muted-hint">No configuration warnings.</p>

        <p class="config-overview__next-step">
          Use the action bar above to save, apply, or start this model. It always shows the next
          required step.
        </p>
      </PageSection>
    </PageSection>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import Button from 'primevue/button'
import Select from 'primevue/select'
import Tag from 'primevue/tag'

import ModelConfigAutoFit from '@/components/models/ModelConfigAutoFit.vue'
import PageSection from '@/components/common/PageSection.vue'
import { ENGINE_DESCRIPTORS } from '@/engines/registry'
import { formatLibraryBytes } from '@/utils/libraryPresentation'
import { formatLabel } from '@/utils/modelPresentation'

const props = defineProps({
  model: { type: Object, required: true },
  config: { type: Object, required: true },
  status: {
    type: Object,
    default: () => ({ key: 'applied', label: 'Applied', severity: 'success', detail: '' }),
  },
  engineOptions: { type: Array, default: () => [] },
  unavailableSavedEngine: { type: Object, default: null },
  sglangRuntime: { type: String, default: 'llm' },
  sglangRuntimeOptions: { type: Array, default: () => [] },
  isSglangEngine: { type: Boolean, default: false },
  isAudioEngine: { type: Boolean, default: false },
  gpuInfo: { type: Object, default: () => ({ gpus: [], cpu_only_mode: true }) },
  modelLimits: { type: Object, default: null },
  apiEndpoint: { type: String, default: '' },
  proxyPort: { type: [Number, String], default: null },
  launchPlan: { type: Object, default: null },
  planAvailable: { type: Boolean, default: true },
  hasUnsavedChanges: { type: Boolean, default: false },
  pendingApply: { type: Boolean, default: false },
  configSaveHeld: { type: Boolean, default: false },
  modelRuntimeQuality: { type: String, default: '' },
  running: { type: Boolean, default: false },
  starting: { type: Boolean, default: false },
  autoFit: {
    type: Object,
    default: () => ({ supported: false, active: false, overrides: [], defaults: null }),
  },
})

const emit = defineEmits([
  'select-engine',
  'update:sglang-runtime',
  'open-engines',
  'go-to-section',
  'restore-auto-fit',
  'review-auto-fit',
])

const modelLabel = computed(
  () => props.model?.display_name || props.model?.base_model_name || 'This model',
)
const format = computed(() => formatLabel(props.model?.format || 'gguf'))
const sizeLabel = computed(() => {
  const bytes = Number(props.model?.file_size)
  return bytes > 0 ? formatLibraryBytes(bytes) : ''
})

const contextFact = computed(() => {
  const configured = props.config?.session_len ?? props.config?.ctx_size
  const numeric = Number(configured)
  if (configured != null && configured !== '' && Number.isFinite(numeric) && numeric > 0) {
    return {
      key: 'context',
      label: 'Context window',
      value: `${numeric.toLocaleString()} tokens`,
      hint: 'Saved for this model.',
    }
  }
  const limit = Number(props.modelLimits?.max_context_length)
  if (Number.isFinite(limit) && limit > 0) {
    return {
      key: 'context',
      label: 'Context window',
      value: 'Engine default',
      hint: `Model maximum ${limit.toLocaleString()} tokens.`,
    }
  }
  return {
    key: 'context',
    label: 'Context window',
    value: 'Engine default',
    hint: 'No model context limit is reported.',
  }
})

const gpuFact = computed(() => {
  const gpus = Array.isArray(props.gpuInfo?.gpus) ? props.gpuInfo.gpus : []
  const names = gpus.map((gpu) => gpu.name).filter(Boolean)
  const mode = props.config?.gpu_mode || 'inherit'
  const selected = Array.isArray(props.config?.gpu_devices) ? props.config.gpu_devices : []
  let placement = 'Engine default'
  if (props.gpuInfo?.cpu_only_mode || mode === 'cpu') placement = 'CPU only'
  else if (mode === 'selected') {
    placement = selected.length
      ? `${selected.length} of ${gpus.length || selected.length} GPUs`
      : 'No GPUs selected'
  } else if (gpus.length) placement = `All ${gpus.length} GPUs`

  const layers = props.config?.n_gpu_layers
  let offload = ''
  if (!props.isAudioEngine) {
    if (layers == null || layers === '') offload = 'engine-managed offload'
    else if (Number(layers) < 0) offload = 'all layers offloaded'
    else if (Number(layers) === 0) offload = 'no layers offloaded'
    else offload = `${Number(layers).toLocaleString()} layers offloaded`
  }

  return {
    key: 'gpu',
    label: 'GPU & offload',
    value: [placement, offload].filter(Boolean).join(' · '),
    hint: names.length ? names.join(', ') : 'No GPU inventory is reported.',
  }
})

const endpointFact = computed(() => {
  if (props.apiEndpoint) {
    return {
      key: 'endpoint',
      label: 'Endpoint',
      value: props.apiEndpoint,
      hint: 'Route reported by the engine contract.',
    }
  }
  if (props.proxyPort) {
    return {
      key: 'endpoint',
      label: 'Endpoint',
      value: `127.0.0.1:${props.proxyPort}`,
      hint: 'Studio proxy port.',
    }
  }
  return {
    key: 'endpoint',
    label: 'Endpoint',
    value: 'Studio proxy',
    hint: 'The inference endpoint is managed by the Studio proxy.',
  }
})

const applyFact = computed(() => {
  if (props.hasUnsavedChanges) {
    return {
      key: 'apply',
      label: 'Apply state',
      value: 'Unsaved edits',
      hint: 'Save changes stores this form; apply makes it active.',
    }
  }
  if (props.pendingApply) {
    return {
      key: 'apply',
      label: 'Apply state',
      value: 'Needs apply',
      hint: 'Saved settings are waiting to be applied.',
    }
  }
  if (props.running) {
    return {
      key: 'apply',
      label: 'Apply state',
      value: 'In use',
      hint: 'The model is running with the saved settings.',
    }
  }
  if (props.status.key === 'unknown' || props.status.key === 'stale') {
    return {
      key: 'apply',
      label: 'Apply state',
      value: props.status.label,
      hint: 'The current apply state cannot be verified.',
    }
  }
  return {
    key: 'apply',
    label: 'Apply state',
    value: props.status.label,
    hint: 'Saved settings are published for the next start.',
  }
})

const facts = computed(() => [
  contextFact.value,
  gpuFact.value,
  endpointFact.value,
  applyFact.value,
])

function engineMark(engineId) {
  return ENGINE_DESCRIPTORS[engineId]?.card?.mark || null
}

function engineTabIndex(engine) {
  if (engine.disabled) return -1
  return props.config.engine === engine.value ? 0 : -1
}

function onEngineKeydown(event) {
  const options = props.engineOptions.filter((option) => !option.disabled)
  if (!options.length) return
  const index = Math.max(
    0,
    options.findIndex((option) => option.value === props.config.engine),
  )
  let next
  if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = (index + 1) % options.length
  else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp')
    next = (index - 1 + options.length) % options.length
  else if (event.key === 'Home') next = 0
  else if (event.key === 'End') next = options.length - 1
  else return
  event.preventDefault()
  const engine = options[next]
  if (!engine) return
  const buttons = event.currentTarget?.querySelectorAll?.('[role="radio"]:not([disabled])')
  buttons?.[next]?.focus()
  if (engine.value !== props.config.engine) emit('select-engine', engine.value)
}

function sectionForReason(reason) {
  const text = String(reason || '').toLowerCase()
  if (/(alias|launch|environment|\benv\b|argument|command)/.test(text)) {
    return { section: 'launch', sectionLabel: 'Open Launch' }
  }
  if (/(engine|version|install|activate)/.test(text)) {
    return { section: 'overview', sectionLabel: 'Review engine' }
  }
  return { section: 'runtime', sectionLabel: 'Open Runtime' }
}

const reviewItems = computed(() => {
  const items = []
  if (props.configSaveHeld) {
    items.push({
      key: 'held',
      tone: 'danger',
      icon: 'pi pi-exclamation-circle',
      title: 'Saving is paused',
      detail:
        'Resolve the configuration problem reported on this page before saving or starting this model.',
    })
  } else if (props.hasUnsavedChanges) {
    items.push({
      key: 'unsaved',
      tone: 'warn',
      icon: 'pi pi-pencil',
      title: 'Unsaved changes',
      detail: 'Save these edits before applying or starting the model.',
    })
  } else if (props.pendingApply) {
    items.push({
      key: 'pending',
      tone: 'warn',
      icon: 'pi pi-bolt',
      title: 'Saved · needs apply',
      detail:
        'The saved settings are not in use yet. Apply them from the action bar before starting the model.',
    })
  }

  if (props.launchPlan?.action === 'invalid') {
    const reasons = Array.isArray(props.launchPlan.reasons) ? props.launchPlan.reasons : []
    for (const [index, reason] of reasons.entries()) {
      items.push({
        key: `invalid-${index}`,
        tone: 'danger',
        icon: 'pi pi-times-circle',
        title: 'Configuration cannot be compiled',
        detail: reason,
        ...sectionForReason(reason),
      })
    }
  }
  for (const [index, warning] of (Array.isArray(props.launchPlan?.warnings)
    ? props.launchPlan.warnings
    : []
  ).entries()) {
    items.push({
      key: `warning-${index}`,
      tone: 'warn',
      icon: 'pi pi-exclamation-triangle',
      title: 'Launch warning',
      detail: warning,
      section: 'launch',
      sectionLabel: 'Open Launch',
    })
  }

  if (!props.engineOptions.length) {
    items.push({
      key: 'engine',
      tone: 'danger',
      icon: 'pi pi-server',
      title: 'No compatible engine',
      detail: 'Install or activate an engine that supports this model artifact.',
      engines: true,
    })
  }

  if (!props.planAvailable) {
    items.push({
      key: 'plan',
      tone: 'info',
      icon: 'pi pi-info-circle',
      title: 'Runtime plan unavailable',
      detail:
        'Saved settings could not be checked against the running model. Refresh this page to try again.',
    })
  } else if (props.modelRuntimeQuality === 'unreachable') {
    items.push({
      key: 'unreachable',
      tone: 'warn',
      icon: 'pi pi-question-circle',
      title: 'Proxy unreachable',
      detail:
        'The running state is unverified. Start stays available, but it may fail until the proxy responds.',
    })
  } else if (props.modelRuntimeQuality === 'stale') {
    items.push({
      key: 'stale',
      tone: 'warn',
      icon: 'pi pi-clock',
      title: 'Status is stale',
      detail: 'The last observation is not current. Refresh to verify the running state.',
    })
  }

  if (props.starting && !props.running) {
    items.push({
      key: 'starting',
      tone: 'info',
      icon: 'pi pi-spin pi-spinner',
      title: 'Model is starting',
      detail: 'This page updates when the model reports running.',
    })
  }

  if (!items.length) {
    items.push({
      key: 'ready',
      tone: 'success',
      icon: props.running ? 'pi pi-check-circle' : 'pi pi-flag',
      title: props.running ? 'Running with these settings' : 'Ready to start',
      detail: props.running
        ? 'The model is using the saved configuration.'
        : 'The configuration is saved and published. Start the model when you are ready.',
    })
  }
  return items
})

const reviewSummary = computed(() => {
  if (props.configSaveHeld) return 'One configuration problem is blocking save and start.'
  if (props.hasUnsavedChanges) return 'Save the form before this model can start with it.'
  if (props.pendingApply) return 'Saved settings are waiting to be applied.'
  if (props.launchPlan?.action === 'invalid') return 'Fix the configuration before applying.'
  if (!props.engineOptions.length) return 'An engine is required before this model can run.'
  if (props.starting && !props.running) return 'The model is starting.'
  if (props.running) return 'The model is active.'
  return 'The saved configuration is ready.'
})
</script>

<style scoped>
.config-overview {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.config-overview__identity-copy {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
  min-width: 0;
}

.config-overview__repo {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  font-size: var(--text-xs);
  color: var(--accent-primary);
  text-decoration: none;
  overflow-wrap: anywhere;
}

.config-overview__repo:hover {
  text-decoration: underline;
}

.config-overview__size {
  font-size: var(--text-xs);
  color: var(--text-secondary);
}

.config-overview__engine {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  margin-top: 0.75rem;
  padding-top: 0.75rem;
  border-top: 1px solid var(--border-primary);
}

.config-overview__fact-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 12rem), 1fr));
  gap: 0.75rem;
  margin: 0.65rem 0 0;
}

.config-overview__fact {
  min-width: 0;
}

.config-overview__fact dt {
  font-size: var(--text-xs);
  font-weight: 600;
  color: var(--text-secondary);
}

.config-overview__fact dd {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
  margin: 0.2rem 0 0;
}

.config-overview__fact-value {
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--text-primary);
  overflow-wrap: anywhere;
}

.config-overview__fact-hint {
  font-size: var(--text-xs);
  line-height: 1.35;
  color: var(--text-secondary);
}

.config-overview__review-summary {
  margin: 0;
  font-size: var(--text-xs);
  color: var(--text-secondary);
  text-align: right;
}

.config-overview__next-step {
  margin: 0.65rem 0 0;
  padding-top: 0.65rem;
  border-top: 1px solid var(--border-primary);
  color: var(--text-secondary);
  font-size: var(--text-xs);
  line-height: 1.45;
}

.config-overview__review-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin: 0.65rem 0 0;
  padding: 0;
  list-style: none;
}

.config-overview__review-item {
  display: flex;
  align-items: flex-start;
  gap: 0.55rem;
  padding: 0.55rem 0.65rem;
  border: 1px solid var(--border-primary);
  border-left-width: 3px;
  border-radius: var(--radius-md);
  background: var(--bg-surface);
}

.config-overview__review-item > .pi {
  margin-top: 0.1rem;
  color: var(--text-secondary);
}

.config-overview__review-item--danger {
  border-left-color: var(--status-error);
}

.config-overview__review-item--danger > .pi {
  color: var(--status-error);
}

.config-overview__review-item--warn {
  border-left-color: var(--status-warning);
}

.config-overview__review-item--warn > .pi {
  color: var(--status-warning);
}

.config-overview__review-item--success {
  border-left-color: var(--status-success);
}

.config-overview__review-item--success > .pi {
  color: var(--status-success);
}

.config-overview__review-item--info {
  border-left-color: var(--status-info);
}

.config-overview__review-item--info > .pi {
  color: var(--status-info);
}

.config-overview__review-body {
  display: flex;
  flex-direction: column;
  gap: 0.1rem;
  min-width: 0;
}

.config-overview__review-body strong {
  font-size: 0.8125rem;
  color: var(--text-primary);
}

.config-overview__review-body p {
  margin: 0;
  font-size: var(--text-xs);
  line-height: 1.45;
  color: var(--text-secondary);
  overflow-wrap: anywhere;
}

.config-overview__review-links {
  display: flex;
  flex-wrap: wrap;
  gap: 0.15rem 0.35rem;
  margin-top: 0.15rem;
}

.config-overview .engine-selector {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
}

.config-overview .engine-option {
  display: inline-flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  width: auto;
  padding: 0.22rem 0.6rem;
  border-radius: var(--radius-md);
  border: 1px solid var(--border-primary);
  background: transparent;
  color: inherit;
  font: inherit;
  font-size: 0.8125rem;
  text-align: center;
  cursor: pointer;
  user-select: none;
  transition:
    border-color 0.15s,
    background 0.15s;
}

.config-overview .engine-option:hover {
  border-color: var(--accent-cyan);
  background: var(--accent-cyan-soft);
}

.config-overview .engine-option.selected {
  border-color: var(--accent-cyan);
  background: var(--accent-cyan-soft);
  color: var(--accent-cyan);
  font-weight: 600;
}

.config-overview .engine-option.disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

.config-overview .engine-option.disabled:hover {
  border-color: var(--border-primary);
  background: transparent;
}

.config-overview .engine-option:focus-visible {
  outline: 2px solid var(--accent-cyan);
  outline-offset: 2px;
}

.config-overview .engine-option-label {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
}

.config-overview .engine-name {
  font-size: 0.8125rem;
}

.config-overview .engine-mark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 1.5rem;
  height: 1.5rem;
  padding: 0 0.45rem;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-primary);
  background: var(--bg-tertiary);
  color: var(--text-secondary);
  font-size: var(--text-xs);
  font-weight: 700;
  line-height: 1;
  letter-spacing: 0.04em;
}

.config-overview .engine-disabled-reason {
  flex-basis: 100%;
  max-width: 100%;
  margin: 0.2rem 0 0;
  color: var(--text-secondary);
  font-size: var(--text-xs);
  line-height: 1.3;
  text-align: center;
}

@media (max-width: 768px) {
  .config-overview .engine-option {
    flex: 1 1 auto;
    padding: 0.4rem 0.6rem;
  }
}
</style>
