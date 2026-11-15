<template>
  <div class="config-card">
    <div class="section-label section-label--inline">
      {{ title }}
      <small class="section-hint"
        ><code>{{ endpoint }}</code></small
      >
    </div>

    <Message v-if="guidance" severity="info" :closable="false" class="config-scan-message">
      {{ guidance }}
    </Message>

    <div v-if="defaultsPreview" class="setparams-preview">
      <button
        type="button"
        class="setparams-preview__label setparams-preview__toggle"
        :aria-expanded="showSetParamsPreview"
        @click="showSetParamsPreview = !showSetParamsPreview"
      >
        <span>Saved request defaults preview</span>
        <i
          class="pi"
          :class="showSetParamsPreview ? 'pi-chevron-up' : 'pi-chevron-down'"
          aria-hidden="true"
        />
      </button>
      <pre v-if="showSetParamsPreview" class="setparams-preview__code">{{
        JSON.stringify(defaultsPreview, null, 2)
      }}</pre>
    </div>

    <template v-if="profiled">
      <div v-if="groups.length" class="tts-subsection">
        <div v-for="group in groups" :key="group.id" class="tts-speech-group">
          <div v-if="groups.length > 1" class="tts-speech-group__label">
            {{ group.label }}
          </div>
          <div class="params-grid section-params">
            <div
              v-for="field in group.fields"
              :key="`${group.id}-${field.key}`"
              class="param-field"
            >
              <label class="param-field__label">
                {{ field.label }}
                <i
                  v-if="field.description || field.hint"
                  class="pi pi-info-circle param-info"
                  v-tooltip.top="requestDefaultFieldTooltip(field)"
                  tabindex="0"
                  aria-label="About request default field"
                />
              </label>
              <ToggleSwitch
                v-if="field.type === 'bool'"
                :model-value="Boolean(fieldValue(field))"
                @update:model-value="(value) => updateField(field, value)"
              />
              <InputNumber
                v-else-if="field.type === 'int' || field.type === 'float'"
                :model-value="fieldValue(field)"
                :minFractionDigits="field.type === 'float' ? 1 : 0"
                :maxFractionDigits="field.type === 'float' ? 6 : 0"
                :min="field.minimum"
                :max="field.maximum"
                class="param-input"
                @update:model-value="(value) => updateField(field, value)"
              />
              <Textarea
                v-else-if="field.type === 'textarea'"
                :model-value="fieldValue(field) || ''"
                :placeholder="field.placeholder || ''"
                rows="2"
                class="w-full textarea-cli param-input"
                @update:model-value="(value) => updateField(field, value)"
              />
              <Select
                v-else-if="fieldOptions(field).length"
                :model-value="fieldValue(field) || ''"
                :options="fieldOptions(field)"
                optionLabel="label"
                optionValue="value"
                :placeholder="field.placeholder || 'Choose a value'"
                showClear
                editable
                class="param-input"
                @update:model-value="(value) => updateField(field, value)"
              />
              <InputText
                v-else
                :model-value="fieldValue(field) || ''"
                :placeholder="field.placeholder || ''"
                class="param-input"
                @update:model-value="(value) => updateField(field, value)"
              />
            </div>
          </div>
        </div>
      </div>
    </template>

    <Message v-else severity="secondary" :closable="false" class="config-scan-message">
      No request defaults profile is available for this audio model.
    </Message>
  </div>

  <details class="config-card config-details-block">
    <summary class="config-details-summary">API example</summary>
    <div class="config-details-body">
      <div class="tts-subsection__head">
        <p class="config-muted-hint">
          Model id:
          <code>{{ modelId || 'your-model-id' }}</code>
        </p>
        <Button
          label="Copy curl"
          icon="pi pi-copy"
          size="small"
          severity="secondary"
          outlined
          type="button"
          @click="copyApiExample"
        />
      </div>
      <Textarea
        :model-value="apiExample"
        readonly
        rows="10"
        class="w-full textarea-cli cmd-preview-textarea"
        autoResize
      />
    </div>
  </details>

  <details v-if="capabilities.length" class="config-card config-details-block">
    <summary class="config-details-summary">
      Request-only parameters
      <span class="config-details-summary__count">{{ capabilities.length }}</span>
    </summary>
    <div class="config-details-body">
      <p class="config-muted-hint">
        Available on requests, not as startup settings. Prefer Defaults above when a matching field
        exists.
      </p>
      <div class="config-toolbar__row">
        <span class="p-input-icon-left config-search-wrap">
          <i class="pi pi-search" aria-hidden="true" />
          <InputText
            v-model="capabilityFilter"
            type="search"
            placeholder="Filter…"
            class="config-search-input"
            aria-label="Filter request parameters"
          />
        </span>
      </div>
      <div class="request-cap-grid" role="list">
        <div
          v-for="param in filteredCapabilities"
          :key="`request-${param.key}`"
          class="request-cap-item"
          role="listitem"
        >
          <code class="request-cap-item__key">{{ param.key }}</code>
          <span class="request-cap-item__label">{{ param.label }}</span>
          <i
            v-if="param.description"
            class="pi pi-info-circle param-info request-cap-item__info"
            v-tooltip.top="paramDescriptionTooltip(param)"
          />
        </div>
      </div>
      <Message
        v-if="!filteredCapabilities.length"
        severity="secondary"
        :closable="false"
        class="config-scan-message"
      >
        No request parameters match your filter.
      </Message>
    </div>
  </details>
</template>

<script setup>
import { computed, ref } from 'vue'
import { useToast } from 'primevue/usetoast'
import Button from 'primevue/button'
import InputText from 'primevue/inputtext'
import InputNumber from 'primevue/inputnumber'
import ToggleSwitch from 'primevue/toggleswitch'
import Select from 'primevue/select'
import Message from 'primevue/message'
import Textarea from 'primevue/textarea'
import { paramDescriptionTooltip } from '@/composables/useAudioModelConfig'
import { copyText } from '@/utils/clipboard'

const props = defineProps({
  title: {
    type: String,
    default: '',
  },
  endpoint: {
    type: String,
    default: '',
  },
  guidance: {
    type: String,
    default: '',
  },
  profiled: {
    type: Boolean,
    default: false,
  },
  groups: {
    type: Array,
    default: () => [],
  },
  resolveValue: {
    type: Function,
    required: true,
  },
  resolveOptions: {
    type: Function,
    default: () => [],
  },
  defaultsPreview: {
    type: Object,
    default: null,
  },
  apiExample: {
    type: String,
    default: '',
  },
  modelId: {
    type: String,
    default: '',
  },
  capabilities: {
    type: Array,
    default: () => [],
  },
})

const emit = defineEmits(['update-request-default'])

const toast = useToast()
const showSetParamsPreview = ref(false)
const capabilityFilter = ref('')

const filteredCapabilities = computed(() => {
  const query = capabilityFilter.value.trim().toLowerCase()
  if (!query) return props.capabilities
  return props.capabilities.filter((param) => {
    const haystack = [param.key, param.label, param.description].join(' ').toLowerCase()
    return haystack.includes(query)
  })
})

function fieldValue(field) {
  return props.resolveValue(field)
}

function fieldOptions(field) {
  return props.resolveOptions(field) || []
}

function updateField(field, value) {
  emit('update-request-default', { field, value })
}

function requestDefaultFieldTooltip(field) {
  const modelHints = [field?.description, field?.hint].filter(Boolean)
  if (modelHints.length) return modelHints.join('\n\n')
  return 'Saved as a request default for this model.'
}

async function copyApiExample() {
  try {
    await copyText(props.apiExample)
    toast.add({
      severity: 'success',
      summary: 'Copied',
      detail: 'API example copied to clipboard.',
      life: 2500,
    })
  } catch {
    toast.add({
      severity: 'warn',
      summary: 'Copy failed',
      detail: 'Select the text manually.',
      life: 4000,
    })
  }
}
</script>

<style scoped>
.config-details-block {
  padding-top: 0.85rem;
  padding-bottom: 0.85rem;
}

.config-details-block > .config-details-summary {
  list-style: none;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
  font-weight: 600;
  text-transform: none;
  letter-spacing: 0;
  color: var(--text-primary);
}

.config-details-block > .config-details-summary::-webkit-details-marker {
  display: none;
}

.config-details-summary__count {
  font-weight: 600;
  text-transform: none;
  letter-spacing: normal;
  opacity: 0.8;
}

.config-details-body {
  margin-top: 0.75rem;
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
}

.setparams-preview__toggle {
  width: 100%;
  border: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  color: inherit;
  cursor: pointer;
  text-align: left;
}
</style>
