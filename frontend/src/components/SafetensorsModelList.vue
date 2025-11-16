<template>
  <div class="safetensors-card">
    <div class="card-header">
      <div>
        <h2>Safetensors Models</h2>
        <p class="subtitle">Run Hugging Face safetensors via LMDeploy TurboMind</p>
      </div>
      <div class="actions">
        <Button 
          icon="pi pi-refresh" 
          @click="$emit('refresh')" 
          :loading="loading"
          severity="secondary"
          text
        />
        <Button 
          icon="pi pi-sync"
          @click="refreshStatus"
          :loading="statusLoading"
          severity="secondary"
          text
          v-tooltip.bottom="'Refresh LMDeploy status'"
        />
      </div>
    </div>

    <div v-if="loading" class="loading-state">
      <i class="pi pi-spin pi-spinner"></i>
      <span>Loading safetensors models...</span>
    </div>

    <div v-if="!lmdeployReady" class="lmdeploy-alert">
      <div class="alert-icon">
        <i class="pi pi-exclamation-triangle"></i>
      </div>
      <div class="alert-content">
        <h3>LMDeploy is not installed</h3>
        <p>
          Install LMDeploy from the new LMDeploy page before starting safetensors runtimes.
          The install happens at runtime, keeping the Docker image slim.
        </p>
        <Button label="Open LMDeploy Page" icon="pi pi-box" size="small" @click="openLmdeployPage" />
      </div>
    </div>

    <div v-if="lmdeployOperation" class="lmdeploy-alert info">
      <div class="alert-icon">
        <i class="pi pi-spin pi-spinner"></i>
      </div>
      <div class="alert-content">
        <h3>Installer running</h3>
        <p>LMDeploy installer is currently {{ lmdeployOperation }}. Runtime controls are disabled until it finishes.</p>
      </div>
    </div>

    <div v-else-if="groupedModels.length > 0" class="model-grid">
      <div 
        v-for="group in groupedModels" 
        :key="group.huggingface_id"
        class="model-card grouped-card"
      >
        <div class="model-card-header group-header">
          <div>
            <div class="model-name">{{ group.huggingface_id }}</div>
            <div class="model-path" v-if="group.metadata?.base_model">{{ group.metadata.base_model }}</div>
          </div>
          <div class="group-summary">
            <span>{{ group.file_count }} {{ group.file_count === 1 ? 'file' : 'files' }}</span>
            <span class="dot">•</span>
            <span>{{ formatSize(group.total_size) }}</span>
          </div>
        </div>

        <div class="model-body grouped-body">
          <div class="file-list">
            <div 
              v-for="file in group.files" 
              :key="file.model_id || file.filename"
              class="file-row static"
            >
              <div class="file-row-header">
                <div>
                  <div class="file-name">{{ file.filename }}</div>
                  <div class="model-meta">
                    <div class="meta-row">
                      <span class="meta-size">{{ formatSize(file.file_size) }}</span>
                      <span class="dot">•</span>
                      <span class="meta-date">{{ formatDate(file.downloaded_at) }}</span>
                    </div>
                    <div v-if="isModelRunning(file)" class="endpoint">
                      <i class="pi pi-link"></i>
                      <span>http://localhost:2001/v1/chat/completions</span>
                    </div>
                  </div>
                </div>
                <span
                  :class="[
                    'status-indicator',
                    isModelRunning(file) ? 'status-running' : 'status-stopped'
                  ]"
                >
                  <i :class="isModelRunning(file) ? 'pi pi-play' : 'pi pi-pause'"></i>
                  {{ isModelRunning(file) ? 'Running' : 'Stopped' }}
                </span>
              </div>
            </div>
          </div>

          <div class="group-actions">
            <div class="action-group">
              <Button 
                label="Configure & Run" 
                icon="pi pi-sliders-h"
                severity="secondary"
                size="small"
                :disabled="!group.files?.length"
                @click="openGroupConfig(group)"
                :loading="isGroupConfigLoading(group)"
              />
              <Button 
                v-if="group.files?.length && group.files.some(isModelRunning)"
                label="Stop" 
                icon="pi pi-stop"
                severity="danger"
                size="small"
                outlined
                :loading="isGroupStopping(group)"
                @click="stopGroupRuntime(group)"
              />
            </div>
            <Button 
              icon="pi pi-trash"
              severity="danger"
              size="small"
              outlined
              text
              :disabled="!group.files?.length"
              @click="$emit('delete', group.files[0])"
            />
          </div>
        </div>
      </div>
    </div>

    <div v-else class="empty-state">
      <i class="pi pi-shield"></i>
      <h3>No Safetensors Models</h3>
      <p>Download safetensors files from the Search tab to prepare them for LMDeploy.</p>
    </div>

    <Dialog 
      v-model:visible="dialogVisible" 
      modal 
      :style="{ width: '720px' }"
      :breakpoints="{ '960px': '90vw', '640px': '95vw' }"
    >
      <template #header>
        <div class="dialog-header">
          <div>
            <h3>{{ selectedModel?.huggingface_id || 'Configure LMDeploy' }}</h3>
            <p>{{ selectedModel?.filename }}</p>
          </div>
          <Tag 
            :severity="selectedModelRunning ? 'success' : 'warning'"
            :value="selectedModelRunning ? 'Running' : 'Stopped'"
          />
        </div>
      </template>

      <div v-if="selectedRuntime">
        <div class="config-section">
          <h4>Sequence & Parallelism</h4>
          <div class="config-grid">
            <div class="config-field span-2">
              <label>Session Length (--session-len)</label>
              <div class="slider-row">
                <Slider v-model="formState.session_len" :min="1024" :max="sessionLimit" :step="256" />
                <InputNumber v-model="formState.session_len" :min="1024" :max="sessionLimit" :step="256" inputId="sessionInput" showButtons />
              </div>
              <small>Max supported: {{ sessionLimit.toLocaleString() }} tokens</small>
            </div>
            <div class="config-field">
              <label>Max Context Tokens (--max-context-token-num)</label>
              <InputNumber v-model="formState.max_context_token_num" :min="formState.session_len" :max="256000" :step="256" showButtons />
            </div>
            <div class="config-field">
              <label>Max Prefill Tokens (--max-prefill-token-num)</label>
              <InputNumber v-model="formState.max_prefill_token_num" :min="formState.session_len" :max="256000" :step="256" showButtons />
            </div>
            <div class="config-field">
              <label>Tensor Parallel (--tp)</label>
              <InputNumber v-model="formState.tensor_parallel" :min="1" :max="8" :step="1" showButtons />
            </div>
            <div class="config-field">
              <label>Tensor Split (--tp-split)</label>
              <InputText v-model="tensorSplitString" placeholder="e.g. 30, 30, 40" />
              <small>Comma-separated percentages for GPU split</small>
            </div>
            <div class="config-field">
              <label>Max Batch Size (--max-batch-size)</label>
              <InputNumber v-model="formState.max_batch_size" :min="1" :max="128" :step="1" showButtons />
            </div>
          </div>
        </div>

        <div class="config-section">
          <h4>Precision & Backend</h4>
          <div class="config-grid">
            <div class="config-field">
              <label>DType (--dtype)</label>
              <Dropdown v-model="formState.dtype" :options="dtypeOptions" optionLabel="label" optionValue="value" />
            </div>
            <div class="config-field">
              <label>Model Format (--model-format)</label>
              <Dropdown v-model="formState.model_format" :options="modelFormatOptions" optionLabel="label" optionValue="value" placeholder="Auto-detect" />
            </div>
            <div class="config-field">
              <label>Quant Policy (--quant-policy)</label>
              <Dropdown v-model="formState.quant_policy" :options="quantPolicyOptions" optionLabel="label" optionValue="value" />
            </div>
            <div class="config-field">
              <label>Communicator (--communicator)</label>
              <Dropdown v-model="formState.communicator" :options="communicatorOptions" optionLabel="label" optionValue="value" />
            </div>
          </div>
        </div>

        <div class="config-section">
          <h4>Cache & Performance</h4>
          <div class="config-grid">
            <div class="config-field">
              <label>Cache Max Entry (--cache-max-entry-count)</label>
              <InputNumber v-model="formState.cache_max_entry_count" :min="0.1" :max="1" :step="0.05" mode="decimal" showButtons />
            </div>
            <div class="config-field">
              <label>Cache Block Seq Len (--cache-block-seq-len)</label>
              <InputNumber v-model="formState.cache_block_seq_len" :min="32" :max="2048" :step="32" showButtons />
            </div>
            <div class="config-field switch-field">
              <label>Prefix Caching (--enable-prefix-caching)</label>
              <InputSwitch v-model="formState.enable_prefix_caching" />
            </div>
            <div class="config-field">
              <label>Rope Scaling (--rope-scaling-factor)</label>
              <InputNumber v-model="formState.rope_scaling_factor" :min="0" :max="10" :step="0.1" mode="decimal" showButtons />
            </div>
            <div class="config-field">
              <label>Tokens / Iter (--num-tokens-per-iter)</label>
              <InputNumber v-model="formState.num_tokens_per_iter" :min="0" :max="262144" :step="64" showButtons />
            </div>
            <div class="config-field">
              <label>Prefill Iters (--max-prefill-iters)</label>
              <InputNumber v-model="formState.max_prefill_iters" :min="1" :max="16" :step="1" showButtons />
            </div>
            <div class="config-field switch-field">
              <label>Enable Metrics (--enable-metrics)</label>
              <InputSwitch v-model="formState.enable_metrics" />
            </div>
          </div>
        </div>

        <div class="config-section">
          <h4>Advanced</h4>
          <div class="config-grid">
            <div class="config-field span-2">
              <label>HF Overrides (--hf-overrides)</label>
              <InputText v-model="formState.hf_overrides" placeholder='{"rope_scaling": ...}' />
            </div>
            <div class="config-field span-2">
              <label>Additional CLI Args</label>
              <InputText v-model="formState.additional_args" placeholder="Custom lmdeploy flags" />
            </div>
          </div>
        </div>

        <Divider />

        <div class="metadata-section">
          <div>
            <h4>Model Metadata</h4>
            <ul>
              <li v-if="metadata.architecture"><strong>Architecture:</strong> {{ metadata.architecture }}</li>
              <li v-if="metadata.base_model"><strong>Base Model:</strong> {{ metadata.base_model }}</li>
              <li v-if="metadata.pipeline_tag"><strong>Pipeline:</strong> {{ metadata.pipeline_tag }}</li>
              <li v-if="metadata.parameters"><strong>Parameters:</strong> {{ metadata.parameters }}</li>
            </ul>
          </div>
          <div v-if="dtypeEntries.length" class="dtype-panel">
            <h4>Tensor dtypes</h4>
            <div class="dtype-tags">
              <Tag 
                v-for="item in dtypeEntries" 
                :key="item.label" 
                :value="`${item.label}: ${item.value.toLocaleString()}`"
                severity="secondary"
              />
            </div>
          </div>
        </div>
      </div>
      <div v-else class="loading-state">
        <i class="pi pi-spin pi-spinner"></i>
        <span>Loading configuration...</span>
      </div>

      <template #footer>
        <div class="dialog-footer">
          <Button 
            label="Save Config" 
            icon="pi pi-save"
            severity="secondary"
            :loading="savingConfig"
            @click="saveConfig"
          />
          <Button 
            v-if="selectedModelRunning"
            label="Stop"
            icon="pi pi-stop"
            severity="danger"
            :loading="dialogStopping"
            @click="stopRuntime()"
          />
          <Button 
            v-else
            label="Start LMDeploy"
            icon="pi pi-play"
            severity="success"
            :disabled="!lmdeployReady || !!lmdeployOperation"
            :loading="dialogStarting"
            @click="startRuntime"
          />
          <Button label="Close" text severity="secondary" @click="dialogVisible = false" />
        </div>
      </template>
    </Dialog>
  </div>
</template>

<script setup>
import { computed, ref, reactive, watch, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import Button from 'primevue/button'
import Dialog from 'primevue/dialog'
import Slider from 'primevue/slider'
import InputNumber from 'primevue/inputnumber'
import InputText from 'primevue/inputtext'
import InputSwitch from 'primevue/inputswitch'
import Dropdown from 'primevue/dropdown'
import Divider from 'primevue/divider'
import Tag from 'primevue/tag'
import { toast } from 'vue3-toastify'
import { useModelStore } from '@/stores/models'

const router = useRouter()
const props = defineProps({
  models: {
    type: Array,
    default: () => []
  },
  loading: {
    type: Boolean,
    default: false
  }
})

defineEmits(['refresh', 'delete'])

const modelStore = useModelStore()
const dialogVisible = ref(false)
const selectedModel = ref(null)
const savingConfig = ref(false)

const dtypeOptions = [
  { label: 'Auto', value: 'auto' },
  { label: 'float16', value: 'float16' },
  { label: 'bfloat16', value: 'bfloat16' }
]
const modelFormatOptions = [
  { label: 'Auto', value: '' },
  { label: 'HF', value: 'hf' },
  { label: 'AWQ', value: 'awq' },
  { label: 'GPTQ', value: 'gptq' },
  { label: 'FP8', value: 'fp8' },
  { label: 'MXFP4', value: 'mxfp4' }
]
const quantPolicyOptions = [
  { label: '0 • No kv quant', value: 0 },
  { label: '4 • 4-bit kv', value: 4 },
  { label: '8 • 8-bit kv', value: 8 }
]
const communicatorOptions = [
  { label: 'NCCL', value: 'nccl' },
  { label: 'Native', value: 'native' },
  { label: 'CUDA IPC', value: 'cuda-ipc' }
]

const formState = reactive({
  session_len: 4096,
  max_context_token_num: 4096,
  max_prefill_token_num: 8192,
  tensor_parallel: 1,
  tensor_split: [],
  max_batch_size: 4,
  dtype: 'auto',
  cache_max_entry_count: 0.8,
  cache_block_seq_len: 64,
  enable_prefix_caching: false,
  quant_policy: 0,
  model_format: '',
  hf_overrides: '',
  enable_metrics: false,
  rope_scaling_factor: 0,
  num_tokens_per_iter: 0,
  max_prefill_iters: 1,
  communicator: 'nccl',
  additional_args: ''
})

const getEntryModelId = (entry) => entry?.model_id ?? entry?.modelId ?? entry?.id

const groupedModels = computed(() => {
  if (!Array.isArray(props.models)) return []
  return [...props.models].sort((a, b) => {
    const aDate = new Date(a.latest_downloaded_at || 0).getTime()
    const bDate = new Date(b.latest_downloaded_at || 0).getTime()
    return bDate - aDate
  })
})

const statusLoading = computed(() => modelStore.lmdeployStatusLoading)

const currentInstanceId = computed(() => modelStore.lmdeployStatus?.running_instance?.model_id)
const installerStatus = computed(() => modelStore.lmdeployStatus?.installer || null)
const lmdeployReady = computed(() => !!installerStatus.value?.installed)
const lmdeployOperation = computed(() => installerStatus.value?.operation || null)
const selectedModelId = computed(() => getEntryModelId(selectedModel.value))
const selectedRuntime = computed(() => {
  const id = selectedModelId.value
  if (!id) return null
  return modelStore.safetensorsRuntime[id]
})
const metadata = computed(() => selectedRuntime.value?.metadata || {})
const sessionLimit = computed(() => selectedRuntime.value?.max_context_length || 256000)
const dtypeEntries = computed(() => {
  const summary = selectedRuntime.value?.tensor_summary?.dtype_counts || {}
  return Object.entries(summary).map(([label, value]) => ({ label, value }))
})
const selectedModelRunning = computed(() => {
  if (!selectedModelId.value) return false
  return currentInstanceId.value === selectedModelId.value
})

watch(() => formState.session_len, (value) => {
  if (formState.max_context_token_num < value) {
    formState.max_context_token_num = value
  }
  if (formState.max_prefill_token_num < value) {
    formState.max_prefill_token_num = value
  }
})

const tensorSplitString = computed({
  get() {
    return (formState.tensor_split || []).join(', ')
  },
  set(value) {
    if (!value) {
      formState.tensor_split = []
      return
    }
    formState.tensor_split = value
      .split(',')
      .map(part => part.trim())
      .filter(Boolean)
      .map(Number)
      .filter(num => !Number.isNaN(num))
  }
})

const isConfigLoading = (entry) => {
  const id = getEntryModelId(entry)
  return !!modelStore.safetensorsRuntimeLoading[id]
}

const isGroupConfigLoading = (group) => {
  if (!group?.files?.length) return false
  return group.files.some(file => isConfigLoading(file))
}

const isStopping = (entry) => {
  const id = getEntryModelId(entry)
  return !!modelStore.lmdeployStopping[id]
}

const isGroupStopping = (group) => {
  if (!group?.files?.length) return false
  return group.files.some(file => isStopping(file))
}

const openGroupConfig = (group) => {
  if (!group?.files?.length) return
  openConfig(group.files[0])
}

const stopGroupRuntime = (group) => {
  if (!group?.files?.length) return
  const runningFile = group.files.find(isModelRunning)
  stopRuntime(runningFile || group.files[0])
}

const dialogStarting = computed(() => {
  if (!selectedModelId.value) return false
  return !!modelStore.lmdeployStarting[selectedModelId.value]
})
const dialogStopping = computed(() => {
  if (!selectedModelId.value) return false
  return !!modelStore.lmdeployStopping[selectedModelId.value]
})

const refreshStatus = async () => {
  try {
    await modelStore.fetchLmdeployStatus()
  } catch (error) {
    console.error(error)
  }
}

const openConfig = async (model) => {
  selectedModel.value = model
  dialogVisible.value = true
  const modelId = getEntryModelId(model)
  try {
    await modelStore.fetchSafetensorsRuntimeConfig(modelId)
  } catch (error) {
    toast.error('Failed to load LMDeploy config')
  }
}

const applyRuntimeConfig = (config) => {
  if (!config) return
  const normalized = { ...config }
  if (normalized.context_length && normalized.session_len === undefined) {
    normalized.session_len = normalized.context_length
  }
  if (normalized.max_batch_tokens && normalized.max_prefill_token_num === undefined) {
    normalized.max_prefill_token_num = normalized.max_batch_tokens
  }
  Object.keys(formState).forEach((key) => {
    if (normalized[key] !== undefined) {
      formState[key] = Array.isArray(normalized[key])
        ? [...normalized[key]]
        : normalized[key]
    }
  })
}

watch(selectedRuntime, (runtime) => {
  if (runtime?.config) {
    applyRuntimeConfig(runtime.config)
  }
}, { immediate: true })

const buildPayload = () => ({
  session_len: formState.session_len,
  max_context_token_num: formState.max_context_token_num,
  max_prefill_token_num: formState.max_prefill_token_num,
  tensor_parallel: formState.tensor_parallel,
  tensor_split: formState.tensor_split || [],
  max_batch_size: formState.max_batch_size,
  dtype: formState.dtype,
  cache_max_entry_count: formState.cache_max_entry_count,
  cache_block_seq_len: formState.cache_block_seq_len,
  enable_prefix_caching: formState.enable_prefix_caching,
  quant_policy: formState.quant_policy,
  model_format: formState.model_format,
  hf_overrides: formState.hf_overrides,
  enable_metrics: formState.enable_metrics,
  rope_scaling_factor: formState.rope_scaling_factor,
  num_tokens_per_iter: formState.num_tokens_per_iter,
  max_prefill_iters: formState.max_prefill_iters,
  communicator: formState.communicator,
  additional_args: formState.additional_args
})

const saveConfig = async () => {
  if (!selectedModelId.value) return
  savingConfig.value = true
  try {
    await modelStore.updateSafetensorsRuntimeConfig(selectedModelId.value, buildPayload())
    toast.success('LMDeploy config saved')
  } catch (error) {
    toast.error('Failed to save config')
  } finally {
    savingConfig.value = false
  }
}

const startRuntime = async () => {
  if (!selectedModelId.value) return
  if (!lmdeployReady.value) {
    toast.error('Install LMDeploy before starting a runtime')
    return
  }
  if (lmdeployOperation.value) {
    toast.info('LMDeploy installer is running—wait until it finishes')
    return
  }
  try {
    await modelStore.startSafetensorsRuntime(selectedModelId.value, buildPayload())
    toast.success('LMDeploy starting…')
  } catch (error) {
    toast.error('Failed to start LMDeploy')
  }
}

const stopRuntime = async (entry = null) => {
  const targetId = entry ? getEntryModelId(entry) : selectedModelId.value
  if (!targetId) return
  try {
    await modelStore.stopSafetensorsRuntime(targetId)
    toast.success('LMDeploy stopped')
  } catch (error) {
    toast.error('Failed to stop LMDeploy')
  }
}

const isModelRunning = (model) => {
  const modelId = getEntryModelId(model)
  return currentInstanceId.value === modelId
}

const formatSize = (bytes) => {
  if (bytes === null || bytes === undefined) return 'Unknown size'
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`
}

const formatDate = (dateString) => {
  if (!dateString) return 'Unknown date'
  const date = new Date(dateString)
  return date.toLocaleString()
}

const openLmdeployPage = () => {
  router.push('/lmdeploy')
}

onMounted(() => {
  refreshStatus()
})
</script>

<style scoped>
.lmdeploy-alert {
  display: flex;
  gap: var(--spacing-md);
  padding: var(--spacing-md);
  background: rgba(255, 193, 7, 0.1);
  border: 1px solid rgba(255, 193, 7, 0.3);
  border-radius: var(--radius-md);
  margin-bottom: var(--spacing-lg);
  align-items: center;
}

.lmdeploy-alert.info {
  background: rgba(59, 130, 246, 0.1);
  border-color: rgba(59, 130, 246, 0.3);
}

.alert-icon {
  font-size: 1.5rem;
  color: var(--status-warning);
}

.alert-content h3 {
  margin: 0 0 var(--spacing-xs) 0;
}

.alert-content p {
  margin: 0 0 var(--spacing-sm) 0;
  color: var(--text-secondary);
}

.safetensors-card {
  margin-top: var(--spacing-xl);
  padding: var(--spacing-lg);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  background: var(--bg-card);
  box-shadow: var(--shadow-sm);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: var(--spacing-md);
  gap: var(--spacing-md);
}

.card-header h2 {
  margin: 0;
  font-weight: 600;
  color: var(--text-primary);
}

.subtitle {
  margin: 4px 0 0;
  font-size: 0.9rem;
  color: var(--text-secondary);
}

.actions {
  display: flex;
  gap: var(--spacing-xs);
}

.loading-state,
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-xl);
  color: var(--text-secondary);
  text-align: center;
}

.model-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: var(--spacing-md);
}

.model-card {
  border: 1px solid var(--border-secondary);
  border-radius: var(--radius-xl);
  padding: var(--spacing-md);
  background: var(--bg-surface);
  box-shadow: var(--shadow-sm);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.grouped-card {
  gap: var(--spacing-md);
}

.model-card-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: var(--spacing-sm);
}

.model-name {
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 2px;
}

.model-path {
  font-size: 0.9rem;
  color: var(--text-secondary);
}

.group-header {
  border-bottom: 1px solid var(--border-secondary);
  padding-bottom: var(--spacing-sm);
}

.group-summary {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.grouped-body {
  padding-top: var(--spacing-sm);
}

.file-list {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.file-row {
  border: 1px solid var(--border-secondary);
  border-radius: var(--radius-lg);
  padding: var(--spacing-sm);
  background: var(--bg-surface-2, var(--bg-card));
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  cursor: default;
}

.file-row-header {
  display: flex;
  justify-content: space-between;
  gap: var(--spacing-sm);
}

.file-name {
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 4px;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: var(--spacing-xxs);
  padding: 4px 8px;
  border-radius: var(--radius-sm);
  font-size: 0.75rem;
  font-weight: 500;
}

.status-running {
  background: rgba(16, 185, 129, 0.12);
  color: var(--accent-green);
  border: 1px solid rgba(16, 185, 129, 0.2);
}

.status-stopped {
  background: rgba(245, 158, 11, 0.12);
  color: #f97316;
  border: 1px solid rgba(245, 158, 11, 0.2);
}

.model-meta {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.meta-row {
  display: flex;
  align-items: center;
  gap: 6px;
}

.endpoint {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--accent-cyan);
  font-size: 0.85rem;
}

.model-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--spacing-sm);
  margin-top: var(--spacing-sm);
}

.group-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: var(--spacing-md);
  gap: var(--spacing-sm);
  flex-wrap: wrap;
}

.config-section {
  border: 1px solid var(--border-secondary);
  border-radius: var(--radius-xl);
  padding: var(--spacing-md);
  background: var(--bg-surface);
  margin-bottom: var(--spacing-lg);
}

.config-section h4 {
  margin: 0 0 var(--spacing-sm);
  font-weight: 600;
  color: var(--text-primary);
}

.action-group {
  display: flex;
  gap: var(--spacing-xs);
}

.dot {
  opacity: 0.6;
}

.dialog-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  gap: var(--spacing-md);
}

.dialog-header h3 {
  margin: 0;
  font-weight: 600;
}

.dialog-header p {
  margin: 2px 0 0;
  color: var(--text-secondary);
  font-size: 0.9rem;
}

.config-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: var(--spacing-md);
}

.config-field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.config-field label {
  font-size: 0.9rem;
  font-weight: 600;
}

.slider-row {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.slider-row :deep(.p-slider) {
  flex: 1;
  min-width: 0;
}

.slider-row :deep(.p-slider .p-slider-range) {
  background: var(--accent-cyan);
}

.slider-row :deep(.p-slider .p-slider-handle) {
  border-color: var(--accent-cyan);
}

.slider-row :deep(.p-inputnumber) {
  width: 140px;
}

.switch-field {
  align-items: center;
  flex-direction: row;
  justify-content: space-between;
}

.span-2 {
  grid-column: span 2;
}

@media (max-width: 640px) {
  .span-2 {
    grid-column: span 1;
  }
}

.metadata-section {
  display: flex;
  flex-wrap: wrap;
  gap: var(--spacing-xl);
  margin-top: var(--spacing-lg);
}

.metadata-section ul {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 0.9rem;
}

.dtype-panel {
  min-width: 220px;
}

.dtype-tags {
  display: flex;
  flex-wrap: wrap;
  gap: var(--spacing-xs);
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: var(--spacing-sm);
}
</style>
