<template>
  <div class="model-config">
    <div class="config-layout">
      <!-- Main Configuration Panel -->
      <div class="config-main">
        <div class="card">
          <div class="card-header">
            <div class="config-header">
              <ModelInfoSection
                :model="model"
                :model-layer-info="modelLayerInfo"
                :has-hf-metadata="hasHfMetadata"
                :hf-metadata="hfMetadata"
                :hf-context-value="hfContextValue"
                :hf-defaults-list="hfDefaultsList"
                :hf-layer-info-list="hfLayerInfoList"
                :hf-metadata-loading="hfMetadataLoading"
                :regenerating-info="regeneratingInfo"
                :save-loading="saveLoading"
              >
                <template #actions>
                  <Button 
                    label="Quick Start" 
                    icon="pi pi-bolt" 
                    size="small"
                    severity="info"
                    outlined
                    @click="showQuickStartModal = true"
                    v-tooltip="'Choose a preset, use the wizard, or let Smart Auto optimize for you'"
                  />
                  <Button icon="pi pi-refresh" @click="regenerateModelInfo" :loading="regeneratingInfo"
                    severity="secondary" size="small" outlined
                    v-tooltip="'Regenerate model information from GGUF metadata'" />
                  <Button label="Save Config" icon="pi pi-save" @click="saveConfig" :loading="saveLoading"
                    severity="success" size="small" />
                </template>
              </ModelInfoSection>
            </div>
          </div>
        </div>

        <!-- Quick Start Modal -->
        <QuickStartModal
          v-model:visible="showQuickStartModal"
          :auto-config-loading="autoConfigLoading"
          :smart-auto-usage-mode="smartAutoUsageMode"
          @wizard="showWizard = true"
          @preset="applyPreset"
          @smart-auto="handleQuickStartSmartAuto"
        />

        <!-- Empty State for New Models -->
        <EmptyState
          v-if="hasNoConfig && !showWizard && !showPreview && !showConfig"
          :visible="hasNoConfig && !showConfig"
          icon="🎯"
          title="Configure Your Model"
          description="This model doesn't have a configuration yet. Start with Smart Auto, choose a preset, or configure manually."
          :show-smart-auto="true"
          :show-presets="true"
          @smart-auto="generateAutoConfig"
          @presets="handleEmptyStatePresets"
          @manual="handleEmptyStateManual"
        />

        <!-- Configuration Grid -->
        <div 
          v-if="!hasNoConfig || showWizard || showPreview || showConfig" 
          class="config-tabs-wrapper"
          @touchstart="handleTabSwipeStart"
          @touchend="handleTabSwipeEnd"
        >
          <div class="config-search-bar" :class="{ 'search-focused': searchFocused }">
            <span class="p-input-icon-left">
              <i class="pi pi-search" />
              <InputText 
                v-model="configSearchQuery" 
                placeholder="Search settings..."
                class="config-search-input"
                aria-label="Search configuration settings"
                @focus="searchFocused = true"
                @blur="searchFocused = false"
              />
            </span>
            <Button 
              v-if="configSearchQuery"
              icon="pi pi-times" 
              @click="configSearchQuery = ''" 
              size="small" 
              text 
              rounded
              severity="secondary"
              aria-label="Clear search"
              v-tooltip="'Clear search'"
            />
          </div>
          <TabView v-model:activeIndex="activeTabIndex" :scrollable="true" class="config-tabs">
          <!-- Essential Settings Tab -->
          <TabPanel header="Essential" icon="pi pi-microchip">
            <div class="tab-content">
              <EssentialSettingsSection
                :config="config"
                :max-gpu-layers="maxGpuLayers"
                :recommended-gpu-layers="recommendedGpuLayers"
                :gpu-layers-tooltip="gpuLayersTooltip"
                :gpu-layers-validation="gpuLayersValidation"
                :gpu-available="gpuAvailable"
                @update-vram-estimate="updateVramEstimate"
              />
            </div>
          </TabPanel>

          <!-- Memory & Context Tab -->
          <TabPanel header="Memory & Context" icon="pi pi-memory">
            <div class="tab-content">
              <ContextParamsSection
                :config="config"
                :max-context-size="maxContextSize"
                :max-batch-size="maxBatchSize"
                :recommended-context-size="recommendedContextSize"
                :recommended-batch-size="recommendedBatchSize"
                :context-size-validation="contextSizeValidation"
                :batch-size-validation="batchSizeValidation"
                :context-size-tooltip="contextSizeTooltip"
                :batch-size-tooltip="batchSizeTooltip"
                @update-vram-estimate="updateVramEstimate"
              />
            </div>
          </TabPanel>

          <!-- Generation Tab -->
          <TabPanel header="Generation" icon="pi pi-cog">
            <div class="tab-content">
              <GenerationParamsSection
                :config="config"
                :max-top-k="maxTopK"
                :recommended-temperature="recommendedTemperature"
                :recommended-top-k="recommendedTopK"
                :recommended-top-p="recommendedTopP"
                :is-min-p-supported="isMinPSupported"
                :is-typical-p-supported="isTypicalPSupported"
                :is-tfs-z-supported="isTfsZSupported"
                :is-presence-penalty-supported="isPresencePenaltySupported"
                :is-frequency-penalty-supported="isFrequencyPenaltySupported"
                :temperature-tooltip="temperatureTooltip"
                :top-k-tooltip="topKTooltip"
                :top-p-tooltip="topPTooltip"
                :repeat-penalty-tooltip="repeatPenaltyTooltip"
              />
              <div class="tab-section">
                <h4 class="tab-section-title">
                  <i class="pi pi-cog"></i>
                  Additional Options
                </h4>
                <ConfigField label="Seed" help-text="Random seed (-1 for random)">
                  <template #input>
                    <InputNumber v-model="config.seed" :min="-1" :max="2147483647" />
                  </template>
                </ConfigField>
                <ConfigField label="Stop Words (comma-separated)" help-text="Words that stop generation">
                  <template #input>
                    <InputText v-model="stopWordsInput" @blur="applyStopWords"
                      placeholder="e.g. \\n, \\n\\n, &lt;/s&gt;" />
                  </template>
                </ConfigField>
                <ConfigField label="Grammar" help-text="Optional grammar string">
                  <template #input>
                    <InputText v-model="config.grammar" placeholder="optional grammar" />
                  </template>
                </ConfigField>
                <ConfigField v-if="isJsonSchemaSupported" label="JSON Schema" help-text="Optional JSON schema">
                  <template #input>
                    <InputText v-model="config.json_schema" placeholder="optional JSON schema" />
                  </template>
                </ConfigField>
                <ConfigField label="Use Jinja Template" help-text="Enable Jinja templating">
                  <template #input>
                    <Checkbox v-model="config.jinja" binary />
                  </template>
                </ConfigField>
              </div>
            </div>
          </TabPanel>

          <!-- Performance Tab -->
          <TabPanel header="Performance" icon="pi pi-tachometer">
            <div class="tab-content">
              <PerformanceSection
                :config="config"
                :model-layer-info="modelLayerInfo"
                :max-parallel="maxParallel"
                :parallel-validation="parallelValidation"
                :gpu-available="gpuAvailable"
                :is-embedding-model="isEmbeddingModel"
                :is-cache-type-v-supported="isCacheTypeVSupported"
              />
            </div>
          </TabPanel>

          <!-- Advanced Tab -->
          <TabPanel header="Advanced" icon="pi pi-wrench">
            <div class="tab-content">
              <AdvancedSection :config="config" />
            </div>
          </TabPanel>

          <!-- Custom Arguments Tab -->
          <TabPanel header="Custom Arguments" icon="pi pi-code">
            <div class="tab-content">
              <CustomArgsSection :config="config" />
            </div>
          </TabPanel>
        </TabView>
      </div>
      </div>

      <!-- Configuration Warnings -->
      <ConfigWarnings :warnings="configWarnings" />
      
      <!-- Memory Status Dashboard (prominent top cards) -->
      <div class="memory-dashboard" id="memory-dashboard">
        <!-- RAM Monitor Card -->
        <MemoryMonitor
          title="System RAM"
          :current-value="realtimeRamData?.used || null"
          :estimated-value="ramEstimate?.estimated_ram || null"
          :total-capacity="totalRamBytes"
          :loading="ramLoading && !ramEstimate && !realtimeRamData"
        />

        <!-- VRAM Monitor Card (only shown when GPU is available) -->
        <MemoryMonitor
          v-if="!systemStore.gpuInfo.cpu_only_mode && systemStore.gpuInfo.device_count > 0"
          title="VRAM"
          :current-value="currentVramBytes"
          :estimated-value="vramEstimate?.estimated_vram || null"
          :total-capacity="totalVramBytes"
          :loading="vramLoading && !vramEstimate && !realtimeVramData"
        />
      </div>

    </div>

    <!-- Configuration Wizard -->
    <ConfigWizard
      :model-visible="showWizard"
      :model="model"
      :gpu-info="systemStore.gpuInfo"
      :model-layer-info="modelLayerInfo"
      @close="showWizard = false"
      @apply-config="handleWizardConfig"
      @go-to-advanced="showWizard = false"
    />

    <!-- Configuration Change Preview -->
    <ConfigChangePreview
      :visible="showPreview"
      @update:visible="showPreview = $event"
      :type="previewData?.type || 'smart-auto'"
      :preset-name="previewData?.presetName || ''"
      :changes="previewData?.changes || []"
      :impact="previewData?.impact"
      :applying="previewApplying"
      @apply="applyPreviewChanges"
      @cancel="showPreview = false"
    />

    <!-- Onboarding Tour -->
    <OnboardingTour
      :visible="showOnboarding"
      :steps="onboardingSteps"
      @update:visible="showOnboarding = $event"
      @complete="handleOnboardingComplete"
      @skip="handleOnboardingSkip"
    />
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useModelStore } from '@/stores/models'
import { useSystemStore } from '@/stores/system'
import { useWebSocketStore } from '@/stores/websocket'
import { toast } from 'vue3-toastify'
import Button from 'primevue/button'
import InputNumber from 'primevue/inputnumber'
import InputText from 'primevue/inputtext'
import Textarea from 'primevue/textarea'
import Checkbox from 'primevue/checkbox'
import Dropdown from 'primevue/dropdown'
import TabView from 'primevue/tabview'
import TabPanel from 'primevue/tabpanel'
import SliderInput from '@/components/SliderInput.vue'
import MemoryMonitor from '@/components/config/MemoryMonitor.vue'
import ConfigField from '@/components/config/ConfigField.vue'
import ConfigWizard from '@/components/config/ConfigWizard.vue'
import ConfigChangePreview from '@/components/config/ConfigChangePreview.vue'
import OnboardingTour from '@/components/config/OnboardingTour.vue'
import EmptyState from '@/components/config/EmptyState.vue'
import ModelInfoSection from '@/components/config/ModelInfoSection.vue'
import HfDefaultsCard from '@/components/config/HfDefaultsCard.vue'
import QuickStartModal from '@/components/config/QuickStartModal.vue'
import GenerationParamsSection from '@/components/config/GenerationParamsSection.vue'
import ContextParamsSection from '@/components/config/ContextParamsSection.vue'
import MemoryParamsSection from '@/components/config/MemoryParamsSection.vue'
import AdvancedSettingsSection from '@/components/config/AdvancedSettingsSection.vue'
import PerformanceSection from '@/components/config/PerformanceSection.vue'
import AdvancedSection from '@/components/config/AdvancedSection.vue'
import CustomArgsSection from '@/components/config/CustomArgsSection.vue'
import EssentialSettingsSection from '@/components/config/EssentialSettingsSection.vue'
import ConfigWarnings from '@/components/config/ConfigWarnings.vue'

// Utils
import { formatFileSize } from '@/utils/formatting'

const route = useRoute()
const router = useRouter()
const modelStore = useModelStore()
const systemStore = useSystemStore()
const wsStore = useWebSocketStore()

// Reactive state
const model = ref(null)
const config = ref({})
const showConfig = ref(false)
const EMBEDDING_PIPELINE_TAGS = ['text-embedding', 'feature-extraction', 'sentence-similarity']
const EMBEDDING_KEYWORDS = ['embedding', 'embed', 'nomic', 'gte', 'e5', 'bge', 'minilm']
const isEmbeddingModel = computed(() => {
  const current = model.value
  if (!current) return false
  if (current.is_embedding_model) {
    return true
  }
  const pipeline = (current.pipeline_tag || '').toLowerCase()
  if (EMBEDDING_PIPELINE_TAGS.includes(pipeline)) {
    return true
  }
  const combined = [
    current.huggingface_id,
    current.base_model_name,
    current.name
  ].filter(Boolean).join(' ').toLowerCase()
  return EMBEDDING_KEYWORDS.some(keyword => combined.includes(keyword))
})
const stopWordsInput = ref('')
const applyStopWords = () => {
  const parts = (stopWordsInput.value || '').split(',').map(s => s.trim()).filter(Boolean)
  config.value.stop = parts
}
const vramEstimate = ref(null)
const vramLoading = ref(false)
const ramEstimate = ref(null)
const ramLoading = ref(false)
let vramEstimateTimeout = null
let ramEstimateTimeout = null
let vramAbortController = null
let ramAbortController = null
const ESTIMATE_DEBOUNCE_MS = 400
const autoConfigLoading = ref(false)
const saveLoading = ref(false)
const modelLayerInfo = ref(null)
const modelRecommendations = ref(null)
const hfMetadata = ref(null)
const hfDefaults = ref(null)
const hfLayerInfo = ref(null)
const hfMetadataLoading = ref(false)
const regeneratingInfo = ref(false)
const layerInfoLoading = ref(false)
const recommendationsLoading = ref(false)
const smartAutoUsageMode = ref('single_user')
const configSearchQuery = ref('')
const searchFocused = ref(false)
const showWizard = ref(false)
const showQuickStartModal = ref(false)

// Handlers for empty state actions
const handleEmptyStatePresets = () => {
  showConfig.value = true
  activeTabIndex.value = 0
}

const handleEmptyStateManual = () => {
  showConfig.value = true
  // Keep activeTabIndex on whatever tab the user last interacted with (default 0)
}

watch(isEmbeddingModel, (value) => {
  if (value && !config.value.embedding) {
    config.value.embedding = true
  }
})

// Real-time memory data from WebSocket
const realtimeRamData = ref(null)
const realtimeVramData = ref(null)

// Performance metrics from unified monitoring

// Active tab index for tabbed interface
const activeTabIndex = ref(0)

// Tab labels for search matching
const tabLabels = [
  { key: 'essential', label: 'Essential', keywords: ['gpu', 'layer', 'thread', 'cpu', 'loading', 'model'] },
  { key: 'memory', label: 'Memory & Context', keywords: ['context', 'memory', 'batch', 'size', 'mmap', 'mlock', 'ram'] },
  { key: 'generation', label: 'Generation', keywords: ['temperature', 'temp', 'top-k', 'top-p', 'repeat', 'penalty', 'sampling', 'token', 'generate', 'mirostat', 'seed', 'grammar', 'jinja'] },
  { key: 'performance', label: 'Performance', keywords: ['parallel', 'flash', 'attention', 'vram', 'low', 'batching', 'offload', 'logits', 'embedding', 'cache', 'kv', 'quantization', 'moe', 'expert'] },
  { key: 'advanced', label: 'Advanced', keywords: ['rope', 'yarn', 'freq', 'scale', 'scaling', 'yaml'] },
  { key: 'custom', label: 'Custom Arguments', keywords: ['custom', 'argument', 'args', 'command', 'line'] }
]

// Swipe between tabs functionality
const tabSwipeStartX = ref(0)
const tabSwipeThreshold = 100

const handleTabSwipeStart = (e) => {
  if (!e.touches || e.touches.length === 0) return
  tabSwipeStartX.value = e.touches[0].clientX
}

const handleTabSwipeEnd = (e) => {
  if (!e.changedTouches || e.changedTouches.length === 0) return
  
  const deltaX = e.changedTouches[0].clientX - tabSwipeStartX.value
  
  // Swipe left (next tab) or right (previous tab)
  if (Math.abs(deltaX) > tabSwipeThreshold) {
    if (deltaX < 0 && activeTabIndex.value < tabLabels.length - 1) {
      // Swipe left - next tab
      activeTabIndex.value++
    } else if (deltaX > 0 && activeTabIndex.value > 0) {
      // Swipe right - previous tab
      activeTabIndex.value--
    }
  }
  
  tabSwipeStartX.value = 0
}

// Supported flags from llama-server
const supportedFlags = ref(null)
const supportedConfigKeys = ref({})

// GPU availability
const gpuAvailable = computed(() => {
  const dc = systemStore.gpuInfo?.device_count || 0
  return dc > 0
})

watch(gpuAvailable, (avail) => {
  if (!avail) {
    if (config.value && config.value.n_gpu_layers !== 0) {
      config.value.n_gpu_layers = 0
      updateVramEstimate(true)
    }
  }
})

onMounted(() => {
  if (!gpuAvailable.value) {
    if (config.value && config.value.n_gpu_layers !== 0) {
      config.value.n_gpu_layers = 0
      updateVramEstimate(true)
    }
  }
})

// VRAM usage percentage
const vramUsagePercentage = computed(() => {
  if (!vramEstimate.value || !systemStore.gpuInfo.total_vram) return 0
  return Math.round((vramEstimate.value.estimated_vram / systemStore.gpuInfo.total_vram) * 100)
})

// RAM usage percentage
const ramUsagePercentage = computed(() => {
  if (!ramEstimate.value || !systemStore.systemStatus.system?.memory?.total) return 0
  return Math.round((ramEstimate.value.estimated_ram / systemStore.systemStatus.system.memory.total) * 100)
})

// Only show RAM warning when we have valid totals and estimate exceeds available
const showRamWarning = computed(() => {
  const est = ramEstimate.value
  if (!est) return false
  const total = est.system_ram_total || systemStore.systemStatus.system?.memory?.total || 0
  const estimated = est.estimated_ram || 0
  return total > 0 && estimated > total
})

// Stacked progress computed values for RAM (current + estimated additional)
const totalRamBytes = computed(() => ramEstimate.value?.system_ram_total || systemStore.systemStatus.system?.memory?.total || 0)
const currentRamBytes = computed(() => ramEstimate.value?.system_ram_used || systemStore.systemStatus.system?.memory?.used || 0)
const estimatedRamBytes = computed(() => ramEstimate.value?.estimated_ram || 0)
const totalEstimatedRamBytes = computed(() => {
  // Total estimated RAM = current system usage + estimated additional RAM
  return currentRamBytes.value + estimatedRamBytes.value
})
const currentRamPercent = computed(() => {
  const total = totalRamBytes.value || 1
  return Math.min(100, Math.max(0, Math.round((currentRamBytes.value / total) * 100)))
})
const additionalRamPercent = computed(() => {
  const total = totalRamBytes.value || 1
  const add = ramEstimate.value?.estimated_ram || 0
  // Cap so current + additional does not exceed 100
  const pct = Math.round((add / total) * 100)
  return Math.max(0, Math.min(100 - currentRamPercent.value, pct))
})

// Stacked progress computed values for VRAM (current + estimated additional)
const totalVramBytes = computed(() => {
  if (realtimeVramData.value) {
    if (typeof realtimeVramData.value.total === 'number') {
      return realtimeVramData.value.total
    }
    if (typeof realtimeVramData.value.total_vram === 'number') {
      return realtimeVramData.value.total_vram
    }
  }
  return systemStore.gpuInfo.total_vram || 0
})
const currentVramBytes = computed(() => {
  if (realtimeVramData.value && typeof realtimeVramData.value.used_vram === 'number') {
    return realtimeVramData.value.used_vram
  }
  const total = systemStore.gpuInfo.total_vram || 0
  const available = systemStore.gpuInfo.available_vram
  if (total && typeof available === 'number') {
    return Math.max(0, total - available)
  }
  return 0
})
const estimatedVramBytes = computed(() => vramEstimate.value?.estimated_vram || 0)
const totalEstimatedVramBytes = computed(() => {
  // Total estimated VRAM = current GPU usage + estimated additional VRAM
  return currentVramBytes.value + estimatedVramBytes.value
})
const currentVramPercent = computed(() => {
  const total = totalVramBytes.value || 1
  return Math.min(100, Math.max(0, Math.round(((currentVramBytes.value || 0) / total) * 100)))
})
const additionalVramPercent = computed(() => {
  const total = totalVramBytes.value || 1
  const add = vramEstimate.value?.estimated_vram || 0
  const pct = Math.round((add / total) * 100)
  return Math.max(0, Math.min(100 - currentVramPercent.value, pct))
})

// RAM Status indicators for new dashboard
const ramStatus = computed(() => {
  if (!ramEstimate.value && !realtimeRamData.value) return 'unknown'
  
  let usagePercent = 0
  if (ramEstimate.value?.system_ram_total) {
    usagePercent = ((totalEstimatedRamBytes.value / ramEstimate.value.system_ram_total) * 100)
  } else if (realtimeRamData.value?.percent) {
    usagePercent = realtimeRamData.value.percent
  }
  
  if (usagePercent < 70) return 'good'
  if (usagePercent < 90) return 'warning'
  return 'critical'
})

const ramStatusClass = computed(() => {
  const status = ramStatus.value
  if (status === 'good') return 'status-good'
  if (status === 'warning') return 'status-warning'
  if (status === 'critical') return 'status-critical'
  return 'status-unknown'
})

const ramStatusText = computed(() => {
  const status = ramStatus.value
  if (status === 'good') return 'Fits Comfortably'
  if (status === 'warning') return 'Tight Fit'
  if (status === 'critical') return 'Won\'t Fit'
  return 'Unknown'
})

const hfContextValue = computed(() => {
  if (hfDefaults.value?.ctx_size) return hfDefaults.value.ctx_size
  if (hfDefaults.value?.context_length) return hfDefaults.value.context_length
  if (hfMetadata.value?.max_context_length) return hfMetadata.value.max_context_length
  if (hfMetadata.value?.context_length) return hfMetadata.value.context_length
  return null
})

const hfDefaultsList = computed(() => {
  if (!hfDefaults.value) return []
  const mapping = [
    { label: 'Context Size', value: hfContextValue.value },
    { label: 'Temperature', value: hfDefaults.value.temperature ?? hfDefaults.value.temp },
    { label: 'Top P', value: hfDefaults.value.top_p },
    { label: 'Top K', value: hfDefaults.value.top_k },
    { label: 'Typical P', value: hfDefaults.value.typical_p },
    { label: 'Min P', value: hfDefaults.value.min_p },
    { label: 'Repeat Penalty', value: hfDefaults.value.repeat_penalty },
    { label: 'Presence Penalty', value: hfDefaults.value.presence_penalty },
    { label: 'Frequency Penalty', value: hfDefaults.value.frequency_penalty },
    { label: 'Seed', value: hfDefaults.value.seed }
  ]
  return mapping.filter(item => item.value !== undefined && item.value !== null)
})

const hasHfMetadata = computed(() => {
  return (
    hfMetadata.value ||
    hfLayerInfo.value ||
    hfDefaultsList.value.length > 0 ||
    (!!hfMetadataLoading.value)
  )
})

const hfLayerInfoList = computed(() => {
  if (!hfLayerInfo.value) return []
  const info = hfLayerInfo.value
  const items = [
    { label: 'Embedding Dim', value: info.embedding_length },
    { label: 'Architecture', value: info.architecture },
    { label: 'Layer Count', value: info.layer_count },
    { label: 'Context (GGUF)', value: info.context_length },
    { label: 'Attention Heads', value: info.attention_head_count },
    { label: 'KV Heads', value: info.attention_head_count_kv },
    { label: 'MoE Experts', value: info.expert_count },
    { label: 'Experts Used', value: info.experts_used_count }
  ]
  return items.filter(item => item.value !== undefined && item.value !== null && item.value !== 0)
})

const ramProgressText = computed(() => {
  const current = currentRamPercent.value
  const additional = additionalRamPercent.value
  return `${current}% used + ${additional}% est • ${formatFileSize(totalEstimatedRamBytes.value)} total`
})

const ramStatusMessage = computed(() => {
  const status = ramStatus.value
  if (status === 'good') {
    const available = totalRamBytes.value - totalEstimatedRamBytes.value
    return `✅ Fits Comfortably - ${formatFileSize(available)} buffer remaining`
  }
  if (status === 'warning') {
    return '⚠️ Memory usage is high - consider reducing context size or batch size'
  }
  if (status === 'critical') {
    return '❌ Usage exceeds available memory - configuration will not work'
  }
  return 'Loading memory information...'
})

// VRAM Status indicators for new dashboard
const vramStatus = computed(() => {
  if (!vramEstimate.value && !realtimeVramData.value) return 'unknown'
  
  const usagePercent = totalVramBytes.value > 0 
    ? (totalEstimatedVramBytes.value / totalVramBytes.value) * 100
    : 0
  
  if (usagePercent < 70) return 'good'
  if (usagePercent < 90) return 'warning'
  return 'critical'
})

const vramStatusClass = computed(() => {
  const status = vramStatus.value
  if (status === 'good') return 'status-good'
  if (status === 'warning') return 'status-warning'
  if (status === 'critical') return 'status-critical'
  return 'status-unknown'
})

const vramStatusText = computed(() => {
  const status = vramStatus.value
  if (status === 'good') return 'Fits Comfortably'
  if (status === 'warning') return 'Tight Fit'
  if (status === 'critical') return 'Won\'t Fit'
  return 'Unknown'
})

const vramProgressText = computed(() => {
  const current = currentVramPercent.value
  const additional = additionalVramPercent.value
  return `${current}% used + ${additional}% est • ${formatFileSize(totalEstimatedVramBytes.value)} total`
})

const vramStatusMessage = computed(() => {
  const status = vramStatus.value
  if (status === 'good') {
    const available = totalVramBytes.value - totalEstimatedVramBytes.value
    return `✅ Fits Comfortably - ${formatFileSize(available)} VRAM available`
  }
  if (status === 'warning') {
    return '⚠️ VRAM usage is high - consider reducing GPU layers or batch size'
  }
  if (status === 'critical') {
    return '❌ Usage exceeds available VRAM - reduce GPU layers or context size'
  }
  return 'Loading VRAM information...'
})

// Maximum values from backend recommendations
const maxGpuLayers = computed(() => {
  return modelRecommendations.value?.gpu_layers?.max || modelLayerInfo.value?.layer_count || 32
})

const maxContextSize = computed(() => {
  return modelRecommendations.value?.context_size?.max || modelLayerInfo.value?.context_length || 131072
})

const maxBatchSize = computed(() => {
  return modelRecommendations.value?.batch_size?.max || 512
})

// Maximum parallel processing based on attention head count
const maxParallel = computed(() => {
  return modelRecommendations.value?.parallel?.max || 8
})

const maxTopK = computed(() => {
  return modelRecommendations.value?.top_k?.max || 200
})

// Recommended values from backend
const recommendedGpuLayers = computed(() => modelRecommendations.value?.gpu_layers?.recommended_value)
const recommendedContextSize = computed(() => modelRecommendations.value?.context_size?.recommended_value)
const recommendedBatchSize = computed(() => modelRecommendations.value?.batch_size?.recommended_value)
const recommendedTemperature = computed(() => modelRecommendations.value?.temperature?.recommended_value)
const recommendedTopK = computed(() => modelRecommendations.value?.top_k?.recommended_value)
const recommendedTopP = computed(() => modelRecommendations.value?.top_p?.recommended_value)

// Load supported flags from the active llama-server binary
const loadSupportedFlags = async () => {
  try {
    const response = await fetch('/api/models/supported-flags')
    if (response.ok) {
      const data = await response.json()
      supportedFlags.value = data.supported_flags || []
      supportedConfigKeys.value = data.supported_config_keys || {}
    } else {
      console.warn('Failed to load supported flags:', response.statusText)
      supportedFlags.value = []
      supportedConfigKeys.value = {}
    }
  } catch (error) {
    console.error('Error loading supported flags:', error)
    supportedFlags.value = []
    supportedConfigKeys.value = {}
  }
}

// Computed properties to check if specific flags are supported
const isTypicalPSupported = computed(() => supportedConfigKeys.value.typical_p === true)
const isMinPSupported = computed(() => supportedConfigKeys.value.min_p === true)
const isTfsZSupported = computed(() => supportedConfigKeys.value.tfs_z === true)
const isPresencePenaltySupported = computed(() => supportedConfigKeys.value.presence_penalty === true)
const isFrequencyPenaltySupported = computed(() => supportedConfigKeys.value.frequency_penalty === true)
const isJsonSchemaSupported = computed(() => supportedConfigKeys.value.json_schema === true)
const isCacheTypeVSupported = computed(() => supportedConfigKeys.value.cache_type_v === true)

// Individual field validation
const gpuLayersValidation = computed(() => {
  if (!modelLayerInfo.value) return null
  if (config.value.n_gpu_layers > modelLayerInfo.value.layer_count) {
    return {
      type: 'error',
      message: `Exceeds model's ${modelLayerInfo.value.layer_count} layers`
    }
  }
  if (config.value.n_gpu_layers === modelLayerInfo.value.layer_count && modelLayerInfo.value.layer_count > 0) {
    return {
      type: 'success',
      message: 'Fully offloaded to GPU'
    }
  }
  return null
})

// Automatically translate context sizes that exceed the GGUF-reported
// context_length into YARN-style scaling parameters for llama.cpp. This makes
// "I want a bigger context" a single control while keeping advanced fields
// in sync with that choice.
watch(
  () => [config.value.ctx_size, modelLayerInfo.value?.context_length],
  ([ctxSize, ctxLengthRaw]) => {
    const targetCtx = Number(ctxSize) || 0
    const baseCtx = Number(ctxLengthRaw) || 0

    // If we don't have a meaningful base context, or the user is within the
    // native context window, clear auto YARN settings (when they still match
    // our previous auto values) and leave any manual tuning untouched.
    if (!baseCtx || targetCtx <= baseCtx) {
      if (config.value.rope_scaling === autoYarnConfig.value.rope_scaling) {
        config.value.rope_scaling = ''
      }
      if (config.value.yarn_ext_factor === autoYarnConfig.value.yarn_ext_factor) {
        config.value.yarn_ext_factor = null
      }
      if (config.value.yarn_attn_factor === autoYarnConfig.value.yarn_attn_factor) {
        config.value.yarn_attn_factor = null
      }
      autoYarnConfig.value = {
        rope_scaling: '',
        yarn_ext_factor: null,
        yarn_attn_factor: null
      }
      return
    }

    const rawFactor = targetCtx / baseCtx
    const factor = rawFactor > 1 ? Number(rawFactor.toFixed(2)) : 1.0

    const nextAuto = {
      rope_scaling: 'yarn',
      yarn_ext_factor: factor,
      // Keep attention factor neutral by default; users can override if needed.
      yarn_attn_factor: 1.0
    }

    // Only auto-populate fields when they are empty or still equal to the
    // last auto-generated values, so manual edits win.
    if (!config.value.rope_scaling || config.value.rope_scaling === autoYarnConfig.value.rope_scaling) {
      config.value.rope_scaling = nextAuto.rope_scaling
    }
    if (
      config.value.yarn_ext_factor == null ||
      config.value.yarn_ext_factor === autoYarnConfig.value.yarn_ext_factor
    ) {
      config.value.yarn_ext_factor = nextAuto.yarn_ext_factor
    }
    if (
      config.value.yarn_attn_factor == null ||
      config.value.yarn_attn_factor === autoYarnConfig.value.yarn_attn_factor
    ) {
      config.value.yarn_attn_factor = nextAuto.yarn_attn_factor
    }

    autoYarnConfig.value = nextAuto
  }
)

const contextSizeValidation = computed(() => {
  if (!modelLayerInfo.value) return null
  if (config.value.ctx_size > modelLayerInfo.value.context_length) {
    return {
      type: 'warning',
      message: `Exceeds model's max of ${modelLayerInfo.value.context_length.toLocaleString()}`
    }
  }
  return null
})

const batchSizeValidation = computed(() => {
  if (!modelLayerInfo.value) return null
  if (config.value.batch_size > maxBatchSize.value) {
    return {
      type: 'warning',
      message: `Exceeds recommended max of ${maxBatchSize.value}`
    }
  }
  return null
})

const parallelValidation = computed(() => {
  if (!modelLayerInfo.value) return null
  if (config.value.parallel > maxParallel.value) {
    return {
      type: 'warning',
      message: `Exceeds recommended max of ${maxParallel.value}`
    }
  }
  return null
})

// Configuration validation warnings
const configWarnings = computed(() => {
  const warnings = []

  if (modelLayerInfo.value) {
    // Check context size
    if (config.value.ctx_size > modelLayerInfo.value.context_length) {
      warnings.push({
        type: 'warning',
        message: `Context size (${config.value.ctx_size}) exceeds model's maximum context length (${modelLayerInfo.value.context_length})`,
        field: 'ctx_size'
      })
    }

    // Check batch size
    if (config.value.batch_size > maxBatchSize.value) {
      warnings.push({
        type: 'warning',
        message: `Batch size (${config.value.batch_size}) exceeds recommended maximum (${maxBatchSize.value}) based on model architecture`,
        field: 'batch_size'
      })
    }

    // Check GPU layers
    if (config.value.n_gpu_layers > modelLayerInfo.value.layer_count) {
      warnings.push({
        type: 'error',
        message: `GPU layers (${config.value.n_gpu_layers}) exceeds model's total layers (${modelLayerInfo.value.layer_count})`,
        field: 'n_gpu_layers'
      })
    }

    // Check parallel processing
    if (config.value.parallel > maxParallel.value) {
      warnings.push({
        type: 'warning',
        message: `Parallel processing (${config.value.parallel}) exceeds recommended maximum (${maxParallel.value}) based on attention head count`,
        field: 'parallel'
      })
    }
  }

  return warnings
})

onMounted(async () => {
  await loadModel()
  await systemStore.fetchSystemStatus()
  await loadSupportedFlags()
  initializeConfig()

  // Subscribe to unified monitoring updates for real-time memory data
  wsStore.subscribeToUnifiedMonitoring((data) => {
    // Extract RAM data from unified stream
    if (data.system?.memory) {
      const memory = data.system.memory
      const total = memory.total ?? 0
      const available = memory.available ?? memory.free ?? 0
      const used = memory.used ?? (total > 0 ? Math.max(0, total - available) : 0)
      const percent = memory.percent ?? (total > 0 ? (used / total) * 100 : 0)
      const swapTotal = memory.swap_total ?? 0
      const swapUsed = memory.swap_used ?? 0
      const swapPercent = memory.swap_percent ?? (swapTotal > 0 ? (swapUsed / swapTotal) * 100 : 0)

      realtimeRamData.value = {
        total,
        available,
        used,
        percent,
        free: memory.free ?? null,
        cached: memory.cached ?? null,
        buffers: memory.buffers ?? null,
        swap_total: swapTotal,
        swap_used: swapUsed,
        swap_percent: swapPercent,
        timestamp: Date.now()
      }
    }

    // Extract VRAM data from unified stream
    if (data.gpu?.vram_data) {
      realtimeVramData.value = data.gpu.vram_data
    }

    // Extract running instance data for performance metrics
  })

  // Automatically trigger initial estimates if no real-time data yet
  if (!realtimeRamData.value) {
    estimateRam()
  }
  if (!realtimeVramData.value && !systemStore.gpuInfo.cpu_only_mode) {
    estimateVram()
  }
})

onUnmounted(() => {
  // WebSocket subscriptions are automatically cleaned up by the store
  if (vramEstimateTimeout) {
    clearTimeout(vramEstimateTimeout)
    vramEstimateTimeout = null
  }
  if (ramEstimateTimeout) {
    clearTimeout(ramEstimateTimeout)
    ramEstimateTimeout = null
  }
  if (vramAbortController) {
    vramAbortController.abort()
    vramAbortController = null
  }
  if (ramAbortController) {
    ramAbortController.abort()
    ramAbortController = null
  }
})

const loadModel = async () => {
  const modelId = route.params.id
  if (modelId) {
    try {
      await modelStore.fetchModels()
      // Find the specific quantization in the grouped structure
      model.value = modelStore.allQuantizations.find(m => m.id === parseInt(modelId))
      if (!model.value) {
        toast.error('The requested model quantization could not be found')
        router.push('/models')
        return
      }

      // Load model layer information
      await loadModelLayerInfo()
      
      // Load model recommendations
      await loadModelRecommendations()
      await loadHfMetadata()

    } catch (error) {
      console.error('Failed to load model:', error)
      toast.error('Failed to load model')
    }
  }
}

const loadModelLayerInfo = async () => {
  if (!model.value) return

  layerInfoLoading.value = true
  try {
    const response = await fetch(`/api/models/${model.value.id}/layer-info`)
    if (response.ok) {
      modelLayerInfo.value = await response.json()
      console.log('Model layer info:', modelLayerInfo.value)
    } else {
      console.warn('Failed to load model layer info, using defaults')
      modelLayerInfo.value = { layer_count: 32 }
    }
  } catch (error) {
    console.error('Error loading model layer info:', error)
    modelLayerInfo.value = { layer_count: 32 }
  } finally {
    layerInfoLoading.value = false
  }
}

const loadModelRecommendations = async () => {
  if (!model.value) return

  recommendationsLoading.value = true
  try {
    const response = await fetch(`/api/models/${model.value.id}/recommendations`)
    if (response.ok) {
      modelRecommendations.value = await response.json()
      console.log('Model recommendations:', modelRecommendations.value)
    } else {
      console.warn('Failed to load model recommendations')
      modelRecommendations.value = null
    }
  } catch (error) {
    console.error('Error loading model recommendations:', error)
    modelRecommendations.value = null
  } finally {
    recommendationsLoading.value = false
  }
}

const loadHfMetadata = async () => {
  if (!model.value) return
  hfMetadataLoading.value = true
  try {
    const data = await modelStore.fetchHfMetadata(model.value.id)
    hfMetadata.value = data?.metadata || null
    hfDefaults.value = data?.hf_defaults || null
    hfLayerInfo.value = data?.gguf_layer_info || null
  } catch (error) {
    console.error('Failed to load Hugging Face metadata:', error)
    hfMetadata.value = null
    hfDefaults.value = null
    hfLayerInfo.value = null
  } finally {
    hfMetadataLoading.value = false
  }
}

const regenerateModelInfo = async () => {
  if (!model.value) return

  regeneratingInfo.value = true
  try {
    const response = await fetch(`/api/models/${model.value.id}/regenerate-info`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      }
    })

    if (response.ok) {
      const result = await response.json()
      toast.success('Model information regenerated successfully')

      // Update model from store if available
      if (modelStore) {
        await modelStore.fetchModels()
        // Reload the current model
        model.value = modelStore.allQuantizations.find(m => m.id === model.value.id)
      }

      // Reload layer info to get updated metadata
      await loadModelLayerInfo()
      
      // Reload recommendations with updated metadata
      await loadModelRecommendations()
      await loadHfMetadata()

      console.log('Regenerated model info:', result)
    } else {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
      toast.error(`Failed to regenerate model info: ${error.detail || 'Unknown error'}`)
    }
  } catch (error) {
    console.error('Error regenerating model info:', error)
    toast.error('Failed to regenerate model information')
  } finally {
    regeneratingInfo.value = false
  }
}

const applyEmbeddingDefaults = () => {
  if (isEmbeddingModel.value && !config.value.embedding) {
    config.value.embedding = true
  }
}

const parseModelConfig = (rawConfig) => {
  if (!rawConfig) return null
  if (typeof rawConfig === 'object') {
    return { ...rawConfig }
  }
  if (typeof rawConfig === 'string') {
    try {
      return JSON.parse(rawConfig)
    } catch (error) {
      console.error('Failed to parse model config:', error)
      return null
    }
  }
  return null
}

const initializeConfig = () => {
  const defaults = getDefaultConfig()
  const parsed = parseModelConfig(model.value?.config)
  if (parsed) {
    config.value = { ...defaults, ...parsed }
  } else {
    config.value = { ...defaults }
  }
  if ('port' in config.value) {
    delete config.value.port
  }
  for (const key in defaults) {
    if (config.value[key] === undefined || config.value[key] === null) {
      config.value[key] = defaults[key]
    }
    if (typeof defaults[key] === 'boolean' && typeof config.value[key] !== 'boolean') {
      config.value[key] = Boolean(config.value[key])
    }
    if (typeof defaults[key] === 'number' && typeof config.value[key] !== 'number') {
      const num = Number(config.value[key])
      config.value[key] = isNaN(num) ? defaults[key] : num
    }
  }
  applyEmbeddingDefaults()
}

const getDefaultConfig = () => ({
  n_gpu_layers: 0,
  main_gpu: 0,
  tensor_split: '',
  ctx_size: 4096,
  batch_size: 256,
  ubatch_size: 128,
  no_mmap: false,
  mlock: false,
  low_vram: false,
  logits_all: false,
  embedding: false,
  cont_batching: true,
  no_kv_offload: false,
  n_predict: -1,
  temp: 0.8,
  temperature: 0.8,
  top_k: 40,
  top_p: 0.9,
  repeat_penalty: 1.1,
  threads: 6,
  threads_batch: 6,
  parallel: 1,
  flash_attn: false,
  cache_type_k: 'f16',
  cache_type_v: null,
  moe_offload_pattern: 'none',
  moe_offload_custom: '',
  rope_freq_base: null,
  rope_freq_scale: null,
  yarn_ext_factor: null,
  yarn_attn_factor: null,
  rope_scaling: '',
  yaml: '',
  customArgs: '',
  min_p: null,
  typical_p: null,
  tfs_z: null,
  presence_penalty: null,
  frequency_penalty: null,
  mirostat: null,
  mirostat_tau: null,
  mirostat_eta: null,
  seed: null,
  stop: [],
  grammar: '',
  json_schema: '',
  jinja: false,
  host: '0.0.0.0',
  timeout: 300
})

// Track the last auto-generated YARN settings so we don't overwrite or erase
// user-provided values when they manually tune advanced parameters.
const autoYarnConfig = ref({
  rope_scaling: '',
  yarn_ext_factor: null,
  yarn_attn_factor: null
})

// Apply architecture preset
const selectedPreset = ref(null)
const applyPreset = async (presetName, skipPreview = false) => {
  if (!model.value) return

  try {
    const response = await fetch(`/api/models/${model.value.id}/architecture-presets`)
    if (!response.ok) throw new Error('Failed to fetch presets')

    const data = await response.json()
    const preset = data.presets[presetName]

    if (preset) {
      // Calculate changes
      const changes = calculateChanges(preset, config.value)
      
      if (!skipPreview && changes.length > 0) {
        // Show preview
        previewData.value = {
          type: 'preset',
          presetName: presetName,
          changes: changes,
          impact: null,
          newConfig: { ...config.value, ...preset }
        }
        showPreview.value = true
        return
      }

      // Apply directly if skipPreview is true
      selectedPreset.value = presetName
      Object.assign(config.value, preset)

      // Re-estimate memory to reflect preset changes
      await estimateVram()
      await estimateRam()

      toast.success(`${presetName.charAt(0).toUpperCase() + presetName.slice(1)} preset applied`)
    }
  } catch (error) {
    toast.error('Failed to apply preset')
  }
}

// Watch for flash attention changes to warn about V cache
watch(() => config.value.flash_attn, (newVal) => {
  // Ensure cache_type_v has a safe value based on flash_attn state
  if (!newVal) {
    // When Flash Attention is disabled, V cache should be null or f16
    if (!config.value.cache_type_v || typeof config.value.cache_type_v !== 'string') {
      config.value.cache_type_v = null
    }
  } else {
    // When Flash Attention is enabled, ensure V cache has a valid string value
    if (!config.value.cache_type_v || typeof config.value.cache_type_v !== 'string') {
      config.value.cache_type_v = config.value.cache_type_k || 'f16'
    }
  }
})

// Handle wizard-generated configuration
const handleWizardConfig = async (wizardConfig) => {
  try {
    if (!model.value) {
      toast.error('No model selected')
      return
    }

    // Apply wizard configuration
    const defaults = getDefaultConfig()
    Object.assign(config.value, defaults, wizardConfig)

    // Re-estimate memory to reflect config changes
    await estimateVram()
    await estimateRam()

    // Automatically save to backend
    saveLoading.value = true
    try {
      await modelStore.updateModelConfig(model.value.id, config.value)
      
      // Reload model to get updated config
      await loadModel()
      
      toast.success('Configuration wizard settings applied and saved successfully')
      showWizard.value = false
    } catch (saveError) {
      console.error('Save config error:', saveError)
      toast.error('Configuration applied but failed to save. Please click Save Config manually.')
    } finally {
      saveLoading.value = false
    }
  } catch (error) {
    console.error('Error applying wizard config:', error)
    toast.error('Failed to apply wizard configuration')
  }
}

// Handle smart auto from Quick Start Modal
const handleQuickStartSmartAuto = (usageMode) => {
  smartAutoUsageMode.value = usageMode
  generateAutoConfig()
}

const generateAutoConfig = async (skipPreview = false) => {
  autoConfigLoading.value = true
  try {
    if (!model.value) {
      toast.error('No model selected')
      return
    }

    // Call backend smart auto API (send preset and usage_mode if provided)
    const params = new URLSearchParams()
    if (selectedPreset.value) {
      params.append('preset', selectedPreset.value)
    }
    if (smartAutoUsageMode.value) {
      params.append('usage_mode', smartAutoUsageMode.value)
    }
    const queryString = params.toString()
    const url = `/api/models/${model.value.id}/smart-auto${queryString ? '?' + queryString : ''}`
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    })

    if (!response.ok) {
      throw new Error(`Smart auto failed: ${response.statusText}`)
    }

    const smartConfig = await response.json()
    const defaults = getDefaultConfig()
    const newConfig = { ...defaults, ...smartConfig }

    // Calculate changes
    const changes = calculateChanges(newConfig, config.value)
    
    // Calculate impact
    const impact = await calculateImpact(newConfig, config.value)

    if (!skipPreview && changes.length > 0) {
      // Show preview
      previewData.value = {
        type: 'smart-auto',
        presetName: '',
        changes: changes,
        impact: impact,
        newConfig: newConfig
      }
      showPreview.value = true
      autoConfigLoading.value = false
      return
    }

    // Apply the smart configuration with defaults fallback
    config.value = newConfig

    // Ensure all fields have safe values (handle nulls/undefined)
    for (const key in defaults) {
      if (config.value[key] === undefined || config.value[key] === null) {
        config.value[key] = defaults[key]
      }
      // Type coercion for critical fields
      if (typeof defaults[key] === 'boolean' && typeof config.value[key] !== 'boolean') {
        config.value[key] = Boolean(config.value[key])
      }
      if (typeof defaults[key] === 'number' && typeof config.value[key] !== 'number') {
        const num = Number(config.value[key])
        config.value[key] = isNaN(num) ? defaults[key] : num
      }
    }

    // Show success message with optimization details
    const isCpuOnlyMode = systemStore.gpuInfo.cpu_only_mode
    const optimizationType = isCpuOnlyMode ? 'CPU-optimized' : 'GPU-optimized'

    toast.success(`${optimizationType} configuration generated successfully`)

    // Update estimates after applying smart config
    await estimateVram()
    await estimateRam()

  } catch (error) {
    toast.error('Failed to generate automatic configuration')
  } finally {
    autoConfigLoading.value = false
  }
}

const estimateVram = async () => {
  if (!model.value) return

  if (vramAbortController) {
    vramAbortController.abort()
  }

  const controller = new AbortController()
  vramAbortController = controller
  vramLoading.value = true

  try {
    const response = await fetch('/api/models/vram-estimate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model_id: model.value.id,
        config: config.value,
        usage_mode: smartAutoUsageMode.value
      }),
      signal: controller.signal
    })

    if (!response.ok) {
      throw new Error('VRAM estimation failed')
    }

    const data = await response.json()
    vramEstimate.value = data
  } catch (error) {
    if (error?.name === 'AbortError') {
      return
    }
    console.error('VRAM estimation error:', error)
    toast.error('Could not estimate VRAM usage')
  } finally {
    if (vramAbortController === controller) {
      vramLoading.value = false
      vramAbortController = null
    }
  }
}

const estimateRam = async () => {
  if (!model.value) return

  if (ramAbortController) {
    ramAbortController.abort()
  }

  const controller = new AbortController()
  ramAbortController = controller
  ramLoading.value = true

  try {
    const response = await fetch('/api/models/ram-estimate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model_id: model.value.id,
        config: config.value,
        usage_mode: smartAutoUsageMode.value
      }),
      signal: controller.signal
    })

    if (!response.ok) {
      throw new Error('RAM estimation failed')
    }

    ramEstimate.value = await response.json()
  } catch (error) {
    if (error?.name === 'AbortError') {
      return
    }
    console.error('RAM estimation error:', error)
    toast.error('Could not estimate RAM usage')
  } finally {
    if (ramAbortController === controller) {
      ramLoading.value = false
      ramAbortController = null
    }
  }
}

const updateVramEstimate = (force = false) => {
  if (force || !vramEstimate.value) {
    return estimateVram()
  }
  if (vramEstimateTimeout) {
    clearTimeout(vramEstimateTimeout)
  }
  vramEstimateTimeout = setTimeout(() => {
    estimateVram()
  }, ESTIMATE_DEBOUNCE_MS)
  return Promise.resolve()
}

const updateRamEstimate = (force = false) => {
  if (force || !ramEstimate.value) {
    return estimateRam()
  }
  if (ramEstimateTimeout) {
    clearTimeout(ramEstimateTimeout)
  }
  ramEstimateTimeout = setTimeout(() => {
    estimateRam()
  }, ESTIMATE_DEBOUNCE_MS)
  return Promise.resolve()
}

const saveConfig = async () => {
  if (!model.value) return

  saveLoading.value = true
  try {
    await modelStore.updateModelConfig(model.value.id, config.value)

    toast.success('Model configuration has been updated')
  } catch (error) {
    console.error('Save config error:', error)
    toast.error('Failed to save configuration')
  } finally {
    saveLoading.value = false
  }
}

// Calculate changes between old and new config
const calculateChanges = (newConfig, oldConfig) => {
  const changes = []
  const importantFields = {
    n_gpu_layers: 'GPU Layers',
    ctx_size: 'Context Size',
    batch_size: 'Batch Size',
    temp: 'Temperature',
    temperature: 'Temperature',
    top_k: 'Top-K',
    top_p: 'Top-P',
    repeat_penalty: 'Repeat Penalty',
    threads: 'CPU Threads',
    flash_attn: 'Flash Attention',
    cont_batching: 'Continuous Batching'
  }

  for (const [key, label] of Object.entries(importantFields)) {
    const oldValue = oldConfig[key]
    const newValue = newConfig[key]
    
    if (oldValue !== newValue && (oldValue !== undefined || newValue !== undefined)) {
      changes.push({
        field: label,
        before: oldValue,
        after: newValue,
        description: getFieldDescription(key, oldValue, newValue)
      })
    }
  }

  return changes
}

// Get description for field change
const getFieldDescription = (key, oldValue, newValue) => {
  const descriptions = {
    n_gpu_layers: oldValue < newValue 
      ? 'More layers offloaded to GPU for faster inference' 
      : 'Fewer layers offloaded to GPU to reduce VRAM usage',
    ctx_size: oldValue < newValue 
      ? 'Increased context window for longer conversations' 
      : 'Reduced context window to save memory',
    batch_size: oldValue < newValue 
      ? 'Increased batch size for better throughput' 
      : 'Reduced batch size to save memory',
    temp: oldValue < newValue 
      ? 'Higher temperature for more creative outputs' 
      : 'Lower temperature for more focused outputs'
  }
  return descriptions[key] || ''
}

// Calculate impact of config changes
const calculateImpact = async (newConfig, oldConfig) => {
  const impact = {}
  
  // Estimate VRAM changes
  const oldVram = vramEstimate.value?.estimated_vram || 0
  const newVram = await estimateVramForConfig(newConfig)
  
  if (oldVram && newVram) {
    const diff = newVram - oldVram
    if (Math.abs(diff) > 1000000000) { // > 1GB
      const diffPercent = ((diff / oldVram) * 100).toFixed(1)
      impact.vram = diff > 0 
        ? `VRAM Usage: ${formatFileSize(oldVram)} → ${formatFileSize(newVram)} (+${diffPercent}%)`
        : `VRAM Usage: ${formatFileSize(oldVram)} → ${formatFileSize(newVram)} (${diffPercent}%)`
    }
  }

  // Estimate performance impact
  const oldLayers = oldConfig.n_gpu_layers || 0
  const newLayers = newConfig.n_gpu_layers || 0
  if (newLayers > oldLayers) {
    const layerIncrease = newLayers - oldLayers
    const estimatedSpeedup = Math.min(50, Math.round((layerIncrease / 32) * 50))
    if (estimatedSpeedup > 5) {
      impact.performance = `Performance: ~${estimatedSpeedup}% faster with more GPU layers`
    }
  }

  return Object.keys(impact).length > 0 ? impact : null
}

// Estimate VRAM for a config without applying it
const estimateVramForConfig = async (testConfig) => {
  if (!model.value) return 0
  
  try {
    const response = await fetch('/api/models/vram-estimate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_id: model.value.id,
        config: testConfig
      })
    })
    if (response.ok) {
      const estimate = await response.json()
      return estimate.estimated_vram || 0
    }
  } catch (error) {
    console.error('VRAM estimation error:', error)
  }
  return vramEstimate.value?.estimated_vram || 0
}

// Apply preview changes
const applyPreviewChanges = async () => {
  if (!previewData.value) return
  
  previewApplying.value = true
  try {
    // Apply the new config
    config.value = previewData.value.newConfig
    
    // Ensure all fields have safe values
    const defaults = getDefaultConfig()
    for (const key in defaults) {
      if (config.value[key] === undefined || config.value[key] === null) {
        config.value[key] = defaults[key]
      }
      // Type coercion for critical fields
      if (typeof defaults[key] === 'boolean' && typeof config.value[key] !== 'boolean') {
        config.value[key] = Boolean(config.value[key])
      }
      if (typeof defaults[key] === 'number' && typeof config.value[key] !== 'number') {
        const num = Number(config.value[key])
        config.value[key] = isNaN(num) ? defaults[key] : num
      }
    }
    
    // Re-estimate memory
    await estimateVram()
    await estimateRam()
    
    // Save the preset name and close preview
    const previewType = previewData.value.type
    const presetName = previewData.value.presetName
    showPreview.value = false
    previewData.value = null
    
    // Show success message
    if (previewType === 'preset' && presetName) {
      selectedPreset.value = presetName
      toast.success(`${presetName.charAt(0).toUpperCase() + presetName.slice(1)} preset applied`)
    } else {
      const isCpuOnlyMode = systemStore.gpuInfo.cpu_only_mode
      const optimizationType = isCpuOnlyMode ? 'CPU-optimized' : 'GPU-optimized'
      toast.success(`${optimizationType} configuration applied successfully`)
    }
  } catch (error) {
    console.error('Error applying changes:', error)
    toast.error('Failed to apply configuration changes')
  } finally {
    previewApplying.value = false
  }
}

// Tooltip generators with architecture-specific recommendations
const getTemperatureTooltip = () => {
  const architecture = modelLayerInfo.value?.architecture?.toLowerCase() || ''
  let baseMsg = 'Controls randomness (0.1=deterministic, 2.0=creative)'

  if (architecture.includes('glm')) {
    baseMsg += ' | GLM: Recommended 1.0'
  } else if (architecture.includes('deepseek')) {
    baseMsg += ' | DeepSeek: Recommended 1.0'
  } else if (architecture.includes('qwen')) {
    baseMsg += ' | Qwen: Recommended 0.7'
  } else if (architecture.includes('codellama') || model.value?.name.toLowerCase().includes('code')) {
    baseMsg += ' | Coding: Recommended 0.1-0.7'
  }

  return baseMsg
}

const getTopKTooltip = () => {
  return `Top-K sampling (limit to top K tokens) | Recommended: 40 for GLM/DeepSeek, 50 for others`
}

const getTopPTooltip = () => {
  const architecture = modelLayerInfo.value?.architecture?.toLowerCase() || ''
  let baseMsg = 'Top-P (nucleus) sampling'

  if (architecture.includes('glm') || architecture.includes('deepseek')) {
    baseMsg += ' | Recommended: 0.95'
  } else if (architecture.includes('qwen')) {
    baseMsg += ' | Recommended: 0.9-0.95'
  } else {
    baseMsg += ' | Recommended: 0.95'
  }

  return baseMsg
}

const getRepeatPenaltyTooltip = () => {
  let baseMsg = 'Penalty for repeating tokens (1.0=no penalty)'

  const ctxLength = modelLayerInfo.value?.context_length || 0
  if (ctxLength > 32768) {
    baseMsg += ' | Long context: Use 1.0-1.05'
  } else if (ctxLength < 2048) {
    baseMsg += ' | Short context: Use 1.1-1.2'
  } else {
    baseMsg += ' | Standard: Use 1.1'
  }

  return baseMsg
}

// Rich tooltip data for key settings
const gpuLayersTooltip = computed(() => {
  const rec = modelRecommendations.value?.gpu_layers
  if (!rec) {
    return {
      description: 'Controls how many model layers are offloaded to GPU. More layers = faster inference but higher VRAM usage.',
      whenToAdjust: 'Increase for faster generation if you have VRAM. Decrease if running out of VRAM.',
      tradeoffs: [
        'Higher values: Faster inference, better for long conversations',
        'Lower values: Lower VRAM usage, falls back to CPU which is slower'
      ],
      recommended: 'For this model: Loading recommendations...',
      ranges: []
    }
  }
  
  return {
    description: 'Controls how many model layers are offloaded to GPU. More layers = faster inference but higher VRAM usage.',
    whenToAdjust: 'Increase for faster generation if you have VRAM. Decrease if running out of VRAM.',
    tradeoffs: [
      'Higher values: Faster inference, better for long conversations',
      'Lower values: Lower VRAM usage, falls back to CPU which is slower'
    ],
    recommended: rec.description,
    ranges: rec.ranges.map(r => `${r.value} layers: ${r.description}`)
  }
})

const contextSizeTooltip = computed(() => {
  const rec = modelRecommendations.value?.context_size
  if (!rec) {
    return {
      description: 'Maximum number of tokens the model can process in context. Higher = can remember more but uses more memory.',
      whenToAdjust: 'Increase for long conversations or documents. Decrease if running low on memory.',
      tradeoffs: [
        'Higher values: Can handle longer conversations, better for documents',
        'Lower values: Lower memory usage, faster processing'
      ],
      recommended: 'For this model: Loading recommendations...',
      ranges: []
    }
  }
  
  return {
    description: 'Maximum number of tokens the model can process in context. Higher = can remember more but uses more memory.',
    whenToAdjust: 'Increase for long conversations or documents. Decrease if running low on memory.',
    tradeoffs: [
      'Higher values: Can handle longer conversations, better for documents',
      'Lower values: Lower memory usage, faster processing'
    ],
    recommended: rec.description,
    ranges: rec.ranges.map(r => `${r.min}-${r.max}: ${r.description}`)
  }
})

const temperatureTooltip = computed(() => {
  const rec = modelRecommendations.value?.temperature
  if (!rec) {
    return {
      description: 'Controls randomness in model responses. Lower = more focused and deterministic, Higher = more creative and varied.',
      whenToAdjust: 'Lower for code/technical tasks. Higher for creative writing. Adjust if outputs are too repetitive or too random.',
      tradeoffs: [
        'Low (0.1-0.3): Focused, deterministic, good for code',
        'Medium (0.7-1.0): Balanced, natural conversations',
        'High (1.5-2.0): Creative, varied, unpredictable'
      ],
      recommended: 'For this model: Loading recommendations...',
      ranges: []
    }
  }
  
  return {
    description: 'Controls randomness in model responses. Lower = more focused and deterministic, Higher = more creative and varied.',
    whenToAdjust: 'Lower for code/technical tasks. Higher for creative writing. Adjust if outputs are too repetitive or too random.',
    tradeoffs: [
      'Low (0.1-0.3): Focused, deterministic, good for code',
      'Medium (0.7-1.0): Balanced, natural conversations',
      'High (1.5-2.0): Creative, varied, unpredictable'
    ],
    recommended: rec.description,
    ranges: rec.ranges.map(r => `${r.min}-${r.max}: ${r.description}`)
  }
})

const batchSizeTooltip = computed(() => {
  const rec = modelRecommendations.value?.batch_size
  if (!rec) {
    return {
      description: 'Number of tokens processed in parallel. Higher = faster but uses more memory.',
      whenToAdjust: 'Increase if you have VRAM available. Decrease if getting out-of-memory errors.',
      tradeoffs: [
        'Higher values: Faster token generation, better throughput',
        'Lower values: Lower memory usage, more sequential processing'
      ],
      recommended: 'For this model: Loading recommendations...',
      ranges: []
    }
  }
  
  return {
    description: 'Number of tokens processed in parallel. Higher = faster but uses more memory.',
    whenToAdjust: 'Increase if you have VRAM available. Decrease if getting out-of-memory errors.',
    tradeoffs: [
      'Higher values: Faster token generation, better throughput',
      'Lower values: Lower memory usage, more sequential processing'
    ],
    recommended: rec.description,
    ranges: rec.ranges.map(r => `${r.min}-${r.max}: ${r.description}`)
  }
})

const topKTooltip = computed(() => {
  const rec = modelRecommendations.value?.top_k
  if (!rec) {
    return {
      description: 'Limits sampling to the top K most likely tokens. Lower = more focused, Higher = more diverse.',
      whenToAdjust: 'Lower for focused outputs. Higher for more variety. Works together with Top-P.',
      tradeoffs: [
        'Lower values (10-30): More focused, deterministic outputs',
        'Medium values (40-50): Balanced diversity (recommended)',
        'Higher values (100+): More random, less coherent outputs'
      ],
      recommended: 'For this model: Loading recommendations...',
      ranges: []
    }
  }
  
  return {
    description: 'Limits sampling to the top K most likely tokens. Lower = more focused, Higher = more diverse.',
    whenToAdjust: 'Lower for focused outputs. Higher for more variety. Works together with Top-P.',
    tradeoffs: [
      'Lower values (10-30): More focused, deterministic outputs',
      'Medium values (40-50): Balanced diversity (recommended)',
      'Higher values (100+): More random, less coherent outputs'
    ],
    recommended: rec.description,
    ranges: rec.ranges.map(r => `${r.min}-${r.max}: ${r.description}`)
  }
})

const topPTooltip = computed(() => {
  const rec = modelRecommendations.value?.top_p
  if (!rec) {
    return {
      description: 'Nucleus sampling: considers tokens with cumulative probability mass up to P. Works with Top-K.',
      whenToAdjust: 'Lower for more focused outputs. Higher for more diversity. Typically keep at 0.9-0.95.',
      tradeoffs: [
        'Lower values (0.7-0.8): More conservative, focused sampling',
        'Medium values (0.9-0.95): Balanced diversity (recommended)',
        'Higher values (0.98-1.0): Includes very low-probability tokens'
      ],
      recommended: 'For this model: Loading recommendations...',
      ranges: []
    }
  }
  
  return {
    description: 'Nucleus sampling: considers tokens with cumulative probability mass up to P. Works with Top-K.',
    whenToAdjust: 'Lower for more focused outputs. Higher for more diversity. Typically keep at 0.9-0.95.',
    tradeoffs: [
      'Lower values (0.7-0.8): More conservative, focused sampling',
      'Medium values (0.9-0.95): Balanced diversity (recommended)',
      'Higher values (0.98-1.0): Includes very low-probability tokens'
    ],
    recommended: rec.description,
    ranges: rec.ranges.map(r => `${r.min}-${r.max}: ${r.description}`)
  }
})

const repeatPenaltyTooltip = computed(() => {
  const ctxLength = modelLayerInfo.value?.context_length || 0
  let recommended = '1.1 for standard contexts'
  let ranges = [
    '1.0: No penalty (allows repetition)',
    '1.1: Standard penalty (recommended)',
    '1.2: Strong penalty (prevents most repetition)'
  ]

  if (ctxLength > 32768) {
    recommended = '1.0-1.05 for long contexts'
    ranges = [
      '1.0-1.05: Minimal penalty (long context models)',
      '1.1: Standard (good for most cases)',
      '1.2+: Strong penalty (short contexts)'
    ]
  } else if (ctxLength < 2048) {
    recommended = '1.1-1.2 for short contexts'
  }

  return {
    description: 'Penalty applied to tokens that have appeared in the context. Higher = less repetition.',
    whenToAdjust: 'Increase if model repeats too much. Decrease if model avoids valid repetition. Adjust based on context length.',
    tradeoffs: [
      'Lower values (1.0-1.05): Allows natural repetition, good for long contexts',
      'Medium values (1.1): Balanced, prevents excessive repetition (recommended)',
      'Higher values (1.2-2.0): Strong prevention, may avoid valid repetition'
    ],
    recommended: recommended,
    ranges: ranges
  }
})

// Search functionality - computed properties to check section/field matches
// Tab matching for search - find which tab contains matching content
const matchingTabIndex = computed(() => {
  const query = configSearchQuery.value.toLowerCase().trim()
  if (!query) return null
  
  const searchTerms = query.split(/\s+/)
  
  for (let i = 0; i < tabLabels.length; i++) {
    const tab = tabLabels[i]
    const matchesAny = searchTerms.some(term => 
      tab.keywords.some(keyword => keyword.includes(term) || term.includes(keyword))
    )
    if (matchesAny) return i
  }
  
  return null
})

// Auto-switch to matching tab when searching
watch(matchingTabIndex, (tabIndex) => {
  if (tabIndex !== null && tabIndex !== undefined) {
    activeTabIndex.value = tabIndex
  }
})

// Check if current model is running
// Handle stop model
const handleStopModel = async () => {
  if (!model.value) return
  
  try {
    await modelStore.stopModel(model.value.id)
    toast.success('Model stopped successfully')
  } catch (error) {
    console.error('Error stopping model:', error)
    toast.error('Failed to stop model')
  }
}

// Watch for config changes to update estimates
watch(config, () => {
  updateVramEstimate()
  updateRamEstimate()
}, { deep: true })

// Watch for model changes to update running state
// Handle onboarding tour
const handleOnboardingComplete = () => {
  localStorage.setItem('model-config-onboarding-completed', 'true')
  showOnboarding.value = false
}

const handleOnboardingSkip = () => {
  showOnboarding.value = false
  // Don't mark as completed if skipped
}

// Check if model has no configuration (empty state)
const valuesDiffer = (value, defaultValue) => {
  if (Array.isArray(value) && Array.isArray(defaultValue)) {
    if (value.length !== defaultValue.length) return true
    return value.some((v, idx) => valuesDiffer(v, defaultValue[idx]))
  }
  return value !== defaultValue && value !== null && value !== undefined && value !== ''
}

const hasNoConfig = computed(() => {
  if (!model.value) return false
  if (!model.value.config) return true
  const parsed = parseModelConfig(model.value.config)
  if (!parsed) return true
  const defaults = getDefaultConfig()
  return !Object.keys(parsed).some(key => valuesDiffer(parsed[key], defaults[key]))
})

</script>

<style scoped>
.model-config {
  min-height: 100vh;
  background: var(--bg-primary);
  overflow-x: hidden;
}

.config-layout {
  display: grid;
  grid-template-columns: 1fr 320px;
  gap: var(--spacing-lg);
  max-width: 1400px;
  margin: 0 auto;
  padding: var(--spacing-lg);
  box-sizing: border-box;
  align-items: start;
}

.config-main {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  overflow-x: hidden;
}

.config-tabs-wrapper {
  width: 100%;
  margin-top: var(--spacing-sm);
}

.config-tabs {
  width: 100%;
}

.config-tabs :deep(.p-tabview-nav) {
  background: transparent;
  border-bottom: 2px solid var(--border-primary);
  border-radius: 0;
  padding: 0;
  margin-bottom: var(--spacing-md);
  gap: 0;
  display: flex;
  align-items: center;
}

.config-tabs :deep(.p-tabview-nav li) {
  margin-right: var(--spacing-sm);
}

.config-tabs :deep(.p-tabview-nav li:last-child) {
  margin-right: 0;
}

.config-tabs :deep(.p-tabview-nav li .p-tabview-nav-link) {
  padding: var(--spacing-sm) var(--spacing-md);
  border-radius: var(--radius-md) var(--radius-md) 0 0;
  transition: all var(--transition-normal);
  color: var(--text-secondary);
  border: none;
  border-bottom: 3px solid transparent;
  margin-bottom: -2px;
  background: transparent;
  font-weight: 500;
}

.config-tabs :deep(.p-tabview-nav li.p-highlight .p-tabview-nav-link) {
  background: transparent;
  color: var(--accent-cyan);
  font-weight: 600;
  border-bottom-color: var(--accent-cyan);
}

.config-tabs :deep(.p-tabview-nav li .p-tabview-nav-link:hover) {
  background: var(--bg-surface);
  color: var(--text-primary);
  border-bottom-color: var(--text-secondary);
}

.config-tabs :deep(.p-tabview-nav li.p-highlight .p-tabview-nav-link:hover) {
  border-bottom-color: var(--accent-cyan);
  color: var(--accent-cyan);
}

.config-tabs :deep(.p-tabview-panels) {
  padding: 0;
  background: transparent;
  border: none;
}

.tab-content {
  padding: var(--spacing-md);
  min-height: 200px;
  background: transparent;
}

.tab-section {
  margin-bottom: var(--spacing-md);
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--spacing-md);
}

.tab-section:last-child {
  margin-bottom: 0;
}

.tab-section-title {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  margin-bottom: var(--spacing-md);
  padding-bottom: var(--spacing-xs);
  border-bottom: 1px solid var(--border-primary);
  color: var(--text-primary);
  font-size: 1rem;
  font-weight: 600;
  grid-column: 1 / -1;
}

.tab-section-title i {
  font-size: 1.2rem;
  color: var(--accent-cyan);
}

.config-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--spacing-md);
}

.model-info {
  flex: 1;
}

.model-tag.tag-architecture {
  background: rgba(59, 130, 246, 0.1);
  color: var(--accent-blue);
  border: 1px solid rgba(59, 130, 246, 0.2);
}

.model-tag.tag-layers {
  background: rgba(34, 211, 238, 0.1);
  color: var(--accent-cyan);
  border: 1px solid rgba(34, 211, 238, 0.2);
}

.model-tag.tag-pipeline {
  background: rgba(59, 130, 246, 0.12);
  color: var(--accent-blue);
  border: 1px solid rgba(59, 130, 246, 0.35);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  font-weight: 600;
}

.embedding-notice {
  margin-top: var(--spacing-sm);
  padding: var(--spacing-sm);
  border-radius: var(--radius-md);
  border: 1px solid rgba(59, 130, 246, 0.3);
  background: rgba(59, 130, 246, 0.08);
  display: flex;
  gap: var(--spacing-sm);
  align-items: flex-start;
}

.embedding-notice i {
  font-size: 1.25rem;
  color: var(--accent-blue);
}

.embedding-notice p {
  margin: 0;
  color: var(--text-secondary);
  font-size: 0.85rem;
}

.hf-defaults-card {
  margin-top: var(--spacing-sm);
  padding: var(--spacing-md);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-primary);
  background: var(--bg-card);
  box-shadow: var(--shadow-sm);
}

.hf-defaults-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: var(--spacing-sm);
}

.hf-defaults-title {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.hf-defaults-title i {
  font-size: 1.1rem;
  color: var(--accent-blue);
}

.hf-defaults-title strong {
  display: block;
  color: var(--text-primary);
}

.hf-defaults-title span {
  display: block;
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.hf-defaults-spinner {
  font-size: 1rem;
  color: var(--text-secondary);
}

.hf-defaults-body {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.hf-defaults-row {
  display: flex;
  justify-content: space-between;
  font-size: 0.9rem;
}

.hf-defaults-label {
  color: var(--text-secondary);
}

.hf-defaults-value {
  font-weight: 600;
  color: var(--text-primary);
}

.hf-defaults-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: var(--spacing-sm);
}

.hf-defaults-subtitle {
  margin-top: var(--spacing-md);
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-secondary);
}

.hf-default-chip {
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  padding: var(--spacing-xs) var(--spacing-sm);
  background: var(--bg-surface);
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.chip-label {
  font-size: 0.75rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--text-secondary);
}

.chip-value {
  font-weight: 600;
  color: var(--text-primary);
}

/* Configuration Warnings */
.config-warnings {
  margin-top: var(--spacing-lg);
  padding: var(--spacing-md);
  background: var(--bg-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow-md);
}

.warnings-list {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  margin-top: var(--spacing-md);
}

.warning-item {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm) var(--spacing-md);
  border-radius: var(--radius-md);
  font-size: 0.9rem;
  transition: all var(--transition-normal);
}

.warning-item.warning {
  background: rgba(245, 158, 11, 0.1);
  border: 1px solid rgba(245, 158, 11, 0.3);
  color: var(--status-warning);
}

.warning-item.error {
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.3);
  color: var(--status-error);
}

.warning-item i {
  font-size: 1rem;
  flex-shrink: 0;
}

.warning-item.warning i {
  color: var(--status-warning);
}

.warning-item.error i {
  color: var(--status-error);
}

.meta-item {
  padding: var(--spacing-xs) var(--spacing-sm);
  background: var(--bg-surface);
  border-radius: var(--radius-sm);
  font-size: 0.875rem;
  color: var(--text-secondary);
}

.header-actions {
  display: flex;
  flex-wrap: wrap;
  gap: var(--spacing-sm);
  align-items: center;
}

.preset-buttons {
  display: flex;
  gap: var(--spacing-xs);
}

.smart-auto-settings-button {
  margin-left: var(--spacing-xs);
}

/* Menu styling for Smart Auto */
:deep(.p-menu) {
  padding: 0.75rem !important;
  min-width: 220px;
  background: var(--surface-ground) !important;
  border: 1px solid var(--surface-border) !important;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
}

:deep(.p-menu .p-menuitem) {
  margin: 0.375rem 0 !important;
}

:deep(.p-menu .p-menuitem-link) {
  padding: 0.875rem 1.125rem !important;
  border-radius: 0.5rem !important;
  transition: all 0.2s ease !important;
  color: var(--text-primary) !important;
  background: transparent !important;
}

:deep(.p-menu .p-menuitem-link .p-menuitem-text) {
  color: var(--text-primary) !important;
  font-weight: 500 !important;
}

:deep(.p-menu .p-menuitem-link .p-menuitem-icon) {
  color: var(--text-secondary) !important;
  margin-right: 0.75rem !important;
}

:deep(.p-menu .p-menuitem-link:hover) {
  background: var(--primary-color) !important;
}

:deep(.p-menu .p-menuitem-link:hover .p-menuitem-text) {
  color: white !important;
}

:deep(.p-menu .p-menuitem-link:hover .p-menuitem-icon) {
  color: white !important;
}

:deep(.p-menu .p-menuitem-separator) {
  margin: 0.625rem 0 !important;
  border-top: 1px solid var(--surface-border) !important;
}

:deep(.p-menu .p-menuitem-link.menu-item-selected),
:deep(.p-menu .p-menuitem-link.active) {
  background: color-mix(in srgb, var(--accent-blue) 18%, transparent) !important;
}

:deep(.p-menu .p-menuitem-link.menu-item-selected .p-menuitem-text),
:deep(.p-menu .p-menuitem-link.active .p-menuitem-text) {
  color: var(--primary-color) !important;
  font-weight: 600 !important;
}

:deep(.p-menu .p-menuitem-link.menu-item-selected .p-menuitem-icon),
:deep(.p-menu .p-menuitem-link.active .p-menuitem-icon) {
  color: var(--primary-color) !important;
}

:deep(.p-menu .p-menuitem-link.menu-item-selected:hover),
:deep(.p-menu .p-menuitem-link.active:hover) {
  background: var(--primary-color) !important;
}

:deep(.p-menu .p-menuitem-link.menu-item-selected:hover .p-menuitem-text),
:deep(.p-menu .p-menuitem-link.active:hover .p-menuitem-text) {
  color: white !important;
}

:deep(.p-menu .p-menuitem-link.menu-item-selected:hover .p-menuitem-icon),
:deep(.p-menu .p-menuitem-link.active:hover .p-menuitem-icon) {
  color: white !important;
}

/* Badge styling for Smart Auto button */
:deep(.p-button .p-badge) {
  padding: 0.4rem 0.7rem !important;
  margin-left: 0.5rem !important;
  border-radius: 0.5rem !important;
  font-size: 0.7rem !important;
  font-weight: 600 !important;
  line-height: 1 !important;
  min-width: auto !important;
  background: rgba(255, 255, 255, 0.35) !important;
  color: rgba(0, 0, 0, 0.9) !important;
  border: 1px solid rgba(0, 0, 0, 0.15) !important;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
}

.usage-mode-badge {
  padding: 0.4rem 0.7rem !important;
  margin-left: 0.5rem !important;
  border-radius: 0.5rem !important;
  font-size: 0.7rem !important;
  font-weight: 600 !important;
  line-height: 1 !important;
  min-width: auto !important;
  background: rgba(255, 255, 255, 0.35) !important;
  color: rgba(0, 0, 0, 0.9) !important;
  border: 1px solid rgba(0, 0, 0, 0.15) !important;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
}


.action-buttons {
  display: flex;
  gap: var(--spacing-xs);
  align-items: center;
}

/* Quick Start Button - Now in header, no separate styles needed */

/* Quick Start Modal Styles */
:deep(.quick-start-modal) {
  max-width: 800px;
  width: 90vw;
}

:deep(.quick-start-modal .p-dialog-header) {
  padding: var(--spacing-xl);
  border-bottom: 1px solid var(--border-primary);
}

:deep(.quick-start-modal .p-dialog-content) {
  padding: var(--spacing-xl);
  max-height: 70vh;
  overflow-y: auto;
}

:deep(.quick-start-modal .p-dialog-footer) {
  padding: var(--spacing-lg) var(--spacing-xl);
  border-top: 1px solid var(--border-primary);
}

.quick-start-modal-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-lg);
}

.quick-start-icon {
  font-size: 3rem;
  flex-shrink: 0;
  line-height: 1;
}

.quick-start-modal-header h3 {
  margin: 0 0 var(--spacing-xs) 0;
  font-size: 1.75rem;
  font-weight: 600;
  color: var(--text-primary);
}

.quick-start-modal-header p {
  margin: 0;
  color: var(--text-secondary);
  font-size: 1rem;
}

.quick-start-content {
  display: grid;
  grid-template-columns: 1fr 1.5fr;
  gap: var(--spacing-xl);
}

.preset-cards {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.preset-card {
  background: var(--bg-surface);
  border: 2px solid var(--border-primary);
  border-radius: var(--radius-lg);
  padding: var(--spacing-lg);
  cursor: pointer;
  transition: all var(--transition-normal);
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
}

.preset-card:hover {
  border-color: var(--accent-cyan);
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}

.preset-card.wizard-card {
  border-color: rgba(34, 211, 238, 0.5);
  background: linear-gradient(135deg, rgba(34, 211, 238, 0.05), rgba(59, 130, 246, 0.05));
}

.preset-card.wizard-card:hover {
  border-color: var(--accent-cyan);
  background: linear-gradient(135deg, rgba(34, 211, 238, 0.1), rgba(59, 130, 246, 0.1));
}

.preset-icon {
  font-size: 2rem;
  flex-shrink: 0;
}

.preset-info h4 {
  margin: 0 0 var(--spacing-xs) 0;
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--text-primary);
}

.preset-info p {
  margin: 0;
  font-size: 0.875rem;
  color: var(--text-secondary);
  line-height: 1.4;
}

.smart-auto-section {
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  padding: var(--spacing-lg);
}

.smart-auto-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  margin-bottom: var(--spacing-sm);
}

.smart-auto-header i {
  font-size: 1.5rem;
  color: var(--accent-primary);
}

.smart-auto-header h4 {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--text-primary);
}

.smart-auto-description {
  margin: 0 0 var(--spacing-lg) 0;
  color: var(--text-secondary);
  font-size: 0.9rem;
  line-height: 1.5;
}

.usage-mode-selector {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  margin-bottom: var(--spacing-lg);
}

.radio-option {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  padding: var(--spacing-md);
  border: 2px solid var(--border-primary);
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: all var(--transition-normal);
  background: transparent;
}

.radio-option:hover {
  border-color: var(--accent-cyan);
  background: rgba(34, 211, 238, 0.05);
}

.radio-option.active {
  border-color: var(--accent-cyan);
  background: rgba(34, 211, 238, 0.1);
}

.radio-option i {
  font-size: 1.5rem;
  color: var(--accent-cyan);
  flex-shrink: 0;
}

.radio-option strong {
  display: block;
  margin-bottom: var(--spacing-xs);
  font-size: 1rem;
  color: var(--text-primary);
}

.radio-option small {
  display: block;
  font-size: 0.875rem;
  color: var(--text-secondary);
}

.smart-auto-button {
  width: 100%;
}

.flash-attention-warning {
  display: flex;
  align-items: flex-start;
  gap: var(--spacing-sm);
  padding: var(--spacing-md);
  margin-bottom: var(--spacing-md);
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: var(--radius-md);
  color: var(--status-error);
}

.flash-attention-warning i {
  font-size: 1.5rem;
  flex-shrink: 0;
  margin-top: 2px;
}

.warning-content strong {
  display: block;
  margin-bottom: var(--spacing-xs);
  font-weight: 600;
}

.warning-content p {
  margin: 0;
  font-size: 0.875rem;
  line-height: 1.5;
}

.expert-info {
  display: flex;
  gap: var(--spacing-sm);
  align-items: center;
  font-size: 0.875rem;
  color: var(--text-secondary);
}

/* Configuration Grid */

.config-search-bar {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  width: 250px;
  transition: width var(--transition-normal);
  margin-bottom: var(--spacing-md);
}

.config-search-bar.search-focused {
  width: 350px;
}

.search-section {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  flex: 0 0 auto;
  width: 200px;
  transition: width var(--transition-normal);
}

.search-section.search-focused {
  width: 300px;
}

.config-search-input {
  width: 100%;
  padding-left: 2rem;
  transition: all var(--transition-normal);
}

.config-search-bar .p-input-icon-left {
  position: relative;
  width: 100%;
}

.config-search-bar .p-input-icon-left i {
  position: absolute;
  left: 0.5rem;
  top: 50%;
  transform: translateY(-50%);
  color: var(--text-secondary);
  z-index: 1;
  font-size: 0.875rem;
}

.section-controls {
  display: flex;
  gap: var(--spacing-sm);
  flex-shrink: 0;
}

.config-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: var(--spacing-xl);
  margin-top: var(--spacing-lg);
}

.config-section {
  background: var(--bg-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow-md);
  transition: all var(--transition-normal);
}

.config-section > summary {
  cursor: pointer;
  padding: var(--spacing-lg);
  list-style: none;
}

.config-section > summary::-webkit-details-marker {
  display: none;
}

.config-section[open] > summary {
  border-bottom: 1px solid var(--border-primary);
}

.config-section:hover {
  box-shadow: var(--shadow-lg);
  transform: translateY(-2px);
}

.section-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: 0;
  color: var(--text-primary);
  font-size: 1.1rem;
  font-weight: 600;
  user-select: none;
}

.title-left {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.section-badge {
  display: inline-block;
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--radius-sm);
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}

.section-badge.essential-badge {
  background: rgba(16, 185, 129, 0.15);
  color: var(--status-success);
}

.section-badge.advanced-badge {
  background: rgba(245, 158, 11, 0.15);
  color: var(--status-warning);
}

.section-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: var(--spacing-lg);
  width: 100%;
  min-width: 0;
  padding: var(--spacing-xl);
}

.config-field {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  min-width: 0;
  width: 100%;
}

.config-field.full-width {
  grid-column: 1 / -1;
}

.config-field label {
  font-weight: 500;
  color: var(--text-primary);
  font-size: 0.9rem;
}

.config-field small {
  color: var(--text-secondary);
  font-size: 0.75rem;
  line-height: 1.3;
}

.inline-validation {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--radius-sm);
  font-size: 0.75rem;
  font-weight: 500;
  margin-top: var(--spacing-xs);
  animation: slideIn 0.2s ease-out;
}

.inline-validation.error {
  background: rgba(239, 68, 68, 0.1);
  color: var(--status-error);
  border: 1px solid rgba(239, 68, 68, 0.2);
}

.inline-validation.warning {
  background: rgba(245, 158, 11, 0.1);
  color: var(--status-warning);
  border: 1px solid rgba(245, 158, 11, 0.2);
}

.inline-validation.success {
  background: rgba(16, 185, 129, 0.1);
  color: var(--status-success);
  border: 1px solid rgba(16, 185, 129, 0.2);
}

.inline-validation i {
  font-size: 0.875rem;
  flex-shrink: 0;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(-4px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Sidebar */
.config-sidebar {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-lg);
  position: sticky;
  top: var(--spacing-xl);
  max-height: calc(100vh - 2 * var(--spacing-xl));
  overflow-y: auto;
  align-self: start;
}

.vram-monitor {
  background: var(--bg-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  padding: var(--spacing-lg);
  box-shadow: var(--shadow-md);
  transition: all var(--transition-normal);
}

.vram-monitor:hover {
  box-shadow: var(--shadow-lg);
  transform: translateY(-2px);
}

.monitor-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-lg);
}

.monitor-header h3 {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  margin: 0;
  color: var(--text-primary);
  font-size: 1.1rem;
}

.monitor-meta {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  font-size: 0.875rem;
  color: var(--text-secondary);
}

.mode-badge {
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  padding: 2px 6px;
  font-weight: 500;
  color: var(--text-primary);
}

.ram-snapshot {
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  padding: 2px 6px;
  font-weight: 500;
  color: var(--text-primary);
}

.monitor-content {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.vram-summary {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.vram-total {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.total-label {
  font-size: 0.875rem;
  color: var(--text-secondary);
}

.total-value {
  font-weight: 700;
  font-size: 1.1rem;
}

.total-value.success {
  color: var(--status-success);
}

.total-value.warning {
  color: var(--status-warning);
}

.vram-progress {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.progress-text {
  font-size: 0.75rem;
  color: var(--text-secondary);
  text-align: center;
}

.vram-breakdown {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.breakdown-item {
  display: flex;
  justify-content: space-between;
  font-size: 0.875rem;
}

.item-label {
  color: var(--text-secondary);
}

.item-value {
  color: var(--text-primary);
  font-weight: 500;
}

.vram-warning {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm);
  background: rgba(245, 158, 11, 0.1);
  border: 1px solid rgba(245, 158, 11, 0.3);
  border-radius: var(--radius-md);
  color: var(--status-warning);
  font-size: 0.875rem;
}

.nvlink-info {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm);
  background: rgba(34, 211, 238, 0.1);
  border: 1px solid rgba(34, 211, 238, 0.3);
  border-radius: var(--radius-md);
  color: var(--accent-cyan);
  font-size: 0.875rem;
}

.monitor-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-xl);
  color: var(--text-muted);
  text-align: center;
}

.monitor-empty i {
  font-size: 2rem;
  color: var(--text-muted);
}

.monitor-loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-xl);
  color: var(--text-secondary);
  text-align: center;
}

.monitor-loading i {
  font-size: 2rem;
  color: var(--accent-primary);
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from {
    transform: rotate(0deg);
  }

  to {
    transform: rotate(360deg);
  }
}

.cpu-mode-info {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.cpu-mode-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  font-weight: 600;
  color: var(--text-primary);
  font-size: 1rem;
}

.cpu-threads-info {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.threads-item {
  display: flex;
  justify-content: space-between;
  font-size: 0.875rem;
}

.threads-label {
  color: var(--text-secondary);
}

.threads-value {
  color: var(--text-primary);
  font-weight: 500;
}

.cpu-performance-tip {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm);
  background: rgba(34, 211, 238, 0.1);
  border: 1px solid rgba(34, 211, 238, 0.3);
  border-radius: var(--radius-md);
  color: var(--accent-cyan);
  font-size: 0.875rem;
}


/* Responsive */
@media (max-width: 1200px) {
  .config-grid {
    grid-template-columns: 1fr;
  }

  .section-grid {
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  }
}

@media (max-width: 768px) {
  .config-layout {
    grid-template-columns: 1fr;
    gap: var(--spacing-lg);
  }

  /* Tablet: Stack memory dashboard */
  .memory-dashboard {
    grid-template-columns: 1fr;
    gap: var(--spacing-lg);
  }

  /* Tablet: Adjust quick start layout */
  .quick-start-content {
    grid-template-columns: 1fr;
    gap: var(--spacing-xl);
  }

  /* Tablet: Full width preset cards */
  .preset-cards {
    width: 100%;
  }

  /* Tablet: Adjust search section */
  .search-section {
    max-width: 100%;
  }

  /* Tablet: 2 column grid for config fields */
  .tab-section {
    grid-template-columns: repeat(2, 1fr);
  }

  .config-search-bar {
    width: 100%;
  }
}

@media (max-width: 600px) {
  .config-layout {
    padding: var(--spacing-md);
  }

  .config-header {
    flex-direction: column;
    align-items: stretch;
    gap: var(--spacing-md);
  }

  .header-actions {
    flex-wrap: wrap;
    justify-content: flex-start;
    gap: var(--spacing-sm);
  }

  .action-buttons {
    flex-wrap: wrap;
  }

  .config-grid {
    grid-template-columns: 1fr;
    gap: var(--spacing-lg);
  }

  /* Mobile: Single column grid for config fields */
  .tab-section {
    grid-template-columns: 1fr;
  }

  .config-search-bar {
    width: 100%;
  }

  /* Memory Dashboard - Stack on mobile */
  .memory-dashboard {
    grid-template-columns: 1fr;
    gap: var(--spacing-md);
  }

  /* Quick Start Modal - Full width on mobile */
  :deep(.quick-start-modal) {
    width: 95vw;
    max-width: none;
  }
  
  :deep(.quick-start-modal .p-dialog-content) {
    max-height: 85vh;
    padding: var(--spacing-lg);
  }
  
  .quick-start-content {
    padding: var(--spacing-lg);
  }

  .quick-start-content {
    grid-template-columns: 1fr;
    gap: var(--spacing-lg);
  }

  .preset-cards {
    flex-direction: column;
  }

  .preset-card {
    min-height: 80px;
    padding: var(--spacing-md);
  }

  .preset-icon {
    font-size: 1.5rem;
  }

  .preset-info h4 {
    font-size: 1rem;
  }

  .preset-info p {
    font-size: 0.85rem;
  }

  /* Smart Auto Section - Full width */
  .smart-auto-section {
    width: 100%;
  }

  .usage-mode-selector {
    flex-direction: column;
    gap: var(--spacing-sm);
  }

  .usage-mode-selector .radio-option {
    min-height: 60px;
    padding: var(--spacing-md);
  }

  /* Config Controls - Full width search */
  .config-controls {
    padding: var(--spacing-md);
  }

  .controls-row {
    flex-direction: column;
    gap: var(--spacing-md);
  }

  .search-section {
    width: 100%;
  }

  .config-search-input {
    width: 100%;
  }

  .section-controls {
    width: 100%;
    justify-content: space-between;
  }

  /* Performance Card - Stack metrics */
  .performance-metrics {
    grid-template-columns: 1fr;
    gap: var(--spacing-md);
  }

  .metric-item {
    padding: var(--spacing-md);
  }

  /* Touch targets - Minimum 44x44px */
  .preset-card,
  .radio-option,
  button,
  .p-button {
    min-height: 44px;
    min-width: 44px;
  }

  /* Larger spacing for touch */
  .config-field {
    gap: var(--spacing-md);
  }

  /* Full width inputs on mobile */
  input[type="number"],
  input[type="text"],
  textarea,
  .p-inputnumber-input,
  .p-inputtext {
    width: 100%;
    min-width: 0;
  }

  /* Adjust section spacing */
  .config-section {
    margin-bottom: var(--spacing-lg);
  }

  /* Collapsible sections - Larger tap area */
  .config-section summary {
    min-height: 48px;
    padding: var(--spacing-md);
  }

  .section-grid {
    grid-template-columns: 1fr;
  }

  .model-meta {
    flex-wrap: wrap;
  }
}

/* Enhanced Focus Styles for Accessibility */
.preset-card:focus {
  outline: 3px solid var(--accent-cyan);
  outline-offset: 2px;
  box-shadow: 0 0 0 3px var(--focus-ring), var(--shadow-md);
}

.preset-card:focus-visible {
  outline: 3px solid var(--accent-cyan);
  outline-offset: 2px;
}

.radio-option:focus {
  outline: 3px solid var(--accent-cyan);
  outline-offset: 2px;
  box-shadow: 0 0 0 3px var(--focus-ring);
}

.radio-option:focus-visible {
  outline: 3px solid var(--accent-cyan);
  outline-offset: 2px;
}

.config-section summary:focus {
  outline: 3px solid var(--accent-cyan);
  outline-offset: 2px;
}

.config-section summary:focus-visible {
  outline: 3px solid var(--accent-cyan);
  outline-offset: 2px;
}

/* Skip to main content link for keyboard navigation */
.skip-link {
  position: absolute;
  top: -40px;
  left: 0;
  background: var(--accent-cyan);
  color: white;
  padding: var(--spacing-sm) var(--spacing-md);
  text-decoration: none;
  z-index: 1000;
  border-radius: var(--radius-md);
}

.skip-link:focus {
  top: var(--spacing-md);
  outline: 3px solid var(--accent-blue);
  outline-offset: 2px;
}

.stacked-bar {
  position: relative;
  width: 100%;
  height: 10px;
  background: var(--bg-secondary);
  border-radius: 5px;
  overflow: hidden;
}

.stacked-bar .bar-current {
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  background: var(--accent-red);
}

.stacked-bar .bar-additional {
  position: absolute;
  top: 0;
  bottom: 0;
  background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
  opacity: 0.8;
  transform-origin: left;
  /* left position is set via inline style dynamically */
}

/* Use inline styles for widths; classes control colors */
.stacked-bar.success {
  box-shadow: inset 0 0 0 1px rgba(16, 185, 129, 0.2);
}

.stacked-bar.warning {
  box-shadow: inset 0 0 0 1px rgba(234, 179, 8, 0.3);
}

/* New Memory Dashboard Styles */
.memory-dashboard {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(400px, 100%), 1fr));
  gap: var(--spacing-lg);
  margin-bottom: var(--spacing-xl);
  width: 100%;
  max-width: 100%;
  box-sizing: border-box;
  min-width: 0;
  overflow-x: visible;
}

@media (max-width: 1400px) {
  .memory-dashboard {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 900px) {
  .memory-dashboard {
    grid-template-columns: 1fr;
    min-width: 0;
  }
}

@media (max-width: 480px) {
  .memory-dashboard {
    grid-template-columns: 1fr;
    gap: var(--spacing-md);
  }
}

.memory-status-card {
  background: var(--gradient-card);
  border: 2px solid var(--border-primary);
  border-radius: var(--radius-xl);
  padding: var(--spacing-xl);
  transition: all var(--transition-normal);
  position: relative;
  overflow: hidden;
  min-width: 0;
  max-width: 100%;
  box-sizing: border-box;
  word-wrap: break-word;
}

.memory-status-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: var(--border-primary);
  transition: all var(--transition-normal);
}

.memory-status-card.status-good::before {
  background: var(--gradient-success);
}

.memory-status-card.status-warning::before {
  background: var(--gradient-warning);
}

.memory-status-card.status-critical::before {
  background: var(--gradient-error);
}

.memory-card-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  margin-bottom: var(--spacing-lg);
}

.memory-status-icon {
  width: 48px;
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: var(--radius-lg);
  font-size: 1.5rem;
  flex-shrink: 0;
}

.memory-status-card.status-good .memory-status-icon {
  background: rgba(16, 185, 129, 0.15);
  color: var(--status-success);
}

.memory-status-card.status-warning .memory-status-icon {
  background: rgba(245, 158, 11, 0.15);
  color: var(--status-warning);
}

.memory-status-card.status-critical .memory-status-icon {
  background: rgba(239, 68, 68, 0.15);
  color: var(--status-error);
}

.memory-card-title {
  flex: 1;
}

.memory-card-title h4 {
  margin: 0 0 var(--spacing-xs) 0;
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--text-primary);
}

.memory-status-badge {
  display: inline-block;
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--radius-sm);
  font-size: 0.875rem;
  font-weight: 600;
  letter-spacing: 0.5px;
}

.memory-status-card.status-good .memory-status-badge {
  background: rgba(16, 185, 129, 0.15);
  color: var(--status-success);
}

.memory-status-card.status-warning .memory-status-badge {
  background: rgba(245, 158, 11, 0.15);
  color: var(--status-warning);
}

.memory-status-card.status-critical .memory-status-badge {
  background: rgba(239, 68, 68, 0.15);
  color: var(--status-error);
}

.memory-status-content {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.memory-usage-display {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  padding: var(--spacing-md);
  background: var(--bg-surface);
  border-radius: var(--radius-md);
}

.usage-item {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  font-size: 0.9rem;
}

.usage-item.total {
  font-weight: 600;
  padding-top: var(--spacing-sm);
  border-top: 1px solid var(--border-primary);
  font-size: 1rem;
}

.usage-label {
  color: var(--text-secondary);
  min-width: 80px;
}

.usage-value {
  color: var(--text-primary);
  font-weight: 500;
  flex: 1;
}

.usage-item.total .usage-value {
  font-weight: 700;
  color: var(--accent-cyan);
}

.usage-fraction {
  color: var(--text-secondary);
  font-size: 0.875rem;
}

.memory-progress-bar {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.progress-label {
  font-size: 0.875rem;
  color: var(--text-secondary);
  text-align: center;
}

.memory-message {
  padding: var(--spacing-md);
  border-radius: var(--radius-md);
  font-size: 0.9rem;
  line-height: 1.5;
  font-weight: 500;
}

.memory-message.status-good {
  background: rgba(16, 185, 129, 0.1);
  color: var(--status-success);
  border: 1px solid rgba(16, 185, 129, 0.2);
}

.memory-message.status-warning {
  background: rgba(245, 158, 11, 0.1);
  color: var(--status-warning);
  border: 1px solid rgba(245, 158, 11, 0.2);
}

.memory-message.status-critical {
  background: rgba(239, 68, 68, 0.1);
  color: var(--status-error);
  border: 1px solid rgba(239, 68, 68, 0.2);
}

.memory-loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: var(--spacing-md);
  padding: var(--spacing-xl);
  color: var(--text-secondary);
}

.memory-loading i {
  font-size: 2rem;
  color: var(--accent-primary);
  animation: spin 1s linear infinite;
}

.advanced-section {
  grid-column: 1 / -1;
  padding: var(--spacing-sm) 0;
}

.advanced-section summary {
  cursor: pointer;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: var(--spacing-sm);
}

.advanced-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: var(--spacing-md);
}
</style>