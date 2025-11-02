<template>
  <Dialog 
    v-model:visible="visible" 
    modal 
    :closable="true"
    :dismissableMask="true"
    :draggable="false"
    class="config-wizard-dialog"
    @hide="$emit('close')"
    @touchstart="handleTouchStart"
    @touchmove="handleTouchMove"
    @touchend="handleTouchEnd"
  >
    <template #header>
      <div class="wizard-header">
        <i class="pi pi-magic"></i>
        <h2>Configuration Wizard</h2>
      </div>
    </template>

    <div class="wizard-content">
      <div class="wizard-steps">
        <div 
          v-for="(step, index) in steps" 
          :key="index"
          class="wizard-step"
          :class="{ active: currentStep === index, completed: currentStep > index }"
        >
          <div class="step-number">{{ index + 1 }}</div>
          <div class="step-label">{{ step.label }}</div>
        </div>
      </div>

      <div class="wizard-step-content">
        <!-- Step 1: Use Case Selection -->
        <div v-if="currentStep === 0" class="step-panel">
          <h3>What are you using this model for?</h3>
          <p class="step-description">Select your primary use case to get optimized settings</p>
          
          <div class="use-case-grid">
            <div 
              v-for="useCase in useCases" 
              :key="useCase.id"
              class="use-case-card"
              :class="{ active: wizardData.useCase === useCase.id }"
              @click="wizardData.useCase = useCase.id"
            >
              <div class="use-case-icon">{{ useCase.icon }}</div>
              <h4>{{ useCase.title }}</h4>
              <p>{{ useCase.description }}</p>
            </div>
          </div>
        </div>

        <!-- Step 2: Resource Allocation -->
        <div v-if="currentStep === 1" class="step-panel">
          <h3>Resource Allocation</h3>
          <p class="step-description">Balance between speed and quality based on your hardware</p>
          
          <div class="resource-section">
            <div class="hardware-info">
              <i class="pi pi-desktop"></i>
              <div>
                <strong>Detected Hardware</strong>
                <p v-if="gpuInfo">
                  {{ gpuInfo.name || `GPU ${gpuInfo.device_count || 0}` }}
                  <span v-if="gpuInfo.total_vram">({{ formatFileSize(gpuInfo.total_vram) }} VRAM)</span>
                </p>
                <p v-else>CPU-only mode</p>
              </div>
            </div>
            
            <div class="speed-quality-slider">
              <label>Speed ←→ Quality</label>
              <SliderInput 
                v-model="wizardData.speedQuality" 
                :min="0" 
                :max="100" 
                :step="1"
                :markers="[
                  { value: 0, label: 'Max Speed', color: 'blue' },
                  { value: 50, label: 'Balanced', color: 'green' },
                  { value: 100, label: 'Max Quality', color: 'purple' }
                ]"
              />
              <div class="slider-description">
                <template v-if="wizardData.speedQuality < 34">
                  <strong>Max Speed Mode</strong>
                  <p>Lower context size, larger batches, optimized GPU layers for maximum throughput</p>
                </template>
                <template v-else-if="wizardData.speedQuality < 67">
                  <strong>Balanced Mode</strong>
                  <p>Optimal balance between speed and quality with moderate context and batch sizes</p>
                </template>
                <template v-else>
                  <strong>Max Quality Mode</strong>
                  <p>Higher context size, better quantization, full GPU offloading for maximum quality</p>
                </template>
              </div>
            </div>
          </div>
        </div>

        <!-- Step 3: Review & Preview -->
        <div v-if="currentStep === 2" class="step-panel">
          <h3>Review Generated Configuration</h3>
          <p class="step-description">Review the settings we've generated for you. You can fine-tune them later.</p>
          
          <div class="config-preview">
            <div class="preview-header">
              <div class="preview-summary">
                <h4>Configuration Summary</h4>
                <div class="summary-badges">
                  <span class="badge">Use Case: {{ getUseCaseTitle(wizardData.useCase) }}</span>
                  <span class="badge">Mode: {{ wizardData.speedQuality < 50 ? 'Speed' : 'Quality' }}</span>
                </div>
              </div>
            </div>
            
            <div class="preview-settings">
              <div class="preview-item" v-for="(value, key) in generatedConfig" :key="key">
                <span class="setting-label">{{ formatSettingName(key) }}:</span>
                <span class="setting-value">{{ formatSettingValue(value) }}</span>
              </div>
            </div>
            
            <div class="preview-impact">
              <div class="impact-item">
                <i class="pi pi-tachometer-alt"></i>
                <span>Estimated Performance: {{ estimatedPerformance }}</span>
              </div>
              <div class="impact-item">
                <i class="pi pi-memory"></i>
                <span>VRAM Usage: {{ estimatedVramUsage }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <template #footer>
      <div class="wizard-footer">
        <Button 
          v-if="currentStep > 0"
          label="Back" 
          icon="pi pi-arrow-left" 
          @click="currentStep--"
          text
        />
        <Button 
          v-if="currentStep < 2"
          label="Next" 
          icon="pi pi-arrow-right" 
          iconPos="right"
          @click="currentStep++"
          :disabled="!canProceed"
        />
        <Button 
          v-if="currentStep === 2"
          label="Apply & Start" 
          icon="pi pi-check" 
          @click="applyConfig"
          severity="success"
          :loading="applying"
        />
        <Button 
          v-if="currentStep === 2"
          label="Advanced Mode" 
          icon="pi pi-sliders-h" 
          @click="goToAdvanced"
          severity="secondary"
          outlined
        />
      </div>
    </template>
  </Dialog>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import Dialog from 'primevue/dialog'
import Button from 'primevue/button'
import SliderInput from '@/components/SliderInput.vue'

const props = defineProps({
  modelVisible: {
    type: Boolean,
    default: false
  },
  model: {
    type: Object,
    default: null
  },
  gpuInfo: {
    type: Object,
    default: () => ({})
  },
  modelLayerInfo: {
    type: Object,
    default: null
  }
})

const emit = defineEmits(['close', 'apply-config', 'go-to-advanced'])

const visible = computed({
  get: () => props.modelVisible,
  set: (val) => {
    if (!val) emit('close')
  }
})

const currentStep = ref(0)
const applying = ref(false)

const steps = [
  { label: 'Use Case' },
  { label: 'Resources' },
  { label: 'Review' }
]

// Touch gesture handling for swipe to dismiss
const touchStartX = ref(0)
const touchStartY = ref(0)
const touchThreshold = 50 // Minimum swipe distance

const handleTouchStart = (e) => {
  if (e.touches && e.touches.length > 0) {
    touchStartX.value = e.touches[0].clientX
    touchStartY.value = e.touches[0].clientY
  }
}

const handleTouchMove = (e) => {
  // Prevent default to allow swipe detection
  if (e.touches && e.touches.length > 0) {
    const deltaX = e.touches[0].clientX - touchStartX.value
    const deltaY = e.touches[0].clientY - touchStartY.value
    
    // If swiping down significantly, allow dismiss
    if (deltaY > touchThreshold && Math.abs(deltaX) < Math.abs(deltaY)) {
      e.preventDefault()
    }
  }
}

const handleTouchEnd = (e) => {
  if (e.changedTouches && e.changedTouches.length > 0) {
    const deltaX = e.changedTouches[0].clientX - touchStartX.value
    const deltaY = e.changedTouches[0].clientY - touchStartY.value
    
    // Swipe down to dismiss
    if (deltaY > touchThreshold && Math.abs(deltaX) < Math.abs(deltaY)) {
      emit('close')
    }
  }
  
  touchStartX.value = 0
  touchStartY.value = 0
}

const useCases = [
  {
    id: 'chat',
    icon: '💬',
    title: 'Chat/Conversation',
    description: 'Natural conversations and Q&A',
    preset: 'conversational'
  },
  {
    id: 'code',
    icon: '💻',
    title: 'Code Generation',
    description: 'Code completion and generation',
    preset: 'coding'
  },
  {
    id: 'creative',
    icon: '✍️',
    title: 'Creative Writing',
    description: 'Stories, articles, creative content',
    preset: 'creative'
  },
  {
    id: 'analysis',
    icon: '🔍',
    title: 'Analysis/Research',
    description: 'Document analysis and research',
    preset: 'conversational'
  }
]

const wizardData = ref({
  useCase: null,
  speedQuality: 50 // 0-100 scale
})

const canProceed = computed(() => {
  if (currentStep.value === 0) {
    return wizardData.value.useCase !== null
  }
  return true
})

// Configuration is now generated by the backend API
// This computed property is kept for preview/estimation purposes only
const generatedConfig = computed(() => {
  // Return a minimal preview config based on selections
  // Actual config will come from backend API
  const useCase = useCases.find(uc => uc.id === wizardData.value.useCase)
  const speedQuality = wizardData.value.speedQuality || 50
  
  return {
    // Placeholder values for preview only
    ctx_size: useCase?.id === 'analysis' ? 16384 : useCase?.id === 'code' ? 8192 : 4096,
    batch_size: speedQuality < 34 ? 512 : speedQuality < 67 ? 384 : 256,
    temp: useCase?.id === 'code' ? 0.3 : useCase?.id === 'creative' ? 1.2 : 0.8,
    temperature: useCase?.id === 'code' ? 0.3 : useCase?.id === 'creative' ? 1.2 : 0.8,
    n_gpu_layers: props.gpuInfo?.device_count > 0 ? (props.modelLayerInfo?.layer_count || 32) : 0
  }
})

const estimatedPerformance = computed(() => {
  const speedQuality = wizardData.value.speedQuality || 50
  
  // Rough estimate based on speed/quality setting
  // Actual performance will depend on backend-generated config
  if (speedQuality < 34) {
    return 'Very Fast (~60+ tok/s)'
  } else if (speedQuality < 67) {
    return 'Fast (~40-50 tok/s)'
  } else {
    return 'Moderate (~25-35 tok/s)'
  }
})

const estimatedVramUsage = computed(() => {
  if (!props.gpuInfo?.total_vram) return 'N/A'
  const hasGPU = props.gpuInfo?.device_count > 0
  if (!hasGPU) return 'CPU-only (0 VRAM)'
  
  // Rough estimate - actual VRAM will be calculated by backend
  const speedQuality = wizardData.value.speedQuality || 50
  const qualityFactor = speedQuality / 100
  
  // Estimate based on quality factor (quality uses more VRAM)
  const baseEstimate = props.gpuInfo.total_vram * (0.6 + (qualityFactor * 0.3)) // 60-90% of total VRAM
  const percentage = Math.round(((baseEstimate / props.gpuInfo.total_vram) * 100))
  
  return `~${formatFileSize(baseEstimate)} (${percentage}%) / ${formatFileSize(props.gpuInfo.total_vram)}`
})

const getUseCaseTitle = (useCaseId) => {
  const useCase = useCases.find(uc => uc.id === useCaseId)
  return useCase?.title || 'Unknown'
}

const formatSettingName = (key) => {
  const names = {
    n_gpu_layers: 'GPU Layers',
    ctx_size: 'Context Size',
    batch_size: 'Batch Size',
    temp: 'Temperature',
    top_k: 'Top-K',
    top_p: 'Top-P',
    repeat_penalty: 'Repeat Penalty'
  }
  return names[key] || key
}

const formatSettingValue = (value) => {
  if (typeof value === 'number') {
    if (value >= 1000) return value.toLocaleString()
    if (value < 1) return value.toFixed(2)
    return Math.round(value)
  }
  return value
}

const formatFileSize = (bytes) => {
  if (!bytes && bytes !== 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

const applyConfig = async () => {
  applying.value = true
  try {
    // Call backend Smart Auto API instead of generating config locally
    if (!props.model?.id) {
      console.error('No model ID available')
      return
    }
    
    const params = new URLSearchParams()
    if (wizardData.value.speedQuality !== undefined) {
      params.append('speed_quality', wizardData.value.speedQuality.toString())
    }
    if (wizardData.value.useCase) {
      params.append('use_case', wizardData.value.useCase)
    }
    // Default to single_user usage mode for wizard
    params.append('usage_mode', 'single_user')
    
    const url = `/api/models/${props.model.id}/smart-auto${params.toString() ? '?' + params.toString() : ''}`
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    })
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Failed to generate configuration' }))
      throw new Error(error.detail || 'Failed to generate configuration')
    }
    
    const backendConfig = await response.json()
    
    // Emit the backend-generated configuration
    emit('apply-config', backendConfig)
    visible.value = false
  } catch (error) {
    console.error('Error generating configuration:', error)
    // Fall back to local generation if API fails (graceful degradation)
    emit('apply-config', generatedConfig.value)
    visible.value = false
  } finally {
    applying.value = false
  }
}

const goToAdvanced = () => {
  emit('go-to-advanced')
  visible.value = false
}

// Reset wizard when closed
watch(visible, (newVal) => {
  if (!newVal) {
    currentStep.value = 0
    wizardData.value = { useCase: null, speedQuality: 50 }
  }
})
</script>

<style scoped>
.config-wizard-dialog {
  max-width: 800px;
}

.wizard-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
}

.wizard-header i {
  font-size: 2rem;
  color: var(--accent-cyan);
}

.wizard-header h2 {
  margin: 0;
  font-size: 1.75rem;
  background: linear-gradient(135deg, #22d3ee, #3b82f6);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.wizard-content {
  padding: var(--spacing-lg) 0;
}

.wizard-steps {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: var(--spacing-md);
  margin-bottom: var(--spacing-xl);
  padding: var(--spacing-lg) 0;
}

.wizard-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--spacing-sm);
  position: relative;
  flex: 1;
  max-width: 150px;
}

.wizard-step:not(:last-child)::after {
  content: '';
  position: absolute;
  top: 20px;
  left: calc(50% + 30px);
  width: calc(100% - 60px);
  height: 2px;
  background: var(--border-primary);
  z-index: 0;
}

.wizard-step.completed:not(:last-child)::after {
  background: var(--accent-cyan);
}

.step-number {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--bg-surface);
  border: 2px solid var(--border-primary);
  color: var(--text-secondary);
  font-weight: 600;
  font-size: 1.1rem;
  position: relative;
  z-index: 1;
  transition: all var(--transition-normal);
}

.wizard-step.active .step-number {
  background: var(--accent-cyan);
  border-color: var(--accent-cyan);
  color: white;
  transform: scale(1.1);
  box-shadow: 0 0 0 4px rgba(34, 211, 238, 0.2);
}

.wizard-step.completed .step-number {
  background: var(--accent-green);
  border-color: var(--accent-green);
  color: white;
}

.step-label {
  font-size: 0.9rem;
  color: var(--text-secondary);
  font-weight: 500;
  text-align: center;
}

.wizard-step.active .step-label {
  color: var(--accent-cyan);
  font-weight: 600;
}

.wizard-step-content {
  min-height: 400px;
}

.step-panel {
  padding: var(--spacing-lg);
}

.step-panel h3 {
  margin: 0 0 var(--spacing-sm) 0;
  font-size: 1.5rem;
  color: var(--text-primary);
}

.step-description {
  margin: 0 0 var(--spacing-xl) 0;
  color: var(--text-secondary);
  font-size: 0.95rem;
}

.use-case-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: var(--spacing-lg);
}

.use-case-card {
  background: var(--bg-surface);
  border: 2px solid var(--border-primary);
  border-radius: var(--radius-lg);
  padding: var(--spacing-lg);
  cursor: pointer;
  transition: all var(--transition-normal);
  text-align: center;
}

.use-case-card:hover {
  border-color: var(--accent-cyan);
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}

.use-case-card.active {
  border-color: var(--accent-primary);
  background: rgba(34, 211, 238, 0.1);
  box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.2);
}

.use-case-icon {
  font-size: 3rem;
  margin-bottom: var(--spacing-sm);
}

.use-case-card h4 {
  margin: 0 0 var(--spacing-xs) 0;
  font-size: 1.1rem;
  color: var(--text-primary);
}

.use-case-card p {
  margin: 0;
  font-size: 0.875rem;
  color: var(--text-secondary);
  line-height: 1.4;
}

.resource-section {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xl);
}

.hardware-info {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  padding: var(--spacing-lg);
  background: var(--bg-surface);
  border-radius: var(--radius-lg);
  border: 1px solid var(--border-primary);
}

.hardware-info i {
  font-size: 2rem;
  color: var(--accent-cyan);
}

.hardware-info strong {
  display: block;
  margin-bottom: var(--spacing-xs);
  color: var(--text-primary);
}

.hardware-info p {
  margin: 0;
  color: var(--text-secondary);
  font-size: 0.9rem;
}

.speed-quality-slider {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.speed-quality-slider label {
  font-weight: 600;
  color: var(--text-primary);
}

.slider-description {
  padding: var(--spacing-md);
  background: var(--bg-surface);
  border-radius: var(--radius-md);
  color: var(--text-secondary);
  font-size: 0.9rem;
  text-align: center;
}

.slider-description strong {
  display: block;
  color: var(--text-primary);
  font-size: 1rem;
  margin-bottom: var(--spacing-xs);
}

.slider-description p {
  margin: var(--spacing-xs) 0 0 0;
  color: var(--text-secondary);
  font-size: 0.875rem;
}

.config-preview {
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  padding: var(--spacing-lg);
}

.preview-header {
  margin-bottom: var(--spacing-lg);
  padding-bottom: var(--spacing-lg);
  border-bottom: 1px solid var(--border-primary);
}

.preview-header h4 {
  margin: 0 0 var(--spacing-md) 0;
  color: var(--text-primary);
}

.summary-badges {
  display: flex;
  gap: var(--spacing-sm);
  flex-wrap: wrap;
}

.badge {
  padding: var(--spacing-xs) var(--spacing-sm);
  background: rgba(34, 211, 238, 0.15);
  color: var(--accent-cyan);
  border-radius: var(--radius-sm);
  font-size: 0.875rem;
  font-weight: 600;
}

.preview-settings {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: var(--spacing-md);
  margin-bottom: var(--spacing-lg);
}

.preview-item {
  display: flex;
  justify-content: space-between;
  padding: var(--spacing-sm);
  background: var(--bg-primary);
  border-radius: var(--radius-sm);
}

.setting-label {
  font-weight: 500;
  color: var(--text-secondary);
}

.setting-value {
  font-weight: 600;
  color: var(--text-primary);
}

.preview-impact {
  display: flex;
  gap: var(--spacing-lg);
  padding-top: var(--spacing-lg);
  border-top: 1px solid var(--border-primary);
}

.impact-item {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  color: var(--text-primary);
}

.impact-item i {
  color: var(--accent-cyan);
}

.wizard-footer {
  display: flex;
  justify-content: space-between;
  gap: var(--spacing-md);
}
</style>

