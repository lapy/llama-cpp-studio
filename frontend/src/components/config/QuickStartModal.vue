<template>
  <Dialog 
    :visible="visible" 
    modal 
    :closable="true"
    :dismissableMask="true"
    :draggable="false"
    class="quick-start-modal"
    @update:visible="$emit('update:visible', $event)"
    @hide="$emit('update:visible', false)"
  >
    <template #header>
      <div class="quick-start-modal-header">
        <div class="quick-start-icon">🚀</div>
        <div>
          <h3>Quick Start</h3>
          <p>Choose a preset, use the wizard, or let Smart Auto optimize for you</p>
        </div>
      </div>
    </template>

    <div class="quick-start-content">
      <div class="preset-cards">
        <div 
          class="preset-card wizard-card" 
          role="button"
          tabindex="0"
          aria-label="Configuration Wizard - Guided 3-step setup for new users"
          @click="handleWizardClick"
          @keydown.enter="handleWizardClick"
          @keydown.space.prevent="handleWizardClick"
        >
          <div class="preset-icon" aria-hidden="true">✨</div>
          <div class="preset-info">
            <h4>Configuration Wizard</h4>
            <p>Guided 3-step setup for new users</p>
          </div>
        </div>
        <div 
          class="preset-card" 
          role="button"
          tabindex="0"
          aria-label="Coding preset - Low temperature, high precision for code generation"
          @click="handlePresetClick('coding')"
          @keydown.enter="handlePresetClick('coding')"
          @keydown.space.prevent="handlePresetClick('coding')"
        >
          <div class="preset-icon" aria-hidden="true">💻</div>
          <div class="preset-info">
            <h4>Coding</h4>
            <p>Low temperature, high precision for code generation</p>
          </div>
        </div>
        <div 
          class="preset-card" 
          role="button"
          tabindex="0"
          aria-label="Chat preset - Balanced settings for natural conversation"
          @click="handlePresetClick('conversational')"
          @keydown.enter="handlePresetClick('conversational')"
          @keydown.space.prevent="handlePresetClick('conversational')"
        >
          <div class="preset-icon" aria-hidden="true">💬</div>
          <div class="preset-info">
            <h4>Chat</h4>
            <p>Balanced settings for natural conversation</p>
          </div>
        </div>
      </div>
      
      <div class="smart-auto-section">
        <div class="smart-auto-header">
          <i class="pi pi-bolt"></i>
          <h4>Smart Auto Configuration</h4>
        </div>
        <p class="smart-auto-description">Automatically optimize settings based on your hardware and use case</p>
        
        <div class="usage-mode-selector" role="radiogroup" aria-label="Usage mode selection">
          <div 
            class="radio-option" 
            role="radio"
            :aria-checked="localUsageMode === 'single_user'"
            tabindex="0"
            aria-label="Single User mode - Sequential requests, maximum context"
            @click="localUsageMode = 'single_user'"
            @keydown.enter="localUsageMode = 'single_user'"
            @keydown.space.prevent="localUsageMode = 'single_user'"
            :class="{ active: localUsageMode === 'single_user' }"
          >
            <i class="pi pi-user" aria-hidden="true"></i>
            <div>
              <strong>Single User</strong>
              <small>Sequential requests, maximum context</small>
            </div>
          </div>
          <div 
            class="radio-option" 
            role="radio"
            :aria-checked="localUsageMode === 'multi_user'"
            tabindex="0"
            aria-label="Multi User Server mode - Parallel requests, optimized batching"
            @click="localUsageMode = 'multi_user'"
            @keydown.enter="localUsageMode = 'multi_user'"
            @keydown.space.prevent="localUsageMode = 'multi_user'"
            :class="{ active: localUsageMode === 'multi_user' }"
          >
            <i class="pi pi-users" aria-hidden="true"></i>
            <div>
              <strong>Multi User Server</strong>
              <small>Parallel requests, optimized batching</small>
            </div>
          </div>
        </div>
        
        <Button 
          label="Generate Optimal Config" 
          icon="pi pi-bolt" 
          @click="handleSmartAuto"
          :loading="autoConfigLoading" 
          size="large" 
          class="smart-auto-button"
          aria-label="Generate optimal configuration automatically based on hardware and use case"
        />
      </div>
    </div>
    
    <template #footer>
      <Button label="Close" icon="pi pi-times" @click="$emit('update:visible', false)" severity="secondary" />
    </template>
  </Dialog>
</template>

<script setup>
// Vue
import { ref, watch } from 'vue'

// PrimeVue
import Dialog from 'primevue/dialog'
import Button from 'primevue/button'

const props = defineProps({
  visible: {
    type: Boolean,
    default: false
  },
  autoConfigLoading: {
    type: Boolean,
    default: false
  },
  smartAutoUsageMode: {
    type: String,
    default: 'single_user'
  }
})

const emit = defineEmits(['update:visible', 'wizard', 'preset', 'smart-auto'])

const localUsageMode = ref(props.smartAutoUsageMode)

watch(() => props.smartAutoUsageMode, (newVal) => {
  localUsageMode.value = newVal
})

watch(() => props.visible, (newVal) => {
  if (newVal) {
    localUsageMode.value = props.smartAutoUsageMode
  }
})

const handleWizardClick = () => {
  emit('wizard')
  emit('update:visible', false)
}

const handlePresetClick = (preset) => {
  emit('preset', preset)
  emit('update:visible', false)
}

const handleSmartAuto = () => {
  emit('smart-auto', localUsageMode.value)
}
</script>

<style scoped>
.quick-start-modal-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
}

.quick-start-icon {
  font-size: 2rem;
  line-height: 1;
}

.quick-start-modal-header h3 {
  margin: 0 0 var(--spacing-xs) 0;
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--text-primary);
}

.quick-start-modal-header p {
  margin: 0;
  color: var(--text-secondary);
  font-size: 0.875rem;
}

.quick-start-content {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xl);
}

.preset-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: var(--spacing-md);
}

.preset-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-lg);
  background: var(--bg-surface);
  border: 2px solid var(--border-primary);
  border-radius: var(--radius-lg);
  cursor: pointer;
  transition: all var(--transition-normal);
  text-align: center;
}

.preset-card:hover {
  border-color: var(--accent-cyan);
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}

.preset-card:focus {
  outline: 2px solid var(--accent-cyan);
  outline-offset: 2px;
}

.preset-icon {
  font-size: 2.5rem;
  line-height: 1;
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
}

.smart-auto-section {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  padding: var(--spacing-lg);
  background: var(--bg-surface);
  border-radius: var(--radius-lg);
  border: 1px solid var(--border-primary);
}

.smart-auto-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.smart-auto-header i {
  font-size: 1.5rem;
  color: var(--accent-cyan);
}

.smart-auto-header h4 {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--text-primary);
}

.smart-auto-description {
  margin: 0;
  color: var(--text-secondary);
  font-size: 0.875rem;
}

.usage-mode-selector {
  display: flex;
  gap: var(--spacing-md);
}

.radio-option {
  flex: 1;
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-md);
  background: var(--bg-card);
  border: 2px solid var(--border-primary);
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: all var(--transition-normal);
}

.radio-option:hover {
  border-color: var(--accent-cyan);
}

.radio-option.active {
  border-color: var(--accent-cyan);
  background: var(--bg-surface);
}

.radio-option:focus {
  outline: 2px solid var(--accent-cyan);
  outline-offset: 2px;
}

.radio-option i {
  font-size: 1.5rem;
  color: var(--accent-cyan);
}

.radio-option strong {
  display: block;
  margin-bottom: var(--spacing-xs);
  color: var(--text-primary);
  font-weight: 600;
}

.radio-option small {
  display: block;
  color: var(--text-secondary);
  font-size: 0.875rem;
}

.smart-auto-button {
  width: 100%;
}
</style>
