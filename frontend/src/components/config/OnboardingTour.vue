<template>
  <div v-if="visible && currentStep !== null" class="onboarding-tour">
    <div class="tour-overlay" @click="skipTour"></div>
    <div 
      class="tour-tooltip" 
      :style="tooltipStyle"
      role="dialog"
      aria-label="Onboarding tour"
      aria-live="polite"
    >
      <div class="tour-header">
        <div class="tour-step-indicator">
          <span class="step-current">{{ currentStep + 1 }}</span>
          <span class="step-separator">/</span>
          <span class="step-total">{{ steps.length }}</span>
        </div>
        <Button 
          icon="pi pi-times" 
          @click="skipTour"
          size="small"
          text
          rounded
          severity="secondary"
          aria-label="Skip tour"
          class="tour-close"
        />
      </div>
      
      <div class="tour-content">
        <h3 class="tour-title">{{ currentStepData.title }}</h3>
        <p class="tour-description">{{ currentStepData.content }}</p>
        
        <div v-if="currentStepData.target" class="tour-target-hint">
          <i class="pi pi-arrow-down" aria-hidden="true"></i>
          <span>See highlighted area below</span>
        </div>
      </div>
      
      <div class="tour-footer">
        <Button 
          v-if="currentStep > 0"
          label="Previous" 
          icon="pi pi-arrow-left"
          @click="previousStep"
          size="small"
          text
          severity="secondary"
          aria-label="Go to previous step"
        />
        <div class="tour-actions-right">
          <Button 
            v-if="currentStep < steps.length - 1"
            label="Skip" 
            @click="skipTour"
            size="small"
            text
            severity="secondary"
            aria-label="Skip remaining steps"
          />
          <Button 
            v-if="currentStep < steps.length - 1"
            label="Next" 
            icon="pi pi-arrow-right"
            icon-pos="right"
            @click="nextStep"
            aria-label="Go to next step"
          />
          <Button 
            v-else
            label="Get Started" 
            icon="pi pi-check"
            icon-pos="right"
            @click="completeTour"
            severity="success"
            aria-label="Complete tour"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue'
import Button from 'primevue/button'

const props = defineProps({
  visible: {
    type: Boolean,
    default: false
  },
  steps: {
    type: Array,
    required: true
  }
})

const emit = defineEmits(['complete', 'skip', 'update:visible'])

const currentStep = ref(null)
const tooltipStyle = ref({})
const targetElements = ref([])

const currentStepData = computed(() => {
  if (currentStep.value === null || !props.steps[currentStep.value]) return null
  return props.steps[currentStep.value]
})

watch(() => props.visible, async (newVal) => {
  if (newVal) {
    currentStep.value = 0
    await nextTick()
    await updateTooltipPosition()
  } else {
    currentStep.value = null
    removeHighlights()
  }
})

watch(currentStep, async () => {
  if (currentStep.value !== null) {
    await nextTick()
    await updateTooltipPosition()
  }
})

const updateTooltipPosition = async () => {
  if (!currentStepData.value?.target) {
    tooltipStyle.value = {
      position: 'fixed',
      top: '50%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
      zIndex: 10000
    }
    removeHighlights()
    return
  }

  const targetSelector = currentStepData.value.target
  const targetElement = document.querySelector(targetSelector)
  
  if (!targetElement) {
    console.warn(`Target element not found: ${targetSelector}`)
    return
  }

  // Highlight target element
  highlightTarget(targetElement)

  // Calculate tooltip position
  const rect = targetElement.getBoundingClientRect()
  const tooltipWidth = 400
  const tooltipHeight = 200
  const spacing = 20

  let top = rect.bottom + spacing
  let left = rect.left + (rect.width / 2) - (tooltipWidth / 2)

  // Adjust if tooltip goes off screen
  if (left < spacing) left = spacing
  if (left + tooltipWidth > window.innerWidth - spacing) {
    left = window.innerWidth - tooltipWidth - spacing
  }

  // If tooltip would go below viewport, show above target
  if (top + tooltipHeight > window.innerHeight - spacing) {
    top = rect.top - tooltipHeight - spacing
  }

  // If still doesn't fit, center on screen
  if (top < spacing) {
    top = (window.innerHeight - tooltipHeight) / 2
    left = (window.innerWidth - tooltipWidth) / 2
  }

  tooltipStyle.value = {
    position: 'fixed',
    top: `${top}px`,
    left: `${left}px`,
    zIndex: 10000
  }
}

const highlightTarget = (element) => {
  removeHighlights()
  
  element.classList.add('tour-highlight')
  targetElements.value.push(element)
  
  // Scroll element into view if needed
  element.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' })
}

const removeHighlights = () => {
  targetElements.value.forEach(el => {
    el.classList.remove('tour-highlight')
  })
  targetElements.value = []
}

const nextStep = () => {
  if (currentStep.value < props.steps.length - 1) {
    currentStep.value++
  } else {
    completeTour()
  }
}

const previousStep = () => {
  if (currentStep.value > 0) {
    currentStep.value--
  }
}

const skipTour = () => {
  emit('skip')
  emit('update:visible', false)
  removeHighlights()
}

const completeTour = () => {
  emit('complete')
  emit('update:visible', false)
  removeHighlights()
  
  // Store completion in localStorage
  localStorage.setItem('model-config-onboarding-completed', 'true')
}

// Handle window resize
const handleResize = () => {
  if (props.visible && currentStep.value !== null) {
    updateTooltipPosition()
  }
}

onMounted(() => {
  window.addEventListener('resize', handleResize)
  window.addEventListener('scroll', handleResize, true)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  window.removeEventListener('scroll', handleResize, true)
  removeHighlights()
})

// Expose removeHighlights for cleanup
defineExpose({
  removeHighlights
})
</script>

<style scoped>
.onboarding-tour {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 9999;
}

.tour-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(4px);
}

.tour-tooltip {
  position: fixed;
  width: 400px;
  max-width: calc(100vw - 40px);
  background: var(--bg-card);
  border: 2px solid var(--accent-cyan);
  border-radius: var(--radius-xl);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5), 0 0 0 4px rgba(34, 211, 238, 0.2);
  padding: var(--spacing-lg);
  animation: fadeInScale 0.3s ease-out;
}

@keyframes fadeInScale {
  from {
    opacity: 0;
    transform: scale(0.9);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}

.tour-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-md);
  padding-bottom: var(--spacing-md);
  border-bottom: 1px solid var(--border-primary);
}

.tour-step-indicator {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--text-secondary);
}

.step-current {
  color: var(--accent-cyan);
  font-size: 1.1rem;
}

.tour-close {
  padding: var(--spacing-xs);
}

.tour-content {
  margin-bottom: var(--spacing-lg);
}

.tour-title {
  margin: 0 0 var(--spacing-sm) 0;
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--text-primary);
}

.tour-description {
  margin: 0;
  font-size: 0.95rem;
  color: var(--text-secondary);
  line-height: 1.5;
}

.tour-target-hint {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  margin-top: var(--spacing-md);
  padding: var(--spacing-sm);
  background: rgba(34, 211, 238, 0.1);
  border-radius: var(--radius-sm);
  font-size: 0.85rem;
  color: var(--accent-cyan);
}

.tour-target-hint i {
  animation: bounce 1s ease-in-out infinite;
}

@keyframes bounce {
  0%, 100% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(4px);
  }
}

.tour-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--spacing-md);
}

.tour-actions-right {
  display: flex;
  gap: var(--spacing-sm);
  margin-left: auto;
}

/* Global style for highlighting target elements - using deep selector */
:deep(.tour-highlight) {
  position: relative;
  z-index: 10000 !important;
  box-shadow: 0 0 0 4px rgba(34, 211, 238, 0.5), 0 0 20px rgba(34, 211, 238, 0.3) !important;
  border-radius: var(--radius-md);
  animation: pulseHighlight 2s ease-in-out infinite;
}

@media (max-width: 600px) {
  .tour-tooltip {
    width: calc(100vw - 20px);
    max-width: none;
  }
}
</style>

