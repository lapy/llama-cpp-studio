<template>
  <div class="slider-input">
    <div class="slider-container">
      <div class="slider-track">
        <div class="slider-fill" :style="{ width: fillPercentage + '%' }"></div>
        <!-- Markers for preset values -->
        <div v-if="markers && markers.length > 0" class="slider-markers">
          <div
            v-for="(marker, index) in markers"
            :key="index"
            class="slider-marker"
            :class="getMarkerClass(marker)"
            :style="{ left: getMarkerPosition(marker) + '%' }"
            :title="marker.label"
          >
            <span class="marker-dot"></span>
            <span v-if="showMarkerLabels" class="marker-label">{{ marker.label }}</span>
          </div>
        </div>
        <!-- Recommended value indicator -->
        <div
          v-if="recommended !== null && recommended !== undefined"
          class="slider-recommended"
          :style="{ left: getRecommendedPosition() + '%' }"
          :title="`Recommended: ${formatValue(recommended)}`"
        >
          <span class="recommended-dot"></span>
          <span class="recommended-line"></span>
        </div>
        <input
          type="range"
          :min="min"
          :max="max"
          :step="step"
          :value="modelValue"
          @input="updateValue"
          class="slider"
          :class="{ 'slider-disabled': disabled, 'near-recommended': isNearRecommended }"
          :disabled="disabled"
        />
      </div>
      <div class="slider-labels">
        <span class="min-label">{{ formatValue(min) }}</span>
        <span class="max-label">{{ formatValue(max) }}</span>
      </div>
    </div>
    <div class="value-display">
      <input
        type="number"
        :min="min"
        :max="max"
        :step="step"
        :value="modelValue"
        @input="updateValue"
        class="number-input"
        :class="{ 'input-disabled': disabled, 'recommended-value': isAtRecommended }"
        :disabled="disabled"
        :maxFractionDigits="maxFractionDigits"
      />
      <span v-if="isAtRecommended && recommended !== null" class="recommended-badge" title="At recommended value">
        ✓
      </span>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  modelValue: {
    type: [Number, String],
    required: true
  },
  min: {
    type: Number,
    default: 0
  },
  max: {
    type: Number,
    default: 100
  },
  step: {
    type: Number,
    default: 1
  },
  maxFractionDigits: {
    type: Number,
    default: 0
  },
  disabled: {
    type: Boolean,
    default: false
  },
  markers: {
    type: Array,
    default: () => []
  },
  recommended: {
    type: Number,
    default: null
  },
  showMarkerLabels: {
    type: Boolean,
    default: true
  }
})

const emit = defineEmits(['update:modelValue', 'input'])

const updateValue = (event) => {
  const value = parseFloat(event.target.value)
  emit('update:modelValue', value)
  emit('input', value)
}

const formatValue = (value) => {
  if (props.maxFractionDigits === 0) {
    return Math.round(value).toString()
  }
  return value.toFixed(props.maxFractionDigits)
}

const fillPercentage = computed(() => {
  const range = props.max - props.min
  const value = parseFloat(props.modelValue) - props.min
  return Math.min(100, Math.max(0, (value / range) * 100))
})

const getMarkerPosition = (marker) => {
  const range = props.max - props.min
  const value = parseFloat(marker.value) - props.min
  
  // Calculate linear position
  const linearPos = (value / range) * 100
  
  // Apply correction: browser's native range slider thumb positioning is non-linear
  // Browser compensates for thumb width (20px), creating position-dependent offset
  // Observed: 19.598% needs 1.0205x, 36.842% needs 0.980x, 88.89% needs 0.955x
  // Quadratic fit: factor = 2.713e-05 * pos² - 0.003889 * pos + 1.0863
  const correctionFactor = 2.713e-05 * linearPos * linearPos - 0.003889 * linearPos + 1.0863
  
  return Math.min(100, Math.max(0, linearPos * correctionFactor))
}

const getMarkerClass = (marker) => {
  return marker.color ? `marker-${marker.color}` : ''
}

const getRecommendedPosition = () => {
  if (props.recommended === null || props.recommended === undefined) return 0
  const range = props.max - props.min
  const value = parseFloat(props.recommended) - props.min
  
  // Calculate linear position
  const linearPos = (value / range) * 100
  
  // Apply correction: browser's native range slider thumb positioning is non-linear
  // Browser compensates for thumb width (20px), creating position-dependent offset
  // Observed: 19.598% needs 1.0205x, 36.842% needs 0.980x, 88.89% needs 0.955x
  // Quadratic fit: factor = 2.713e-05 * pos² - 0.003889 * pos + 1.0863
  const correctionFactor = 2.713e-05 * linearPos * linearPos - 0.003889 * linearPos + 1.0863
  
  return Math.min(100, Math.max(0, linearPos * correctionFactor))
}

const isAtRecommended = computed(() => {
  if (props.recommended === null || props.recommended === undefined) return false
  const current = parseFloat(props.modelValue)
  const rec = parseFloat(props.recommended)
  const threshold = props.step || 1
  return Math.abs(current - rec) < threshold
})

const isNearRecommended = computed(() => {
  if (props.recommended === null || props.recommended === undefined) return false
  if (isAtRecommended.value) return true
  const current = parseFloat(props.modelValue)
  const rec = parseFloat(props.recommended)
  const range = props.max - props.min
  const threshold = range * 0.05 // Within 5% of range
  return Math.abs(current - rec) < threshold
})
</script>

<style scoped>
.slider-input {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  width: 100%;
}

.slider-container {
  position: relative;
  width: 100%;
}

.slider-track {
  position: relative;
  width: 100%;
  height: 32px;
  background: transparent;
  border-radius: 4px;
  display: flex;
  align-items: center;
}

.slider-track::before {
  content: '';
  position: absolute;
  top: 50%;
  left: 0;
  right: 0;
  height: 8px;
  transform: translateY(-50%);
  background: var(--bg-secondary);
  border-radius: 4px;
  border: 2px solid var(--border-secondary);
  box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
  z-index: 0;
}

.slider-fill {
  position: absolute;
  top: 50%;
  left: 0;
  height: 8px;
  transform: translateY(-50%);
  background: var(--gradient-primary);
  border-radius: 4px;
  pointer-events: none;
  z-index: 1;
  transition: width var(--transition-normal);
}

.slider {
  position: absolute;
  top: 50%;
  left: 0;
  width: 100%;
  height: 8px;
  transform: translateY(-50%);
  border-radius: 4px;
  background: transparent;
  outline: none;
  -webkit-appearance: none;
  appearance: none;
  cursor: pointer;
  transition: all var(--transition-normal);
  z-index: 2;
}

.slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: var(--gradient-primary);
  cursor: pointer;
  border: 2px solid var(--bg-primary);
  box-shadow: var(--shadow-md);
  transition: all var(--transition-normal);
  z-index: 3;
}

.slider::-webkit-slider-thumb:hover {
  transform: scale(1.2);
  box-shadow: var(--shadow-lg);
}

.slider::-moz-range-thumb {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: var(--gradient-primary);
  cursor: pointer;
  border: 2px solid var(--bg-primary);
  box-shadow: var(--shadow-md);
  transition: all var(--transition-normal);
  z-index: 3;
}

.slider::-moz-range-thumb:hover {
  transform: scale(1.2);
  box-shadow: var(--shadow-lg);
}

.slider::-webkit-slider-track {
  background: var(--bg-tertiary);
  height: 8px;
  border-radius: 4px;
  border: 1px solid var(--border-primary);
}

.slider::-moz-range-track {
  background: var(--bg-tertiary);
  height: 8px;
  border-radius: 4px;
  border: 1px solid var(--border-primary);
}

.slider-disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.slider-disabled::-webkit-slider-thumb {
  cursor: not-allowed;
}

.slider-disabled::-moz-range-thumb {
  cursor: not-allowed;
}

.slider-labels {
  display: flex;
  justify-content: space-between;
  margin-top: var(--spacing-xs);
  font-size: 0.75rem;
  color: var(--text-muted);
}

.value-display {
  display: flex;
  justify-content: center;
}

.number-input {
  width: 80px;
  padding: var(--spacing-xs) var(--spacing-sm);
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  color: var(--text-primary);
  text-align: center;
  font-size: 0.9rem;
  transition: all var(--transition-normal);
}

.number-input:focus {
  outline: none;
  border-color: var(--accent-cyan);
  box-shadow: 0 0 0 2px var(--focus-ring);
}

.input-disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.slider-markers {
  position: absolute;
  top: 10px;
  left: 0;
  right: 0;
  height: 100%;
  pointer-events: none;
  z-index: 2;
}

.slider-marker {
  position: absolute;
  top: 50%;
  transform: translate(-50%, -50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
}

.marker-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.6);
  border: 1px solid rgba(255, 255, 255, 0.8);
  box-shadow: 0 0 4px rgba(0, 0, 0, 0.3);
}

.marker-label {
  font-size: 0.65rem;
  color: var(--text-secondary);
  white-space: nowrap;
  margin-top: 4px;
  font-weight: 500;
}

.marker-blue .marker-dot {
  background: #3b82f6;
  border-color: #60a5fa;
}

.marker-green .marker-dot {
  background: #22c55e;
  border-color: #4ade80;
}

.marker-purple .marker-dot {
  background: #a855f7;
  border-color: #c084fc;
}

.marker-yellow .marker-dot {
  background: #f59e0b;
  border-color: #fbbf24;
}

.slider-recommended {
  position: absolute;
  top: 0;
  transform: translate(-50%, 0);
  height: 100%;
  pointer-events: none;
  z-index: 3;
}

.recommended-dot {
  position: absolute;
  top: 50%;
  left: 0;
  transform: translateY(-50%);
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--status-success);
  border: 2px solid var(--bg-primary);
  box-shadow: 0 0 8px rgba(16, 185, 129, 0.4), 0 2px 4px rgba(0, 0, 0, 0.2);
  animation: pulse 2s ease-in-out infinite;
}

.recommended-line {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 2px;
  height: 8px;
  background: var(--status-success);
  opacity: 0.5;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
    transform: translateY(-50%) scale(1);
  }
  50% {
    opacity: 0.7;
    transform: translateY(-50%) scale(1.1);
  }
}

.slider.near-recommended {
  opacity: 1;
}

.slider.near-recommended::-webkit-slider-thumb {
  box-shadow: 0 0 16px rgba(16, 185, 129, 0.4), 0 4px 8px rgba(0, 0, 0, 0.2);
}

.recommended-value {
  border-color: var(--status-success);
  background: rgba(16, 185, 129, 0.1);
}

.recommended-badge {
  position: absolute;
  right: -24px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--status-success);
  font-size: 1rem;
  font-weight: bold;
}
</style>
