<template>
  <div class="slider-input">
    <div class="slider-container">
      <div class="slider-track">
        <div class="slider-fill" :style="{ width: fillPercentage + '%' }"></div>
        <input
          type="range"
          :min="min"
          :max="max"
          :step="step"
          :value="modelValue"
          @input="updateValue"
          class="slider"
          :class="{ 'slider-disabled': disabled }"
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
        :class="{ 'input-disabled': disabled }"
        :disabled="disabled"
        :maxFractionDigits="maxFractionDigits"
      />
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
  height: 8px;
  background: #1a1f2e;
  border-radius: 4px;
  border: 2px solid #2d3748;
  box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.4);
}

.slider-fill {
  position: absolute;
  top: 0;
  left: 0;
  height: 8px;
  background: linear-gradient(90deg, #22d3ee, #3b82f6);
  border-radius: 4px;
  pointer-events: none;
  z-index: 1;
  transition: width var(--transition-normal);
  box-shadow: 0 0 8px rgba(34, 211, 238, 0.3);
}

.slider {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 8px;
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
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: linear-gradient(135deg, #22d3ee, #3b82f6);
  cursor: pointer;
  border: 3px solid #ffffff;
  box-shadow: 0 0 12px rgba(34, 211, 238, 0.5), 0 4px 8px rgba(0, 0, 0, 0.3);
  transition: all var(--transition-normal);
  z-index: 3;
  position: relative;
}

.slider::-webkit-slider-thumb:hover {
  transform: scale(1.15);
  box-shadow: 0 0 16px rgba(34, 211, 238, 0.7), 0 6px 12px rgba(0, 0, 0, 0.4);
}

.slider::-moz-range-thumb {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: linear-gradient(135deg, #22d3ee, #3b82f6);
  cursor: pointer;
  border: 3px solid #ffffff;
  box-shadow: 0 0 12px rgba(34, 211, 238, 0.5), 0 4px 8px rgba(0, 0, 0, 0.3);
  transition: all var(--transition-normal);
  z-index: 3;
  position: relative;
}

.slider::-moz-range-thumb:hover {
  transform: scale(1.15);
  box-shadow: 0 0 16px rgba(34, 211, 238, 0.7), 0 6px 12px rgba(0, 0, 0, 0.4);
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

/* Progress bar effect */
.slider::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  height: 6px;
  background: var(--gradient-primary);
  border-radius: 3px;
  width: calc((var(--value) - var(--min)) / (var(--max) - var(--min)) * 100%);
  pointer-events: none;
}
</style>
