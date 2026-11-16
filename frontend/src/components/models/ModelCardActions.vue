<template>
  <div class="model-actions" :class="{ 'model-actions--compact': compact }">
    <div class="model-actions__row">
      <Tag
        v-if="needsSetup"
        class="model-actions__setup-tag"
        value="Needs setup"
        severity="warn"
        :title="setupHint"
        v-tooltip.top="setupHint"
      />

      <Button
        ref="primaryRef"
        class="model-actions__primary"
        :label="primary.label"
        :icon="primary.icon"
        :severity="primary.severity"
        :size="compact ? 'small' : undefined"
        :loading="primary.loading"
        :disabled="primary.disabled"
        :aria-busy="primary.busy ? 'true' : undefined"
        :aria-label="primary.ariaLabel"
        :aria-describedby="primary.describedby"
        v-tooltip.top="primary.tooltip"
        @click="onPrimary"
      />

      <span class="model-actions__slot model-actions__slot--stop">
        <Button
          v-if="showStop"
          ref="stopRef"
          icon="pi pi-stop-circle"
          text
          severity="warning"
          size="small"
          :aria-label="stopLabel"
          v-tooltip.top="stopTooltip"
          @click="activate('stop', $event)"
        />
      </span>

      <Button
        v-if="isAudio && runtimeState.key !== 'running'"
        :label="compact ? undefined : 'Open Audio'"
        icon="pi pi-volume-up"
        text
        severity="secondary"
        size="small"
        :aria-label="`Open audio for ${label}`"
        v-tooltip.top="'Open audio'"
        @click="activate('audio', $event)"
      />

      <Button
        v-if="!needsSetup"
        :label="compact ? undefined : 'Configure'"
        icon="pi pi-cog"
        text
        severity="secondary"
        size="small"
        :aria-label="`Configure ${label}`"
        v-tooltip.top="'Configure'"
        @click="activate('configure', $event)"
      />

      <details class="model-actions__menu" @click.stop>
        <summary :aria-label="`More actions for ${label}`">
          <i v-if="compact" class="pi pi-ellipsis-v" aria-hidden="true" />
          <template v-else>More</template>
        </summary>
        <button
          v-if="needsSetup"
          type="button"
          class="model-actions__start-defaults"
          :aria-label="`Start ${label} with defaults`"
          title="Start without reviewing settings. The engine and context defaults are used."
          @click="activate('start', $event)"
        >
          Start with defaults
        </button>
        <ModelWarmupButton
          :model="model"
          :disabled="runtimeState.key === 'starting' || runtimeState.key === 'stopping'"
        />
        <button type="button" class="model-actions__delete" @click="activate('delete', $event)">
          Delete
        </button>
      </details>
    </div>

    <p
      v-if="errorId && errorText"
      :id="errorId"
      class="model-actions__error"
      :title="errorText"
      role="alert"
    >
      {{ errorText }}
    </p>
  </div>
</template>

<script setup>
import { computed, nextTick, ref, useId, watch } from 'vue'
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import { modelNeedsSetup, modelRuntimeState } from '@/utils/modelPresentation'
import { isAudioPackage } from '@/utils/audioModel'
import ModelWarmupButton from '@/components/ModelWarmupButton.vue'

const props = defineProps({
  model: {
    type: Object,
    required: true,
  },
  /** Display name for accessible labels; falls back to the quantization name. */
  name: {
    type: String,
    default: '',
  },
  isStarting: {
    type: Boolean,
    default: false,
  },
  isStopping: {
    type: Boolean,
    default: false,
  },
  /** Icon-only secondary actions, used by the compact list layout. */
  compact: {
    type: Boolean,
    default: false,
  },
  /** Use inside clickable rows/headers so clicks do not bubble. */
  stopPropagation: {
    type: Boolean,
    default: false,
  },
})

const emit = defineEmits(['start', 'stop', 'configure', 'connect', 'audio', 'delete', 'refresh'])

const primaryRef = ref(null)
const stopRef = ref(null)
const generatedId = useId()
const errorId = computed(() =>
  props.model?.last_error ? `model-error-${generatedId.replace(/[^a-zA-Z0-9_-]/g, '-')}` : '',
)
const label = computed(
  () =>
    props.name || props.model?.quantization || props.model?.name || String(props.model?.id || ''),
)
const runtimeState = computed(() =>
  modelRuntimeState(props.model, {
    isStarting: props.isStarting,
    isStopping: props.isStopping,
  }),
)
const isAudio = computed(() => isAudioPackage(props.model))
const needsSetup = computed(() => modelNeedsSetup(props.model, runtimeState.value))
const setupHint = computed(
  () =>
    'This model was downloaded but its engine and context settings have not been reviewed. Set it up before starting.',
)
const errorText = computed(() => String(props.model?.last_error || '').trim())
const showStop = computed(
  () => runtimeState.value.key === 'running' || runtimeState.value.key === 'starting',
)
const stopLabel = computed(() => `Stop ${label.value}`)
const stopTooltip = computed(() =>
  runtimeState.value.key === 'starting' ? 'Cancel starting this model' : 'Stop this model',
)

/** One primary action per state; the label and action change together. */
const primary = computed(() => {
  const state = runtimeState.value
  if (state.key === 'starting')
    return {
      label: 'Starting…',
      icon: 'pi pi-play',
      loading: true,
      disabled: true,
      busy: true,
      tooltip: 'Starting. Stop cancels this start.',
      ariaLabel: `${label.value} is starting`,
    }
  if (state.key === 'stopping')
    return {
      label: 'Stopping…',
      icon: 'pi pi-stop',
      loading: true,
      disabled: true,
      busy: true,
      tooltip: 'Stopping the model.',
      ariaLabel: `${label.value} is stopping`,
    }
  if (needsSetup.value && ['stopped', 'unknown', 'stale'].includes(state.key))
    return {
      label: 'Set up',
      icon: 'pi pi-cog',
      severity: 'info',
      action: 'configure',
      tooltip: setupHint.value,
      ariaLabel: `Set up ${label.value}`,
    }
  if (state.key === 'stopped') {
    return {
      label: 'Start',
      icon: 'pi pi-play',
      severity: 'success',
      action: 'start',
      tooltip: 'Start this model',
      ariaLabel: `Start ${label.value}`,
    }
  }
  if (state.key === 'running')
    return isAudio.value
      ? {
          label: 'Open Audio',
          icon: 'pi pi-volume-up',
          action: 'audio',
          tooltip: 'Open audio',
          ariaLabel: `Open audio for ${label.value}`,
        }
      : {
          label: 'Connect',
          icon: 'pi pi-link',
          action: 'connect',
          tooltip: 'Connect an app to this model',
          ariaLabel: `Connect ${label.value}`,
        }
  if (state.key === 'error')
    return {
      label: 'Retry start',
      icon: 'pi pi-replay',
      action: 'start',
      tooltip: 'Start this model again',
      ariaLabel: `Retry start for ${label.value}`,
      describedby: errorText.value ? errorId.value : undefined,
    }
  return {
    label: 'Refresh status',
    icon: 'pi pi-refresh',
    action: 'refresh',
    tooltip: 'Check the current status again',
    ariaLabel: `Refresh status for ${label.value}`,
  }
})

function focusTarget(componentRef) {
  const element = componentRef?.$el
  if (element && typeof element.focus === 'function') element.focus({ preventScroll: true })
}

/**
 * Remember the action the user activated so focus can follow the state change:
 * Start → Stop (while starting) → Connect / Retry start; Stop → Start.
 */
const pendingFocus = ref('')

function activate(action, event) {
  if (props.stopPropagation) event?.stopPropagation()
  if (['start', 'stop', 'refresh'].includes(action)) pendingFocus.value = action
  emit(action, action === 'connect' ? props.model : props.model?.id)
}

function onPrimary(event) {
  if (!primary.value.action) return
  activate(primary.value.action, event)
}

watch(
  () => runtimeState.value.key,
  async (key) => {
    const action = pendingFocus.value
    if (!action) return
    await nextTick()
    if (action === 'start') {
      if (key === 'starting') {
        focusTarget(stopRef.value)
        return
      }
      pendingFocus.value = ''
      focusTarget(primaryRef.value)
      return
    }
    if (action === 'stop') {
      if (key === 'stopping') return
      pendingFocus.value = ''
      focusTarget(primaryRef.value)
      return
    }
    pendingFocus.value = ''
    if (key !== 'stopping') focusTarget(primaryRef.value)
  },
)
</script>

<style scoped>
.model-actions {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 0.25rem;
  min-width: 15rem;
}

.model-actions__row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: 0.4rem;
  width: 100%;
  min-height: 2.75rem;
}

/* Reserve width so swapping Start/Starting…/Connect/Refresh status never reflows the row. */
.model-actions__primary {
  min-width: 10rem;
}

/* Keep the tag beside the primary action without growing the row much. */
.model-actions__setup-tag {
  white-space: nowrap;
  font-size: var(--text-xs, 0.75rem);
  /* Keep the setup tag at the start of the action row while the buttons stay
     right-aligned; without this the shrink-wrapped tag floats mid-row. */
  margin-right: auto;
}

.model-actions__primary :deep(.p-button-label) {
  white-space: nowrap;
}

.model-actions__slot {
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

/* Keep the icon-only Stop position reserved in states that do not offer it. */
.model-actions__slot--stop {
  min-width: 2.75rem;
}

.model-actions--compact .model-actions__primary {
  min-width: 10.5rem;
}

.model-actions__error {
  margin: 0;
  max-width: 22rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--status-error, #f87171);
  font-size: var(--text-xs, 0.75rem);
  line-height: 1.25;
  text-align: right;
}

.model-actions__menu {
  position: relative;
}

.model-actions__menu summary {
  list-style: none;
  cursor: pointer;
  min-height: 2rem;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  padding: 0.2rem 0.55rem;
  font-size: 0.8rem;
  color: var(--text-primary);
  background: var(--bg-secondary);
}

.model-actions__menu summary::-webkit-details-marker {
  display: none;
}

.model-actions__menu button {
  display: block;
  width: 100%;
  margin-top: 0.25rem;
  color: var(--text-primary);
  background: var(--bg-secondary);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  cursor: pointer;
  font: inherit;
  padding: 0.25rem 0.5rem;
}

.model-actions__menu .model-actions__delete {
  color: var(--status-error);
}

@media (max-width: 600px) {
  .model-actions {
    min-width: 0;
    align-items: flex-start;
  }

  .model-actions__row {
    flex-wrap: nowrap;
    justify-content: flex-start;
    gap: 0.25rem;
  }

  .model-actions__primary {
    min-width: 0;
  }

  .model-actions__slot--stop {
    min-width: 0;
  }

  .model-actions__error {
    max-width: 100%;
    text-align: left;
  }
}
</style>
