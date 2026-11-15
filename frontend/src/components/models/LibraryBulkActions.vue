<template>
  <div v-if="models.length || busy || lastResult" class="library-bulk">
    <div
      v-if="models.length"
      class="library-bulk__selection"
      role="region"
      aria-label="Bulk model actions"
    >
      <span class="library-bulk__count">{{ selectionText }}</span>
      <Button
        label="Start selected"
        icon="pi pi-play"
        size="small"
        :disabled="!startable.length || busy"
        :loading="busy && activeAction === 'start'"
        @click="run('start')"
      />
      <Button
        label="Stop selected"
        icon="pi pi-stop"
        size="small"
        severity="warning"
        :disabled="!stoppable.length || busy"
        :loading="busy && activeAction === 'stop'"
        @click="run('stop')"
      />
      <Button
        label="Clear selection"
        size="small"
        text
        severity="secondary"
        :disabled="busy"
        @click="$emit('clear')"
      />
    </div>
    <p v-if="busy" class="library-bulk__progress" role="status">{{ progressText }}</p>
    <div v-else-if="lastResult" class="library-bulk__result" role="status">
      <span>{{ resultText }}</span>
      <Button label="Dismiss" size="small" text severity="secondary" @click="lastResult = null" />
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import Button from 'primevue/button'
import { useToast } from 'primevue/usetoast'
import { useModelStore } from '@/stores/models'
import { formatAxiosDetail } from '@/utils/apiErrors'

const props = defineProps({
  /** Selected quantizations, restricted by the library view to visible models. */
  models: { type: Array, default: () => [] },
  /** Model ids with a single-model start/stop already in flight. */
  busyIds: { type: Array, default: () => [] },
})
const emit = defineEmits(['complete', 'clear'])

const modelStore = useModelStore()
const toast = useToast()
const busy = ref(false)
const activeAction = ref('')
const lastResult = ref(null)
const progress = ref({ total: 0, done: 0, failed: 0, skipped: 0 })

const busySet = computed(() => new Set((props.busyIds || []).map((id) => String(id))))

function modelStatus(model) {
  return String(model?.status || model?.run_state || '').toLowerCase()
}

/** Mirrors the per-model start/stop button availability. */
function eligibleFor(model, action) {
  const id = String(model?.id || '')
  if (!id || busySet.value.has(id)) return false
  if (action === 'stop') return Boolean(model.is_active)
  return !model.is_active && modelStatus(model) !== 'loading'
}

const startable = computed(() => props.models.filter((model) => eligibleFor(model, 'start')))
const stoppable = computed(() => props.models.filter((model) => eligibleFor(model, 'stop')))

const selectionText = computed(() => {
  const parts = [`${props.models.length} selected`]
  if (startable.value.length) parts.push(`${startable.value.length} can start`)
  if (stoppable.value.length) parts.push(`${stoppable.value.length} can stop`)
  if (!startable.value.length && !stoppable.value.length) parts.push('no eligible actions')
  return parts.join(' · ')
})

const progressText = computed(() => {
  const verb = activeAction.value === 'stop' ? 'Stopping' : 'Starting'
  const completed = progress.value.done + progress.value.failed + progress.value.skipped
  const position = Math.min(completed + 1, progress.value.total)
  return `${verb} ${position} of ${progress.value.total}…`
})

function modelLabel(model) {
  return model?.quantization || model?.name || String(model?.id || 'model')
}

function describeResult(result) {
  const verb = result.action === 'start' ? 'Started' : 'Stopped'
  const parts = [
    `${verb} ${result.done} of ${result.total} selected model${result.total === 1 ? '' : 's'}`,
  ]
  if (result.failed) parts.push(`${result.failed} failed`)
  if (result.skipped) parts.push(`${result.skipped} skipped`)
  return `${parts.join(' · ')}.`
}

const resultText = computed(() => (lastResult.value ? describeResult(lastResult.value) : ''))

/**
 * Run one bulk action sequentially. Re-checking each model against the current
 * selection keeps a run from touching models that disappeared or became
 * ineligible after it started (for example after a filter change or refresh).
 */
async function run(action) {
  if (busy.value) return
  const targets = action === 'start' ? [...startable.value] : [...stoppable.value]
  if (!targets.length) return
  busy.value = true
  activeAction.value = action
  lastResult.value = null
  progress.value = { total: targets.length, done: 0, failed: 0, skipped: 0 }
  const verb = action === 'start' ? 'start' : 'stop'

  for (const target of targets) {
    const current = props.models.find((model) => String(model?.id) === String(target.id))
    if (!current || !eligibleFor(current, action)) {
      progress.value.skipped += 1
      continue
    }
    try {
      if (action === 'start') await modelStore.startModel(current.id)
      else await modelStore.stopModel(current.id)
      progress.value.done += 1
    } catch (error) {
      progress.value.failed += 1
      toast.add({
        severity: 'error',
        summary: `Could not ${verb} ${modelLabel(current)}`,
        detail: formatAxiosDetail(error),
        life: 5000,
      })
    }
  }

  const result = {
    action,
    total: progress.value.total,
    done: progress.value.done,
    failed: progress.value.failed,
    skipped: progress.value.skipped,
  }
  lastResult.value = result
  busy.value = false
  activeAction.value = ''
  toast.add({
    severity: result.failed || result.skipped ? 'warn' : 'success',
    summary: action === 'start' ? 'Bulk start finished' : 'Bulk stop finished',
    detail: describeResult(result),
    life: 5000,
  })
  emit('complete', result)
}
</script>

<style scoped>
.library-bulk {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
  padding: 0.5rem 0.75rem;
  border: 1px solid var(--border-secondary);
  border-radius: var(--radius-md);
  background: var(--bg-tertiary);
}

.library-bulk__selection {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
  width: 100%;
}

.library-bulk__count {
  font-size: var(--text-sm);
  color: var(--text-secondary);
}

.library-bulk__progress,
.library-bulk__result {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
  margin: 0;
  font-size: var(--text-sm);
  color: var(--text-secondary);
}
</style>
