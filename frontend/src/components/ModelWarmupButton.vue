<template>
  <Button
    v-if="available"
    label="Warmup"
    icon="pi pi-bolt"
    text
    severity="secondary"
    size="small"
    :aria-label="`Warm up ${model.name || model.quantization || model.id}`"
    :loading="busy"
    :disabled="disabled || busy || loading"
    v-tooltip.top="'Send a short inference request to warm the running model'"
    @click.stop="warmup"
  />
</template>

<script setup>
import { computed, ref } from 'vue'
import axios from 'axios'
import Button from 'primevue/button'
import { useToast } from 'primevue/usetoast'

const props = defineProps({
  model: { type: Object, required: true },
  disabled: { type: Boolean, default: false },
})
const toast = useToast()
const busy = ref(false)
const available = computed(
  () =>
    props.model.is_active &&
    props.model.format !== 'audio_cpp' &&
    (props.model.config?.engine || props.model.engine) !== 'audio_cpp',
)
const loading = computed(
  () =>
    (props.model.status || props.model.run_state) === 'loading' ||
    ['stale', 'unreachable'].includes(props.model.runtime_quality),
)
async function warmup() {
  if (!available.value || props.disabled || loading.value || busy.value) return
  busy.value = true
  try {
    const { data } = await axios.post(
      `/api/models/${encodeURIComponent(props.model.id)}/connect-test`,
    )
    if (!(data.status_code >= 200 && data.status_code < 300))
      throw new Error(`Inference returned HTTP ${data.status_code}`)
    toast.add({ severity: 'success', summary: 'Model warmed up', life: 2500 })
  } catch (error) {
    toast.add({
      severity: 'error',
      summary: 'Warmup failed',
      detail: error?.response?.data?.detail || error.message,
      life: 5000,
    })
  } finally {
    busy.value = false
  }
}
</script>
