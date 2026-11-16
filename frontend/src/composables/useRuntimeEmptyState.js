import { computed, ref } from 'vue'

const SAVED_PARAM_LABELS = {
  ctx_size: 'Context size',
  n_gpu_layers: 'GPU layers',
  batch_size: 'Batch size',
  ubatch_size: 'Micro-batch size',
  flash_attn: 'Flash attention',
  cache_type_k: 'KV cache K type',
  cache_type_v: 'KV cache V type',
  tensor_split: 'Tensor split',
  main_gpu: 'Main GPU',
  cont_batching: 'Continuous batching',
}

function savedParamLabel(key) {
  if (SAVED_PARAM_LABELS[key]) return SAVED_PARAM_LABELS[key]
  return key.replace(/[_-]+/g, ' ').replace(/\b\w/g, (character) => character.toUpperCase())
}

const RUNTIME_META_KEYS = new Set([
  'engine',
  'engines',
  'runtime_env',
  'custom_args',
  'set_params_by_id',
  'swap_aliases',
  'request_options',
  'model_alias',
  'last_reviewed_fingerprint',
  'default_voice_preset',
  'voice_presets',
  'speech_defaults',
  'transcription_defaults',
  'task_defaults',
])

/**
 * State for the Runtime tab's empty state.
 *
 * The engine parameter catalog only exists after the engine binary has been
 * scanned. Until then we still show the engine's saved values (read from the
 * stored engine section and the form) and offer an explicit rescan, rather than
 * rendering a blank tab.
 */
export function useRuntimeEmptyState({ config, engineOptions, fetchParamRegistry, router }) {
  const selectedEngineLabel = computed(
    () =>
      engineOptions.value.find((option) => option.value === config.value.engine)?.label ||
      config.value.engine ||
      'this engine',
  )

  const savedRuntimeEntries = computed(() => {
    const entries = new Map()
    const section = config.value.engines?.[config.value.engine]
    for (const source of [section, config.value]) {
      if (!source || typeof source !== 'object') continue
      for (const [key, value] of Object.entries(source)) {
        if (entries.has(key) || RUNTIME_META_KEYS.has(key)) continue
        if (value == null || !['string', 'number', 'boolean'].includes(typeof value)) continue
        if (typeof value === 'string' && !value.trim()) continue
        entries.set(key, { key, label: savedParamLabel(key), value })
      }
    }
    return [...entries.values()].slice(0, 12)
  })

  const runtimeRescanning = ref(false)

  async function rescanRuntimeParams() {
    if (runtimeRescanning.value) return
    runtimeRescanning.value = true
    try {
      await fetchParamRegistry(config.value.engine, { rescan: true })
    } finally {
      runtimeRescanning.value = false
    }
  }

  function openEnginesPage() {
    router.push('/engines')
  }

  return {
    selectedEngineLabel,
    savedRuntimeEntries,
    runtimeRescanning,
    rescanRuntimeParams,
    openEnginesPage,
  }
}
