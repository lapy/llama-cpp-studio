import { computed, ref } from 'vue'

export const AUDIO_NESTED_SCOPE_KEYS = {
  load_option: 'load_options',
  session_option: 'session_options',
  request_option: 'request_options',
}

export const AUDIO_STUDIO_KNOWN_KEYS = [
  'lazy_load',
  'voice_presets',
  'default_voice_preset',
  'speech_defaults',
  'transcription_defaults',
  'task_defaults',
  'last_reviewed_fingerprint',
]

export const AUDIO_REQUEST_DEFAULTS_KEYS = [
  'speech_defaults',
  'transcription_defaults',
  'task_defaults',
]

/** Keys produced when server help documents ``--load-option key=value`` syntax. */
export function isBogusAudioConfigKey(key) {
  const name = String(key || '')
  return !name || name === 'key=value' || name.includes('=')
}

export function requestDefaultsHasContent(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  for (const [key, item] of Object.entries(value)) {
    if (key === 'options') {
      if (item && typeof item === 'object' && Object.keys(item).some((k) => {
        const nested = item[k]
        return nested != null && nested !== ''
      })) {
        return true
      }
      continue
    }
    if (item != null && item !== '' && !(Array.isArray(item) && item.length === 0)) {
      return true
    }
  }
  return false
}

/** Clear non-active request-defaults objects so save validation does not fail after family/task switches. */
export function pruneStaleAudioRequestDefaults(config, keepKey) {
  if (!config || typeof config !== 'object') return false
  const keep = String(keepKey || '').trim()
  let changed = false
  for (const key of AUDIO_REQUEST_DEFAULTS_KEYS) {
    if (key === keep) continue
    if (!requestDefaultsHasContent(config[key])) continue
    config[key] = {}
    changed = true
  }
  if (keep !== 'speech_defaults') {
    if (config.voice_presets && typeof config.voice_presets === 'object'
      && Object.keys(config.voice_presets).length) {
      config.voice_presets = {}
      changed = true
    }
    if (config.default_voice_preset != null && config.default_voice_preset !== '') {
      config.default_voice_preset = null
      changed = true
    }
  }
  return changed
}

export function instructionsPolicyHint(policy, {
  source = '',
  vocabulary = null,
} = {}) {
  const key = String(policy || '').trim().toLowerCase()
  const fromEngine = String(source || '').trim().toLowerCase() === 'engine'
  const origin = fromEngine ? 'from engine' : 'Studio fallback'
  const vocab = Array.isArray(vocabulary)
    ? vocabulary.map((item) => String(item || '').trim()).filter(Boolean)
    : []
  const vocabHint = vocab.length
    ? ` Vocabulary examples: ${vocab.slice(0, 8).join(', ')}.`
    : ''
  if (key === 'soft_tags') {
    return (
      `Use comma-separated voice attributes (for example: female, young adult, british accent). `
      + `(${origin})${vocabHint}`
    )
  }
  if (key === 'caption_option') {
    return `This model uses options.caption for voice design, not the instructions field. (${origin})`
  }
  if (key === 'text_prefix') {
    return (
      'Put style cues at the start of input text (for example: "(calm) Hello world"), '
      + `not in instructions. (${origin})`
    )
  }
  if (key === 'openai_instruct') {
    return `Natural-language style instructions are accepted for this speech model. (${origin})`
  }
  if (key === 'none') {
    return `This model does not accept an instructions field in request defaults. (${origin})`
  }
  return ''
}

export const LAZY_LOAD_PARAM = {
  key: 'lazy_load',
  label: 'Lazy load weights',
  type: 'bool',
  scope: 'process',
  supported: true,
  default: false,
  description:
    'Defer loading model weights until the first request. Written to the audio.cpp sidecar and applied when llama-swap starts this model.',
}

export function isOptionalConfigParam(param) {
  return param?.required !== true
}

export function coerceAudioParamValue(param, value) {
  const scalarType = param.scalar_type || param.type
  if (scalarType === 'int') {
    if (typeof value === 'number' && !Number.isNaN(value)) return Math.trunc(value)
    if (typeof value === 'string' && value.trim() !== '') {
      const parsed = parseInt(value, 10)
      if (!Number.isNaN(parsed)) return parsed
    }
  }
  if (scalarType === 'float') {
    if (typeof value === 'number' && !Number.isNaN(value)) return value
    if (typeof value === 'string' && value.trim() !== '') {
      const parsed = parseFloat(value)
      if (!Number.isNaN(parsed)) return parsed
    }
  }
  return value
}

export function defaultValueForAudioParam(param) {
  if (param.value_kind === 'repeatable') {
    return Array.isArray(param.default) ? [...param.default] : []
  }
  if (isDelimitedEnumParam(param)) {
    return normalizeCsvEnumValue(param.default, param)
  }
  if (param.value_kind === 'flag') return param.negative_flag ? null : true
  return param.default ?? null
}

export function isDelimitedEnumParam(param) {
  return ['csv_enum', 'semicolon_enum'].includes(param.value_kind) || param.type === 'multiselect'
}

export function delimitedEnumSeparator(param) {
  return param.value_kind === 'semicolon_enum' ? ';' : ','
}

export function normalizeCsvEnumValue(value, param) {
  if (Array.isArray(value)) {
    return value.filter((v) => v != null && v !== '')
  }
  if (value == null || value === '') return []
  if (typeof value === 'string') {
    return value.split(delimitedEnumSeparator(param)).map((s) => s.trim()).filter(Boolean)
  }
  return [value]
}

export function jsonParamDisplay(value) {
  if (value == null || value === '') return ''
  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (!trimmed) return ''
    try {
      return JSON.stringify(JSON.parse(trimmed), null, 2)
    } catch {
      return value
    }
  }
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value, null, 2)
    } catch {
      return ''
    }
  }
  return String(value)
}

export function paramDescriptionTooltip(param) {
  const parts = [param.description].filter(Boolean)
  if (param.primary_flag) {
    parts.push(`Primary flag: ${param.primary_flag}`)
  }
  if (param.negative_flag) {
    parts.push(`Negative flag: ${param.negative_flag}`)
  } else if (param.flags?.length) {
    parts.push(`CLI: ${param.flags.join(', ')}`)
  }
  return parts.join('\n\n') || param.label || param.key
}

export function paramMatchesSearch(param, queryRaw, hideUnsupported = false) {
  if (hideUnsupported && param.supported === false) return false
  const raw = queryRaw.trim().toLowerCase()
  if (!raw) return false
  const hay = [
    param.label || '',
    param.key || '',
    param.description || '',
    ...(param.flags || []),
  ]
    .join(' ')
    .toLowerCase()
  const tokens = raw.split(/\s+/).filter(Boolean)
  return tokens.every((t) => hay.includes(t))
}

export function fieldStorageHint(field, { presetField = false, viaProxy = true } = {}) {
  if (presetField) {
    return 'Written to the audio.cpp sidecar as part of a voice preset'
  }
  if (field.nested || field.options_key) {
    const key = field.options_key || field.key
    return viaProxy
      ? `Injected as options.${key} via llama-swap setParams`
      : `Stored in options.${key}`
  }
  if (field.type === 'path') {
    return 'Path relative to the bundle directory on the server'
  }
  return viaProxy
    ? 'Injected via llama-swap setParams when you apply config'
    : ''
}

function coercePreviewNumber(value, asInt) {
  if (typeof value === 'number' && !Number.isNaN(value)) {
    return asInt ? Math.trunc(value) : value
  }
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = asInt ? parseInt(value, 10) : parseFloat(value)
    if (!Number.isNaN(parsed)) return parsed
  }
  return null
}

/** Mirror backend audio_request_defaults_to_swap_set_params type normalization (paths resolve on apply). */
export function buildSwapSetParamsPreview(defaultsKey, defaults) {
  if (!defaults || typeof defaults !== 'object' || Array.isArray(defaults)) return null
  const out = {}
  if (defaultsKey === 'speech_defaults') {
    const intKeys = new Set(['seed', 'top_k', 'max_tokens', 'num_inference_steps'])
    const floatKeys = new Set([
      'temperature', 'top_p', 'guidance_scale', 'repetition_penalty',
    ])
    for (const [key, value] of Object.entries(defaults)) {
      if (key === 'options') continue
      if (value == null || value === '') continue
      if (intKeys.has(key)) {
        const n = coercePreviewNumber(value, true)
        if (n != null) out[key] = n
        continue
      }
      if (floatKeys.has(key)) {
        const n = coercePreviewNumber(value, false)
        if (n != null) out[key] = n
        continue
      }
      out[key] = typeof value === 'string' ? value.trim() : value
    }
    if (defaults.options && typeof defaults.options === 'object') {
      const options = {}
      for (const [key, value] of Object.entries(defaults.options)) {
        if (value == null || value === '') continue
        options[key] = typeof value === 'string' ? value.trim() : value
      }
      if (Object.keys(options).length) out.options = options
    }
    return Object.keys(out).length ? out : null
  }
  if (defaultsKey === 'transcription_defaults') {
    if (defaults.language) out.language = String(defaults.language).trim()
    if (defaults.stream != null) out.stream = Boolean(defaults.stream)
    const options = {}
    if (defaults.options && typeof defaults.options === 'object') {
      for (const [key, value] of Object.entries(defaults.options)) {
        if (value == null || value === '') continue
        options[key] = value
      }
    }
    for (const key of ['max_tokens', 'temperature', 'top_p', 'top_k', 'seed']) {
      if (defaults[key] == null || defaults[key] === '') continue
      const asInt = key === 'max_tokens' || key === 'top_k' || key === 'seed'
      const n = coercePreviewNumber(defaults[key], asInt)
      if (n != null) options[key] = n
    }
    if (defaults.prompt) options.text = String(defaults.prompt).trim()
    if (Object.keys(options).length) out.options = options
    return Object.keys(out).length ? out : null
  }
  for (const [key, value] of Object.entries(defaults)) {
    if (key === 'options' || key === 'prompt') continue
    if (value != null && value !== '') out[key] = value
  }
  const options = {}
  if (defaults.options && typeof defaults.options === 'object') {
    for (const [key, value] of Object.entries(defaults.options)) {
      if (value == null || value === '') continue
      options[key] = value
    }
  }
  if (defaults.prompt) options.text = String(defaults.prompt).trim()
  if (Object.keys(options).length) out.options = options
  return Object.keys(out).length ? out : null
}

export function useAudioModelConfig(config, paramRegistry, enginesStore, llamaSwapStableId) {
  const catalogSections = computed(() =>
    Array.isArray(paramRegistry.value?.sections) ? paramRegistry.value.sections : [],
  )

  const catalogParamList = computed(() => {
    const out = []
    for (const section of catalogSections.value) {
      for (const param of section.params || []) {
        if (param?.reserved) continue
        out.push(param)
      }
    }
    return out
  })

  const audioEditableParams = computed(() => {
    const seen = new Set()
    const params = catalogParamList.value.filter((param) => {
      const scope = param.scope || 'process'
      const identity = `${scope}:${param.key}`
      if (seen.has(identity) || param.read_only || scope === 'request_option') return false
      seen.add(identity)
      return true
    })
    if (!seen.has('process:lazy_load')) {
      params.push(LAZY_LOAD_PARAM)
    }
    return params
  })

  const audioConfigGroups = computed(() => {
    const params = audioEditableParams.value
    const definitions = [
      {
        id: 'model',
        label: 'Model identity',
        description: 'Which prepared bundle, task, and mode this server instance runs.',
        scopes: ['model'],
        defaultExpanded: true,
      },
      {
        id: 'runtime',
        label: 'Runtime & startup',
        description: 'Backend, device, threading, and startup behavior written to the sidecar.',
        scopes: ['process'],
        defaultExpanded: true,
        extraParams: [LAZY_LOAD_PARAM],
      },
      {
        id: 'load',
        label: 'Load options',
        description: 'Applied once when weights are loaded from the bundle.',
        scopes: ['load_option'],
        defaultExpanded: false,
      },
      {
        id: 'session',
        label: 'Session options',
        description: 'Defaults used when audio.cpp opens a processing session.',
        scopes: ['session_option'],
        defaultExpanded: false,
      },
    ]
    return definitions
      .map((group) => {
        const scoped = params.filter((param) => group.scopes.includes(param.scope || 'process'))
        const extra = group.extraParams || []
        const merged = [...scoped]
        for (const extraParam of extra) {
          if (!merged.some((item) => item.key === extraParam.key)) {
            merged.push(extraParam)
          }
        }
        return { ...group, params: merged }
      })
      .filter((group) => group.params.length)
  })

  const audioRequestCapabilities = computed(() => {
    const seen = new Set()
    return catalogParamList.value.filter((param) => {
      const requestOnly = param.read_only || param.scope === 'request_option'
      if (!requestOnly || seen.has(param.key)) return false
      seen.add(param.key)
      return true
    })
  })

  const taskProfile = computed(() => paramRegistry.value?.task_profile || null)

  const isProfiledAudioModel = computed(() => Boolean(taskProfile.value))

  const requestFieldGroups = computed(() => (
    Array.isArray(paramRegistry.value?.request_field_groups)
      ? paramRegistry.value.request_field_groups
      : []
  ))

  const requestDefaultsKey = computed(() => (
    paramRegistry.value?.request_defaults_key || 'task_defaults'
  ))

  const apiEndpoint = computed(() => paramRegistry.value?.api_endpoint || '/v1/tasks/run')

  const apiExampleHint = computed(() => paramRegistry.value?.api_example_hint || '')

  const instructionsPolicy = computed(() => (
    paramRegistry.value?.instructions_policy || ''
  ))

  const instructionsPolicyGuidance = computed(() => (
    instructionsPolicyHint(instructionsPolicy.value, {
      source: paramRegistry.value?.instructions_policy_source || '',
      vocabulary: paramRegistry.value?.instructions_vocabulary || null,
    })
  ))

  const contractFingerprint = computed(() => (
    paramRegistry.value?.contract_fingerprint || ''
  ))

  const contractReviewRequired = computed(() => {
    const fp = String(contractFingerprint.value || '').trim()
    if (!fp) return Boolean(paramRegistry.value?.contract_review_required)
    const reviewed = String(
      config.value?.last_reviewed_fingerprint
      || paramRegistry.value?.last_reviewed_fingerprint
      || '',
    ).trim()
    return reviewed !== fp
  })

  function markContractReviewed() {
    if (!config.value || !contractFingerprint.value) return
    config.value.last_reviewed_fingerprint = contractFingerprint.value
    pruneStaleAudioRequestDefaults(config.value, requestDefaultsKey.value)
  }

  const requestDefaultsSectionTitle = computed(() => {
    const key = requestDefaultsKey.value
    if (key === 'speech_defaults') return 'Speech synthesis defaults'
    if (key === 'transcription_defaults') return 'Transcription defaults'
    return 'Task request defaults'
  })

  const audioTaskKind = computed(() => {
    const endpoint = apiEndpoint.value
    if (endpoint === '/v1/audio/speech') return 'tts'
    if (endpoint === '/v1/audio/transcriptions') return 'asr'
    return 'task'
  })

  const taskKindMeta = computed(() => {
    const kind = audioTaskKind.value
    const byKind = {
      tts: {
        label: 'Text-to-Speech',
        short: 'TTS',
        icon: 'pi-volume-up',
        tagSeverity: 'info',
        tabHint: 'Voice & synthesis',
      },
      asr: {
        label: 'Speech-to-Text',
        short: 'ASR',
        icon: 'pi-microphone',
        tagSeverity: 'success',
        tabHint: 'Transcription',
      },
      task: {
        label: 'Audio task',
        short: 'Task',
        icon: 'pi-sliders-h',
        tagSeverity: 'secondary',
        tabHint: 'Generic task',
      },
    }
    return byKind[kind] || byKind.task
  })

  const swapSetParamsPreview = computed(() => (
    buildSwapSetParamsPreview(requestDefaultsKey.value, config.value?.[requestDefaultsKey.value])
  ))

  const configuredDefaultsCount = computed(() => {
    const preview = swapSetParamsPreview.value
    if (!preview) return 0
    let count = Object.keys(preview).filter((key) => key !== 'options').length
    if (preview.options && typeof preview.options === 'object') {
      count += Object.keys(preview.options).length
    }
    return count
  })

  const defaultsApplyHint = computed(() => {
    const endpoint = apiEndpoint.value
    const key = requestDefaultsKey.value
    if (audioTaskKind.value === 'asr') {
      return (
        `Saved ${key} fields are injected into JSON ${endpoint} requests via llama-swap `
        + 'setParams. Audio file paths are still sent per request.'
      )
    }
    if (audioTaskKind.value === 'tts') {
      return (
        `Saved ${key} fields are injected into JSON ${endpoint} requests via llama-swap `
        + 'setParams after you apply config. Voice presets use the sidecar instead.'
      )
    }
    return (
      `Saved ${key} fields are injected into JSON ${endpoint} requests via llama-swap `
      + 'setParams when supported. Some inputs remain per-request only.'
    )
  })

  const setupProgress = computed(() => {
    const items = setupChecklist.value
    if (!items.length) return 0
    const done = items.filter((item) => item.done).length
    return Math.round((done / items.length) * 100)
  })

  const supportsVoicePresets = computed(() => {
    if (!isProfiledAudioModel.value) return false
    if (typeof paramRegistry.value?.supports_voice_presets === 'boolean') {
      return paramRegistry.value.supports_voice_presets
    }
    return requestDefaultsKey.value === 'speech_defaults'
  })

  const taskWorkflowTags = computed(() => {
    const workflows = taskProfile.value?.workflows || []
    return workflows.map((item) => String(item).replace(/_/g, ' '))
  })

  const voicePresetFieldDefs = computed(() => {
    const fields = new Map()
    for (const group of requestFieldGroups.value) {
      for (const field of group.fields || []) {
        if (field.preset_field) {
          fields.set(field.key, field)
        }
      }
    }
    if (!fields.size) {
      return [
        { key: 'voice_id', label: 'Built-in voice id', type: 'string', placeholder: 'alba' },
        {
          key: 'voice_ref',
          label: 'Reference audio (WAV)',
          type: 'path',
          placeholder: 'samples/reference.wav',
        },
        {
          key: 'reference_text',
          label: 'Reference transcript',
          type: 'textarea',
          placeholder: 'Transcript for the reference clip…',
        },
      ]
    }
    return [...fields.values()]
  })

  let voicePresetIdSeq = 0
  const voicePresetIdByName = new Map()
  const voicePresetNameDrafts = ref({})

  function ensureVoicePresetRowId(name) {
    if (!voicePresetIdByName.has(name)) {
      voicePresetIdSeq += 1
      voicePresetIdByName.set(name, `vp-${voicePresetIdSeq}`)
    }
    return voicePresetIdByName.get(name)
  }

  function forgetVoicePresetRowId(name) {
    voicePresetIdByName.delete(name)
  }

  function moveVoicePresetRowId(oldName, newName) {
    const id = voicePresetIdByName.get(oldName)
    if (id) {
      voicePresetIdByName.delete(oldName)
      voicePresetIdByName.set(newName, id)
      return
    }
    ensureVoicePresetRowId(newName)
  }

  function clearVoicePresetNameDraft(name) {
    if (!Object.prototype.hasOwnProperty.call(voicePresetNameDrafts.value, name)) return
    const drafts = { ...voicePresetNameDrafts.value }
    delete drafts[name]
    voicePresetNameDrafts.value = drafts
  }

  const voicePresetRows = computed(() => {
    const presets = config.value?.voice_presets
    if (!presets || typeof presets !== 'object' || Array.isArray(presets)) return []
    const names = new Set(Object.keys(presets))
    for (const known of [...voicePresetIdByName.keys()]) {
      if (!names.has(known)) voicePresetIdByName.delete(known)
    }
    return Object.entries(presets).map(([name, preset]) => ({
      id: ensureVoicePresetRowId(name),
      name,
      preset: preset && typeof preset === 'object' ? preset : {},
    }))
  })

  function voicePresetNameDraft(name) {
    if (Object.prototype.hasOwnProperty.call(voicePresetNameDrafts.value, name)) {
      return voicePresetNameDrafts.value[name]
    }
    return name
  }

  function setVoicePresetNameDraft(name, value) {
    voicePresetNameDrafts.value = {
      ...voicePresetNameDrafts.value,
      [name]: value == null ? '' : String(value),
    }
  }

  function commitVoicePresetRename(name) {
    const hasDraft = Object.prototype.hasOwnProperty.call(voicePresetNameDrafts.value, name)
    const draft = voicePresetNameDrafts.value[name]
    clearVoicePresetNameDraft(name)
    if (!hasDraft) return
    renameVoicePreset(name, draft)
  }

  const defaultVoicePresetOptions = computed(() => {
    const options = voicePresetRows.value.map((row) => ({
      label: row.name,
      value: row.name,
    }))
    options.unshift({ label: 'Use inline default object', value: '__inline__' })
    return options
  })

  const defaultVoicePresetSelection = computed(() => {
    const value = config.value?.default_voice_preset
    if (typeof value === 'string') return value
    if (value && typeof value === 'object') return '__inline__'
    return null
  })

  const audioInspectionSummary = computed(() => {
    const inspection = paramRegistry.value?.inspection || {}
    const summary = []
    if (inspection.family) summary.push(`Family: ${inspection.family}`)
    for (const task of inspection.tasks || []) {
      if (!task?.task) continue
      const modes = Array.isArray(task.modes) && task.modes.length
        ? ` (${task.modes.join(', ')})`
        : ''
      summary.push(`${task.task}${modes}`)
    }
    for (const capability of Object.keys(inspection.capabilities || {})) {
      if (inspection.capabilities[capability]) summary.push(capability)
    }
    return [...new Set(summary)]
  })

  const setupChecklist = computed(() => {
    const items = []
    const cfg = config.value || {}
    items.push({
      id: 'family',
      label: 'Model family selected',
      done: Boolean(cfg.family),
      detail: cfg.family || 'Required in Runtime',
    })
    items.push({
      id: 'task',
      label: 'Task selected',
      done: Boolean(cfg.task),
      detail: cfg.task || 'Required in Runtime',
    })
    items.push({
      id: 'backend',
      label: 'Runtime backend set',
      done: Boolean(cfg.backend),
      detail: cfg.backend ? `Backend: ${cfg.backend}` : 'Required in Runtime',
    })
    if (supportsVoicePresets.value) {
      items.push({
        id: 'voice',
        label: 'Voice preset configured (optional)',
        done: voicePresetRows.value.length > 0,
        detail: voicePresetRows.value.length
          ? `${voicePresetRows.value.length} preset(s) in sidecar`
          : 'Optional asset',
        tab: 'assets',
      })
    }
    if (configuredDefaultsCount.value > 0) {
      items.push({
        id: 'defaults',
        label: 'Proxy defaults configured',
        done: true,
        detail: `${configuredDefaultsCount.value} field(s) via llama-swap setParams`,
        tab: 'api',
      })
    } else if (isProfiledAudioModel.value) {
      const policy = String(instructionsPolicy.value || '').toLowerCase()
      const needsGuidance = ['soft_tags', 'caption_option', 'text_prefix'].includes(policy)
      items.push({
        id: 'defaults',
        label: needsGuidance ? 'Proxy defaults recommended' : 'Proxy defaults (optional)',
        done: false,
        detail: needsGuidance
          ? (instructionsPolicyGuidance.value || 'Review Defaults for style/caption settings')
          : 'Optional proxy defaults',
        tab: 'api',
      })
    }
    return items
  })

  function audioParamHasExplicitValue(param, sourceConfig = config.value) {
    const nestedKey = AUDIO_NESTED_SCOPE_KEYS[param.scope]
    if (nestedKey) {
      const nested = sourceConfig?.[nestedKey]
      return Boolean(
        nested
        && typeof nested === 'object'
        && !Array.isArray(nested)
        && Object.prototype.hasOwnProperty.call(nested, param.key),
      )
    }
    return Object.prototype.hasOwnProperty.call(sourceConfig || {}, param.key)
  }

  function audioParamValue(param, sourceConfig = config.value) {
    if (audioParamHasExplicitValue(param, sourceConfig)) {
      const nestedKey = AUDIO_NESTED_SCOPE_KEYS[param.scope]
      if (nestedKey) {
        return sourceConfig[nestedKey][param.key]
      }
      return sourceConfig[param.key]
    }
    return defaultValueForAudioParam(param)
  }

  function clearOptionalParamValue(param, target) {
    if (isOptionalConfigParam(param)) {
      target[param.key] = null
    } else {
      delete target[param.key]
    }
  }

  function audioParamOptions(param) {
    if (param.key === 'mode') {
      const task = config.value?.task
      const taskRow = (paramRegistry.value?.inspection?.tasks || [])
        .find((item) => item?.task === task)
      if (taskRow?.modes?.length) {
        return taskRow.modes.map((mode) => ({ value: mode, label: mode }))
      }
    }
    if (param.key === 'backend') {
      const descriptor = (enginesStore?.engineDescriptors || [])
        .find((item) => item.id === 'audio_cpp')
      const available = descriptor?.available_runtime_backends
      if (Array.isArray(available) && available.length) {
        return available.map((backend) => ({ value: backend, label: backend }))
      }
    }
    return Array.isArray(param.options) ? param.options : []
  }

  function setAudioParamValue(param, value) {
    if (!config.value) return
    const nestedKey = AUDIO_NESTED_SCOPE_KEYS[param.scope]
    const empty = value === undefined || value === null || value === ''
      || (typeof value === 'number' && Number.isNaN(value))
      || (Array.isArray(value) && value.length === 0)
    if (nestedKey) {
      if (!config.value[nestedKey] || typeof config.value[nestedKey] !== 'object') {
        config.value[nestedKey] = {}
      }
      if (empty) clearOptionalParamValue(param, config.value[nestedKey])
      else config.value[nestedKey][param.key] = coerceAudioParamValue(param, value)
      return
    }
    if (empty) {
      if (isOptionalConfigParam(param)) config.value[param.key] = null
      else delete config.value[param.key]
    } else {
      config.value[param.key] = coerceAudioParamValue(param, value)
    }
    if (param.key === 'task') {
      const modeParam = audioEditableParams.value.find((item) => item.key === 'mode')
      if (!modeParam) return
      const options = audioParamOptions(modeParam)
      if (!options.some((option) => option.value === config.value.mode)) {
        const offline = options.find((option) => option.value === 'offline')
        config.value.mode = offline?.value || options[0]?.value || null
      }
    }
  }

  function updateAudioJsonParam(param, raw) {
    if (!raw || !String(raw).trim()) {
      setAudioParamValue(param, null)
      return
    }
    try {
      setAudioParamValue(param, JSON.parse(raw))
    } catch {
      setAudioParamValue(param, raw)
    }
  }

  function ensureRequestDefaultsShape() {
    const key = requestDefaultsKey.value
    if (
      !config.value[key]
      || typeof config.value[key] !== 'object'
      || Array.isArray(config.value[key])
    ) {
      config.value[key] = {}
    }
  }

  function requestFieldKey(field) {
    return field.request_field || field.speech_field || field.transcription_field || field.key
  }

  function requestDefaultValue(field) {
    ensureRequestDefaultsShape()
    const defaults = config.value[requestDefaultsKey.value]
    if (field.nested || field.options_key) {
      const options = defaults.options
      if (field.key === 'prompt') {
        if (Object.prototype.hasOwnProperty.call(defaults, 'prompt')) return defaults.prompt
        if (options && Object.prototype.hasOwnProperty.call(options, 'text')) return options.text
        return null
      }
      const optKey = field.options_key || field.key
      if (options && typeof options === 'object' && Object.prototype.hasOwnProperty.call(options, optKey)) {
        return options[optKey]
      }
      return field.type === 'bool' ? false : null
    }
    const key = requestFieldKey(field)
    if (Object.prototype.hasOwnProperty.call(defaults, key)) return defaults[key]
    return field.type === 'bool' ? false : null
  }

  function setRequestDefaultValue(field, value) {
    ensureRequestDefaultsShape()
    const defaults = config.value[requestDefaultsKey.value]
    const empty = value === undefined || value === null || value === ''
      || (typeof value === 'number' && Number.isNaN(value))
    if (field.key === 'prompt') {
      const text = value == null ? '' : String(value).trim()
      if (!text) {
        if (Object.prototype.hasOwnProperty.call(defaults, 'prompt')) defaults.prompt = null
        if (defaults.options?.text !== undefined) defaults.options.text = null
      } else {
        defaults.prompt = text
      }
      return
    }
    if (field.nested || field.options_key) {
      if (!defaults.options || typeof defaults.options !== 'object') {
        defaults.options = {}
      }
      const key = field.options_key || field.key
      if (empty) defaults.options[key] = null
      else if (field.type === 'bool') {
        defaults.options[key] = Boolean(value)
      } else if (field.type === 'int') {
        defaults.options[key] = parseInt(value, 10)
      } else if (field.type === 'float') {
        defaults.options[key] = parseFloat(value)
      } else {
        defaults.options[key] = String(value)
      }
      return
    }
    const key = requestFieldKey(field)
    if (empty) defaults[key] = null
    else if (field.type === 'bool') {
      defaults[key] = Boolean(value)
    } else {
      defaults[key] = value
    }
  }

  function ensureTtsConfigShape() {
    if (
      !config.value.voice_presets
      || typeof config.value.voice_presets !== 'object'
      || Array.isArray(config.value.voice_presets)
    ) {
      config.value.voice_presets = {}
    }
    ensureRequestDefaultsShape()
  }

  function addVoicePreset() {
    ensureTtsConfigShape()
    let index = 1
    let name = 'preset-1'
    while (config.value.voice_presets[name]) {
      index += 1
      name = `preset-${index}`
    }
    config.value.voice_presets[name] = {}
    ensureVoicePresetRowId(name)
  }

  function removeVoicePreset(name) {
    ensureTtsConfigShape()
    if (!config.value.voice_presets[name]) return
    delete config.value.voice_presets[name]
    forgetVoicePresetRowId(name)
    clearVoicePresetNameDraft(name)
    if (config.value.default_voice_preset === name) {
      config.value.default_voice_preset = null
    }
  }

  function renameVoicePreset(oldName, newName) {
    ensureTtsConfigShape()
    const trimmed = String(newName || '').trim()
    if (!trimmed || trimmed === oldName) return
    if (config.value.voice_presets[trimmed]) return
    moveVoicePresetRowId(oldName, trimmed)
    clearVoicePresetNameDraft(oldName)
    config.value.voice_presets[trimmed] = config.value.voice_presets[oldName] || {}
    delete config.value.voice_presets[oldName]
    if (config.value.default_voice_preset === oldName) {
      config.value.default_voice_preset = trimmed
    }
  }

  function setVoicePresetField(name, key, value) {
    ensureTtsConfigShape()
    if (!config.value.voice_presets[name]) {
      config.value.voice_presets[name] = {}
    }
    // Keep raw text while typing; trimming on every keystroke fights the controlled
    // input and can drop focus. Backend normalize_voice_presets cleans on apply.
    const text = value == null ? '' : String(value)
    if (text === '') delete config.value.voice_presets[name][key]
    else config.value.voice_presets[name][key] = text
  }

  function setDefaultVoicePresetSelection(value) {
    if (!value) {
      config.value.default_voice_preset = null
      return
    }
    if (value === '__inline__') {
      if (typeof config.value.default_voice_preset !== 'object') {
        config.value.default_voice_preset = {}
      }
      return
    }
    config.value.default_voice_preset = value
  }

  const requestApiExample = computed(() => {
    const modelId = config.value?.model_alias || llamaSwapStableId?.value || 'your-model-id'
    const endpoint = apiEndpoint.value
    const defaultsKey = requestDefaultsKey.value
    const defaults = config.value?.[defaultsKey]
    const body = { model: modelId }

    if (endpoint === '/v1/audio/speech') {
      body.input = 'Hello from audio.cpp.'
    } else if (endpoint === '/v1/audio/transcriptions') {
      body.audio = '/path/to/speech.wav'
    } else {
      body.task = config.value?.task || 'gen'
      body.family = config.value?.family || 'ace_step'
      if (config.value?.mode) body.mode = config.value.mode
      body.audio = '/path/to/input.wav'
      body.text = 'Example prompt text.'
    }

    if (defaults && typeof defaults === 'object') {
      for (const [key, value] of Object.entries(defaults)) {
        if (key === 'options') continue
        if (key === 'prompt' && endpoint === '/v1/audio/transcriptions') continue
        if (value != null && value !== '') body[key] = value
      }
      if (defaults.prompt && endpoint === '/v1/audio/transcriptions') {
        body.options = { ...(body.options || {}), text: defaults.prompt }
      }
      if (defaults.options && typeof defaults.options === 'object' && Object.keys(defaults.options).length) {
        body.options = { ...(body.options || {}), ...defaults.options }
      }
    }

    if (endpoint === '/v1/audio/speech') {
      const defaultPreset = config.value?.default_voice_preset
      if (typeof defaultPreset === 'string' && defaultPreset && !body.voice) {
        body.voice = defaultPreset
      }
    }

    const lines = [
      `curl http://localhost:2000${endpoint} \\`,
      "  -H 'Content-Type: application/json' \\",
    ]
    if (endpoint === '/v1/audio/speech') {
      lines.push('  -o speech.wav \\')
    }
    lines.push("  -d '" + JSON.stringify(body, null, 2).replace(/'/g, "'\\''") + "'")
    if (endpoint === '/v1/tasks/run') {
      lines.push('')
      lines.push('# Direct upstream fallback:')
      lines.push(`curl http://localhost:2000/upstream/${modelId}${endpoint} \\`)
      lines.push("  -H 'Content-Type: application/json' \\")
      lines.push("  -d '" + JSON.stringify(body, null, 2).replace(/'/g, "'\\''") + "'")
    }
    if (endpoint === '/v1/audio/transcriptions') {
      lines.push('')
      lines.push('# Multipart (OpenAI-compatible upload):')
      lines.push(`curl http://localhost:2000${endpoint} \\`)
      lines.push(`  -F "model=${modelId}" \\`)
      lines.push('  -F "file=@speech.wav"')
    }
    return lines.join('\n')
  })

  function filterGroupParams(params, searchQuery, hideUnsupported) {
    const q = searchQuery.trim()
    if (!q && !hideUnsupported) return params
    return params.filter((param) => {
      if (hideUnsupported && param.supported === false) return false
      if (!q) return true
      return paramMatchesSearch(param, q, hideUnsupported)
    })
  }

  return {
    catalogParamList,
    audioEditableParams,
    audioConfigGroups,
    audioRequestCapabilities,
    taskProfile,
    isProfiledAudioModel,
    requestFieldGroups,
    requestDefaultsKey,
    apiEndpoint,
    apiExampleHint,
    instructionsPolicy,
    instructionsPolicyGuidance,
    contractReviewRequired,
    contractFingerprint,
    markContractReviewed,
    requestDefaultsSectionTitle,
    audioTaskKind,
    taskKindMeta,
    swapSetParamsPreview,
    configuredDefaultsCount,
    defaultsApplyHint,
    setupProgress,
    supportsVoicePresets,
    taskWorkflowTags,
    voicePresetFieldDefs,
    voicePresetRows,
    voicePresetNameDraft,
    setVoicePresetNameDraft,
    commitVoicePresetRename,
    defaultVoicePresetOptions,
    defaultVoicePresetSelection,
    audioInspectionSummary,
    setupChecklist,
    requestApiExample,
    audioParamValue,
    audioParamOptions,
    setAudioParamValue,
    updateAudioJsonParam,
    ensureRequestDefaultsShape,
    requestDefaultValue,
    setRequestDefaultValue,
    ensureTtsConfigShape,
    addVoicePreset,
    removeVoicePreset,
    renameVoicePreset,
    setVoicePresetField,
    setDefaultVoicePresetSelection,
    filterGroupParams,
  }
}
