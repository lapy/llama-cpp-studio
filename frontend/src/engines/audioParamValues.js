/**
 * Pure value helpers for audio.cpp model parameters.
 *
 * Extracted from `useAudioModelConfig` (module-size ratchet): these do not
 * depend on the composable's reactive state, and the model config view and the
 * composable both need them. The composable re-exports them so existing
 * imports keep working.
 */

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
    return value
      .split(delimitedEnumSeparator(param))
      .map((s) => s.trim())
      .filter(Boolean)
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
