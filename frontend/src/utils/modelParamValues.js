/**
 * Pure helpers for engine parameter values.
 *
 * Extracted from ModelConfig.vue so the view stays within its module-size
 * budget while remaining the only place that mutates the draft. Every helper is
 * a pure function of a parameter definition (and sometimes a saved value), so
 * the CSV-enum and "is overridden" rules stay independently testable.
 */

/** Parameters whose value is a list encoded as one delimited string. */
export function isDelimitedEnumParam(param) {
  return ['csv_enum', 'semicolon_enum'].includes(param?.value_kind) || param?.type === 'multiselect'
}

/** Separator used by the engine's delimited enum syntax. */
export function delimitedEnumSeparator(param) {
  return param?.value_kind === 'semicolon_enum' ? ';' : ','
}

/** Normalize a saved/typed CSV-enum value to a trimmed string array. */
export function normalizeCsvEnumValue(value, param) {
  if (Array.isArray(value)) {
    return value.filter((item) => item != null && item !== '')
  }
  if (value == null || value === '') return []
  if (typeof value === 'string') {
    return value
      .split(delimitedEnumSeparator(param))
      .map((item) => item.trim())
      .filter(Boolean)
  }
  return [value]
}

/**
 * Draft value seeded when a parameter is added to the pane (or missing from a
 * saved section). Keeps flags and delimited enums in the shape the controls use.
 */
export function defaultValueForParam(param) {
  if (['repeatable', 'space_list'].includes(param?.value_kind)) {
    return Array.isArray(param.default) ? [...param.default] : []
  }
  if (isDelimitedEnumParam(param)) {
    return normalizeCsvEnumValue(param.default, param)
  }
  if (param?.value_kind === 'flag') return param.negative_flag ? null : true
  return param?.default ?? null
}

/**
 * Whether a saved engine section explicitly carries a value for this parameter.
 * `null` is meaningful for parameters whose "unset" state is stored explicitly.
 */
export function paramIsActiveInSection(section, param) {
  if (!Object.prototype.hasOwnProperty.call(section || {}, param?.key)) return false
  const value = section[param.key]
  if (value === null) return true
  if (param.value_kind === 'flag') {
    return value === true || (Boolean(param.negative_flag) && value === false)
  }
  if (['repeatable', 'space_list'].includes(param.value_kind)) {
    return Array.isArray(value) ? value.length > 0 : Boolean(value)
  }
  if (isDelimitedEnumParam(param)) {
    return normalizeCsvEnumValue(value, param).length > 0
  }
  return value !== undefined && value !== null && value !== ''
}
