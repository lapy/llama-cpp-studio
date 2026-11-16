import { describe, expect, it } from 'vitest'

import {
  defaultValueForParam,
  delimitedEnumSeparator,
  isDelimitedEnumParam,
  normalizeCsvEnumValue,
  paramIsActiveInSection,
} from './modelParamValues'

const samplers = {
  key: 'samplers',
  type: 'multiselect',
  value_kind: 'semicolon_enum',
  default: ['top_k', 'top_p'],
}

const flag = {
  key: 'jinja',
  type: 'bool',
  value_kind: 'flag',
  default: true,
}

describe('delimited enum helpers', () => {
  it('detects delimited enums and their separator', () => {
    expect(isDelimitedEnumParam(samplers)).toBe(true)
    expect(isDelimitedEnumParam({ value_kind: 'csv_enum' })).toBe(true)
    expect(isDelimitedEnumParam({ type: 'multiselect' })).toBe(true)
    expect(isDelimitedEnumParam({ value_kind: 'scalar', type: 'int' })).toBe(false)
    expect(delimitedEnumSeparator(samplers)).toBe(';')
    expect(delimitedEnumSeparator({ value_kind: 'csv_enum' })).toBe(',')
  })

  it('normalizes arrays, delimited strings, and scalars', () => {
    expect(normalizeCsvEnumValue(['a', '', null, ' b '], samplers)).toEqual(['a', ' b '])
    expect(normalizeCsvEnumValue('a; b ;;c', samplers)).toEqual(['a', 'b', 'c'])
    expect(normalizeCsvEnumValue('a, b', { value_kind: 'csv_enum' })).toEqual(['a', 'b'])
    expect(normalizeCsvEnumValue(null, samplers)).toEqual([])
    expect(normalizeCsvEnumValue(7, samplers)).toEqual([7])
  })
})

describe('defaultValueForParam', () => {
  it('keeps list, enum, flag, and scalar defaults in control shape', () => {
    expect(defaultValueForParam(samplers)).toEqual(['top_k', 'top_p'])
    expect(defaultValueForParam({ value_kind: 'repeatable', default: ['a'] })).toEqual(['a'])
    expect(defaultValueForParam({ value_kind: 'repeatable' })).toEqual([])
    expect(defaultValueForParam(flag)).toBe(true)
    expect(defaultValueForParam({ value_kind: 'flag', negative_flag: '--no-x' })).toBeNull()
    expect(defaultValueForParam({ default: 0 })).toBe(0)
    expect(defaultValueForParam({})).toBeNull()
  })
})

describe('paramIsActiveInSection', () => {
  const ctx = { key: 'ctx_size', type: 'int', value_kind: 'scalar' }

  it('only treats explicit saved values as active', () => {
    expect(paramIsActiveInSection({}, ctx)).toBe(false)
    expect(paramIsActiveInSection({ ctx_size: 0 }, ctx)).toBe(true)
    expect(paramIsActiveInSection({ ctx_size: null }, ctx)).toBe(true)
    expect(paramIsActiveInSection({ ctx_size: '' }, ctx)).toBe(false)
  })

  it('understands flags, lists, and delimited enums', () => {
    expect(paramIsActiveInSection({ jinja: true }, flag)).toBe(true)
    expect(paramIsActiveInSection({ jinja: false }, flag)).toBe(false)
    expect(paramIsActiveInSection({ jinja: false }, { ...flag, negative_flag: '--no-jinja' })).toBe(
      true,
    )
    expect(paramIsActiveInSection({ samplers: [] }, samplers)).toBe(false)
    expect(paramIsActiveInSection({ samplers: 'top_k;top_p' }, samplers)).toBe(true)
    expect(paramIsActiveInSection({ samplers: ['top_k'] }, samplers)).toBe(true)
  })
})
