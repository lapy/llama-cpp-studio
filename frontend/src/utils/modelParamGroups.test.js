import { describe, expect, it } from 'vitest'
import {
  PARAM_GROUPS,
  groupForParam,
  groupParameters,
  paramMatchesSearch,
} from './modelParamGroups'

const GROUP_IDS = [
  'context',
  'memory',
  'speculative',
  'sampling',
  'server',
  'multimodal',
  'model',
  'performance',
  'logging',
  'other',
]

describe('PARAM_GROUPS', () => {
  it('declares the ordered group metadata', () => {
    expect(PARAM_GROUPS.map((group) => group.id)).toEqual(GROUP_IDS)
    for (const group of PARAM_GROUPS) {
      expect(group.title).toBeTruthy()
      expect(group.description).toBeTruthy()
      expect(group.icon).toMatch(/^pi /)
      expect(typeof group.defaultOpen).toBe('boolean')
    }
  })

  it('opens only Context and Memory by default', () => {
    expect(PARAM_GROUPS.filter((group) => group.defaultOpen).map((group) => group.id)).toEqual([
      'context',
      'memory',
    ])
  })
})

describe('groupForParam', () => {
  it.each([
    ['ctx_size', 'context'],
    ['n_ctx', 'context'],
    ['batch_size', 'context'],
    ['n_batch', 'context'],
    ['ubatch_size', 'context'],
    ['parallel', 'context'],
    ['cont_batching', 'context'],
    ['cache_reuse', 'context'],
    ['n_keep', 'context'],
    ['n_discard', 'context'],
    ['session_len', 'context'],
    ['rope_freq_base', 'context'],
    ['yarn_orig_ctx', 'context'],
    ['keep', 'context'],
    ['swa_full', 'context'],
    ['n_gpu_layers', 'memory'],
    ['main_gpu', 'memory'],
    ['gpu_split', 'memory'],
    ['split_mode', 'memory'],
    ['tensor_split', 'memory'],
    ['offload_kqv', 'memory'],
    ['mlock', 'memory'],
    ['mmap', 'memory'],
    ['flash_attn', 'memory'],
    ['cache_type_k', 'memory'],
    ['cache_prompt', 'memory'],
    ['cache_ram', 'memory'],
    ['kv_unified', 'memory'],
    ['direct_io', 'memory'],
    ['override_kv', 'memory'],
    ['temperature', 'sampling'],
    ['temp', 'sampling'],
    ['top_k', 'sampling'],
    ['top_p', 'sampling'],
    ['min_p', 'sampling'],
    ['repeat_penalty', 'sampling'],
    ['presence_penalty', 'sampling'],
    ['frequency_penalty', 'sampling'],
    ['seed', 'sampling'],
    ['grammar', 'sampling'],
    ['mirostat_tau', 'sampling'],
    ['dry_multiplier', 'sampling'],
    ['n_predict', 'sampling'],
    ['reasoning_effort', 'sampling'],
    ['adaptive_target', 'sampling'],
    ['backend_sampling', 'sampling'],
    ['host', 'server'],
    ['port', 'server'],
    ['api_key', 'server'],
    ['request_timeout', 'server'],
    ['slots', 'server'],
    ['props', 'server'],
    ['metrics', 'server'],
    ['chat_template', 'server'],
    ['jinja', 'server'],
    ['embedding', 'server'],
    ['rerank', 'server'],
    ['ui_config', 'server'],
    ['mcp_servers_json', 'server'],
    ['tools', 'server'],
    ['json_schema', 'server'],
    ['ssl_cert_file', 'server'],
    ['cors_origins', 'server'],
    ['embeddings', 'server'],
    ['reranking', 'server'],
    ['reuse_port', 'server'],
    ['prefill_assistant', 'server'],
    ['mmproj', 'multimodal'],
    ['image_min_tokens', 'multimodal'],
    ['vision', 'multimodal'],
    ['audio', 'multimodal'],
    ['clip_model', 'multimodal'],
    ['video_fps', 'multimodal'],
    ['mtmd_batch_max_tokens', 'multimodal'],
    ['fit', 'memory'],
    ['fit_ctx', 'memory'],
    ['fit_target', 'memory'],
    ['hf_repo', 'model'],
    ['model_url', 'model'],
    ['lora', 'model'],
    ['control_vector', 'model'],
    ['spec_type', 'speculative'],
    ['spec_draft_model', 'speculative'],
    ['lookup_cache_static', 'speculative'],
    ['threads', 'performance'],
    ['n_threads', 'performance'],
    ['threads_batch', 'performance'],
    ['numa', 'performance'],
    ['prio', 'performance'],
    ['poll', 'performance'],
    ['cpu', 'performance'],
    ['cpu_moe', 'performance'],
    ['verbose', 'logging'],
    ['log_level', 'logging'],
    ['log_file', 'logging'],
    ['debug', 'logging'],
    ['perf', 'logging'],
  ])('maps %s to %s', (key, expectedGroup) => {
    expect(groupForParam({ key }).id).toBe(expectedGroup)
  })

  it('returns the group metadata and falls back to Other for unknown keys', () => {
    expect(groupForParam({ key: 'ctx_size' }).title).toBe('Context & tokens')
    expect(groupForParam({ key: 'mystery_key' }).id).toBe('other')
    expect(groupForParam({ key: 'mystery_key' }).title).toBe('Engine-specific')
    // `logits_all` is an output control, not a logging key: ambiguous names stay
    // in Other by design (see the module documentation).
    expect(groupForParam({ key: 'logits_all' }).id).toBe('other')
    expect(groupForParam({ key: 'template' }).id).toBe('other')
    expect(groupForParam({}).id).toBe('other')
    expect(groupForParam(null).id).toBe('other')
  })

  it('accepts a raw key string for convenience', () => {
    expect(groupForParam('temperature').id).toBe('sampling')
  })

  it('normalizes case and whitespace', () => {
    expect(groupForParam({ key: '  Temperature ' }).id).toBe('sampling')
  })
})

describe('groupParameters', () => {
  it('returns ordered groups with counts and modified counts', () => {
    const params = [
      { key: 'temperature', modified: false },
      { key: 'ctx_size', modified: true },
      { key: 'top_p', modified: true },
      { key: 'mystery_key', modified: true },
    ]

    const groups = groupParameters(params)

    expect(groups.map((group) => group.id)).toEqual(['context', 'sampling', 'other'])
    expect(groups[0]).toMatchObject({ count: 1, modifiedCount: 1 })
    expect(groups[1].params.map((param) => param.key)).toEqual(['temperature', 'top_p'])
    expect(groups[1]).toMatchObject({ count: 2, modifiedCount: 1 })
    expect(groups[2]).toMatchObject({ count: 1, modifiedCount: 1 })
  })

  it('supports a custom modified predicate', () => {
    const config = { temperature: 0.2, seed: null }
    const params = [{ key: 'temperature' }, { key: 'seed' }]

    const [group] = groupParameters(params, {
      isModified: (param) => config[param.key] != null,
    })

    expect(group.id).toBe('sampling')
    expect(group.modifiedCount).toBe(1)
  })

  it('omits empty groups and invalid entries', () => {
    expect(groupParameters([])).toEqual([])
    expect(groupParameters(null)).toEqual([])
    expect(groupParameters([{ key: '' }, null, undefined])).toEqual([])
  })

  it('is deterministic: group order is fixed and input order is preserved inside groups', () => {
    const params = [
      { key: 'seed' },
      { key: 'ctx_size' },
      { key: 'host' },
      { key: 'temperature' },
      { key: 'temperature' },
    ]

    const first = groupParameters(params)
    const second = groupParameters(params)

    expect(first.map((group) => group.id)).toEqual(['context', 'sampling', 'server'])
    expect(first[1].params.map((param) => param.key)).toEqual([
      'seed',
      'temperature',
      'temperature',
    ])
    expect(second.map((group) => group.id)).toEqual(first.map((group) => group.id))
    expect(second[1].params.map((param) => param.key)).toEqual(
      first[1].params.map((param) => param.key),
    )
  })
})

describe('paramMatchesSearch', () => {
  const temperature = {
    key: 'temperature',
    label: 'Temperature',
    description: 'Sampling temperature',
    flags: ['--temperature', '--temp'],
    supported: true,
  }

  it('matches key, label, description, and flags with multi-word AND', () => {
    expect(paramMatchesSearch(temperature, 'temp')).toBe(true)
    expect(paramMatchesSearch(temperature, 'TEMPERATURE')).toBe(true)
    expect(paramMatchesSearch(temperature, 'sampling temp')).toBe(true)
    expect(paramMatchesSearch(temperature, '--temp')).toBe(true)
    expect(paramMatchesSearch(temperature, 'sampling missing')).toBe(false)
  })

  it('does not match an empty query or unsupported params when hidden', () => {
    expect(paramMatchesSearch(temperature, '   ')).toBe(false)
    expect(paramMatchesSearch({ ...temperature, supported: false }, 'temp')).toBe(true)
    expect(
      paramMatchesSearch({ ...temperature, supported: false }, 'temp', { hideUnsupported: true }),
    ).toBe(false)
  })
})
