import { describe, expect, it } from 'vitest'

import {
  buildImportPreview,
  joinCliTokens,
  parseCliCommand,
  tokenizeCli,
  valuesEqual,
} from './parseCliCommand'

const catalog = [
  {
    key: 'temperature',
    label: 'Temperature',
    type: 'float',
    scalar_type: 'float',
    value_kind: 'scalar',
    primary_flag: '--temperature',
    flags: ['--temperature', '--temp'],
    supported: true,
  },
  {
    key: 'ctx_size',
    label: 'Context Size',
    type: 'int',
    scalar_type: 'int',
    value_kind: 'scalar',
    primary_flag: '--ctx-size',
    flags: ['--ctx-size'],
    supported: true,
  },
  {
    key: 'n_gpu_layers',
    label: 'GPU Layers',
    type: 'int',
    scalar_type: 'int',
    value_kind: 'scalar',
    primary_flag: '--n-gpu-layers',
    flags: ['--n-gpu-layers', '--gpu-layers'],
    default: 'auto',
    supported: true,
  },
  {
    key: 'jinja',
    label: 'Jinja',
    type: 'bool',
    value_kind: 'flag',
    primary_flag: '--jinja',
    negative_flag: '--no-jinja',
    flags: ['--jinja', '--no-jinja'],
    supported: true,
  },
  {
    key: 'lora',
    label: 'LoRA',
    type: 'list',
    value_kind: 'repeatable',
    primary_flag: '--lora',
    flags: ['--lora'],
    supported: true,
  },
  {
    key: 'chat_template_kwargs',
    label: 'Chat template kwargs',
    type: 'json',
    value_kind: 'json_object',
    primary_flag: '--chat-template-kwargs',
    flags: ['--chat-template-kwargs'],
    supported: true,
  },
  {
    key: 'cache_type_k',
    label: 'Cache type K',
    type: 'select',
    value_kind: 'enum',
    primary_flag: '--cache-type-k',
    flags: ['--cache-type-k'],
    options: ['f16', 'q8_0'],
    supported: true,
  },
  {
    key: 'host',
    label: 'Host',
    type: 'string',
    value_kind: 'scalar',
    primary_flag: '--host',
    flags: ['--host'],
    reserved: false,
    supported: true,
  },
  {
    key: 'port',
    label: 'Port',
    type: 'int',
    value_kind: 'scalar',
    primary_flag: '--port',
    flags: ['--port'],
    reserved: true,
    supported: true,
  },
  {
    key: 'model',
    label: 'Model',
    type: 'string',
    value_kind: 'scalar',
    primary_flag: '--model',
    flags: ['--model'],
    reserved: true,
    supported: true,
  },
  {
    key: 'legacy_mirostat',
    label: 'Mirostat',
    type: 'int',
    value_kind: 'scalar',
    primary_flag: '--mirostat',
    flags: ['--mirostat'],
    supported: false,
  },
]

function keysOf(rows) {
  return rows.map((row) => row.key)
}

describe('tokenizeCli', () => {
  it('splits flags, quoted values, and equals-form tokens', () => {
    const { tokens, parseError } = tokenizeCli(
      `--temp 0.7 --ctx-size=8192 --chat-template-kwargs '{"a":1}'`,
    )
    expect(parseError).toBeNull()
    expect(tokens).toEqual(['--temp', '0.7', '--ctx-size=8192', '--chat-template-kwargs', '{"a":1}'])
  })

  it('reports unclosed quotes', () => {
    const { parseError, tokens } = tokenizeCli(`--temp "0.7`)
    expect(parseError).toMatch(/unclosed quote/i)
    expect(tokens).toEqual(['--temp', '0.7'])
  })
})

describe('parseCliCommand', () => {
  it('extracts catalog params from a full llama-swap style command', () => {
    const parsed = parseCliCommand(
      '${LLAMA_CPP_BIN} --model ${MODEL} --port ${PORT} --host 127.0.0.1 --ctx-size 8192 --n-gpu-layers -1 --temp=0.6 --jinja',
      catalog,
    )
    expect(keysOf(parsed.params).sort()).toEqual(['ctx_size', 'jinja', 'n_gpu_layers', 'temperature'])
    expect(parsed.params.find((row) => row.key === 'temperature').value).toBe(0.6)
    expect(parsed.params.find((row) => row.key === 'ctx_size').value).toBe(8192)
    expect(parsed.params.find((row) => row.key === 'n_gpu_layers').value).toBe(-1)
    expect(parsed.params.find((row) => row.key === 'jinja').value).toBe(true)
    expect(parsed.reserved.map((row) => row.flag).sort()).toEqual(['--host', '--model', '--port'])
    expect(parsed.params.some((row) => row.key === 'host')).toBe(false)
    expect(parsed.params.some((row) => row.key === 'port')).toBe(false)
  })

  it('skips Studio-owned host and port even when the catalog does not mark host reserved', () => {
    const parsed = parseCliCommand('--host 0.0.0.0 --port 8081 --temp 0.5', catalog)
    expect(keysOf(parsed.params)).toEqual(['temperature'])
    const host = parsed.reserved.find((row) => row.flag === '--host')
    const port = parsed.reserved.find((row) => row.flag === '--port')
    expect(host.reason).toMatch(/listen address and port/i)
    expect(port.reason).toMatch(/listen address and port/i)
    expect(host.value).toBe('0.0.0.0')
  })

  it('skips model / alias / mmproj identity flags', () => {
    const parsed = parseCliCommand(
      '--model /weights/model.gguf --alias my-app --mmproj /weights/mm.gguf --ctx-size 4096',
      catalog,
    )
    expect(keysOf(parsed.params)).toEqual(['ctx_size'])
    expect(parsed.reserved.map((row) => row.flag).sort()).toEqual(['--alias', '--mmproj', '--model'])
  })

  it('does not import unknown or removed catalog flags', () => {
    const parsed = parseCliCommand('--ctx-size 2048 --some-removed-flag 9 --also-gone', catalog)
    expect(keysOf(parsed.params)).toEqual(['ctx_size'])
    expect(parsed.unknown).toHaveLength(2)
    expect(parsed.unknown[0].reason).toMatch(/not in this engine catalog/i)
    expect(joinCliTokens(parsed.leftoverTokens)).toBe('--some-removed-flag 9 --also-gone')
  })

  it('keeps deprecated / unsupported catalog flags out of the import list', () => {
    const parsed = parseCliCommand('--mirostat 2 --temp 0.4', catalog)
    expect(keysOf(parsed.params)).toEqual(['temperature'])
    expect(parsed.unsupported).toHaveLength(1)
    expect(parsed.unsupported[0].key).toBe('legacy_mirostat')
    expect(parsed.unsupported[0].value).toBe(2)
    expect(parsed.unsupported[0].reason).toMatch(/deprecated or unsupported/i)
  })

  it('merges repeatable flags and parses JSON / negative flags', () => {
    const parsed = parseCliCommand(
      `--lora a.gguf --lora b.gguf --no-jinja --chat-template-kwargs '{"enable_thinking":false}'`,
      catalog,
    )
    expect(parsed.params.find((row) => row.key === 'lora').value).toEqual(['a.gguf', 'b.gguf'])
    expect(parsed.params.find((row) => row.key === 'jinja').value).toBe(false)
    expect(parsed.params.find((row) => row.key === 'chat_template_kwargs').value).toEqual({
      enable_thinking: false,
    })
  })

  it('warns when an enum value is not in the catalog options', () => {
    const parsed = parseCliCommand('--cache-type-k q4_0', catalog)
    expect(parsed.params[0].value).toBe('q4_0')
    expect(parsed.warnings.some((msg) => /not in the catalog options/i.test(msg))).toBe(true)
  })

  it('does not treat leftover Studio flags as custom-args leftovers', () => {
    const parsed = parseCliCommand('--port 9 --host 1.2.3.4 --unknown-x 1', catalog)
    expect(parsed.leftoverTokens).toEqual(['--unknown-x', '1'])
    expect(parsed.reserved).toHaveLength(2)
  })

  it('parses KEY=value environment assignments and ignores engine command words', () => {
    const parsed = parseCliCommand(
      [
        'CUDA_VISIBLE_DEVICES=0,1,2,3',
        'FLASHINFER_DISABLE_VERSION_CHECK=1',
        'NCCL_P2P_LEVEL=NVL',
        'SGLANG_ENABLE_SPEC_V2=1',
        'export SGLANG_MAMBA_CONV_DTYPE=float16',
        'PORT=8081',
        'HOST=0.0.0.0',
        'LLAMA_STUDIO_MODEL_PATH=/evil.gguf',
        'sglang',
        'serve',
        '--temp 0.3',
      ].join('\n'),
      catalog,
    )
    expect(parsed.env.map((row) => row.key).sort()).toEqual([
      'CUDA_VISIBLE_DEVICES',
      'FLASHINFER_DISABLE_VERSION_CHECK',
      'NCCL_P2P_LEVEL',
      'SGLANG_ENABLE_SPEC_V2',
      'SGLANG_MAMBA_CONV_DTYPE',
    ])
    expect(parsed.env.find((row) => row.key === 'CUDA_VISIBLE_DEVICES').value).toBe('0,1,2,3')
    expect(keysOf(parsed.params)).toEqual(['temperature'])
    expect(parsed.unknown).toEqual([])
    expect(parsed.reserved.some((row) => row.key === 'PORT')).toBe(true)
    expect(parsed.reserved.some((row) => row.key === 'HOST')).toBe(true)
    expect(parsed.reserved.some((row) => row.key === 'LLAMA_STUDIO_MODEL_PATH')).toBe(true)
    expect(parsed.reserved.find((row) => row.key === 'PORT').reason).toMatch(/listen address and port/i)
    expect(parsed.reserved.find((row) => row.key === 'LLAMA_STUDIO_MODEL_PATH').reason).toMatch(/LLAMA_STUDIO_/i)
  })
})

describe('buildImportPreview', () => {
  it('marks new vs update vs unchanged against current form values', () => {
    const parsed = parseCliCommand('--temp 0.7 --ctx-size 8192', catalog)
    const preview = buildImportPreview(parsed, { temperature: 0.7 })
    expect(preview.find((row) => row.key === 'temperature').change).toBe('unchanged')
    expect(preview.find((row) => row.key === 'ctx_size').change).toBe('new')
  })
})

describe('valuesEqual', () => {
  it('compares arrays and numeric strings loosely', () => {
    expect(valuesEqual([1, 2], [1, 2])).toBe(true)
    expect(valuesEqual(0.7, '0.7')).toBe(true)
    expect(valuesEqual({ a: 1 }, { a: 1 })).toBe(true)
  })
})
