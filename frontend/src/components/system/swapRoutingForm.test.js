import { describe, it, expect } from 'vitest'
import {
  buildRoutingPayload,
  collectCatalogTargetIds,
  metadataFromRows,
  metadataToRows,
  normalizeTargetList,
  parseTargets,
  strategyHint,
  validateRoutingForm,
} from './swapRoutingForm'

describe('swapRoutingForm', () => {
  it('parses comma/semicolon/newline targets', () => {
    expect(parseTargets('a, b; c\nd')).toEqual(['a', 'b', 'c', 'd'])
    expect(parseTargets('  ')).toEqual([])
    expect(normalizeTargetList(['a', ' a ', '', 'b', 'a'])).toEqual(['a', 'b'])
  })

  it('validates required ids, targets, pins, and collisions', () => {
    expect(
      validateRoutingForm(
        [{ id: '', strategy: 'warm', targets: [] }],
        [{ id: 'coding', pins: [{ key: '', target: '' }] }]
      )
    ).toEqual([
      'Selector 1: id is required',
      'Profile «coding»: at least one pin is required',
    ])

    const errors = validateRoutingForm(
      [
        { id: 'fast', strategy: 'warm', targets: ['model-a'] },
        { id: 'fast', strategy: 'nope', targetsText: '' },
      ],
      [{ id: 'fast', pins: [{ key: 'llm', target: 'model-a' }] }]
    )
    expect(errors).toEqual(
      expect.arrayContaining([
        'Selector id «fast» is duplicated',
        'Selector «fast»: at least one target is required',
        'Selector «fast»: strategy must be warm, pin, or spillover',
        'Id «fast» is used by both a profile and a selector',
      ])
    )
  })

  it('builds selector/profile payload including spillover, unlisted, and metadata', () => {
    const payload = buildRoutingPayload(
      [
        {
          id: 'busy',
          strategy: 'spillover',
          targets: ['a', 'b'],
          spillover: 4,
          name: 'Busy',
          description: 'desc',
          unlisted: true,
          metadataRows: [
            { key: 'owner', value: 'team' },
            { key: '', value: 'ignored' },
          ],
        },
        { id: '', strategy: 'pin', targets: ['ignored'] },
      ],
      [
        {
          id: 'coding',
          description: 'Dev',
          pins: [
            { key: 'llm', target: 'busy' },
            { key: '', target: 'ignored' },
          ],
        },
      ]
    )
    expect(payload).toEqual({
      selectors: {
        busy: {
          strategy: 'spillover',
          targets: ['a', 'b'],
          name: 'Busy',
          description: 'desc',
          unlisted: true,
          metadata: { owner: 'team' },
          settings: { spillover: 4 },
        },
      },
      profiles: {
        coding: {
          description: 'Dev',
          pins: { llm: 'busy' },
        },
      },
    })
  })

  it('round-trips metadata rows and strategy hints', () => {
    expect(metadataToRows({ a: 1 })).toEqual([{ key: 'a', value: '1' }])
    expect(metadataFromRows([{ key: 'a', value: '1' }, { key: '', value: 'x' }])).toEqual({
      a: '1',
    })
    expect(strategyHint('warm')).toMatch(/already loaded/i)
    expect(strategyHint('spillover')).toMatch(/overflow/i)
  })

  it('collects catalog target ids from models and aliases', () => {
    const ids = collectCatalogTargetIds([
      {
        llama_swap_id: 'org-model.q4_k_m',
        routing_name: 'friendly',
        config: {
          model_alias: 'friendly',
          set_params_by_id: [{ sub_id: 'high' }],
          engines: {
            llama_cpp: { model_alias: 'engine-alias' },
          },
        },
      },
    ])
    expect(ids).toEqual(
      expect.arrayContaining([
        'org-model.q4_k_m',
        'friendly',
        'friendly:high',
        'high',
        'engine-alias',
      ])
    )
  })
})
