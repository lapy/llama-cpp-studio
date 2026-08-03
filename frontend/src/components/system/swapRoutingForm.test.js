import { describe, it, expect } from 'vitest'
import {
  buildRoutingPayload,
  parseTargets,
  validateRoutingForm,
} from './swapRoutingForm'

describe('swapRoutingForm', () => {
  it('parses comma/semicolon/newline targets', () => {
    expect(parseTargets('a, b; c\nd')).toEqual(['a', 'b', 'c', 'd'])
    expect(parseTargets('  ')).toEqual([])
  })

  it('validates required ids, targets, pins, and collisions', () => {
    expect(
      validateRoutingForm(
        [{ id: '', strategy: 'warm', targetsText: '' }],
        [{ id: 'coding', pins: [{ key: '', target: '' }] }]
      )
    ).toEqual([
      'Selector 1: id is required',
      'Profile «coding»: at least one pin is required',
    ])

    const errors = validateRoutingForm(
      [
        { id: 'fast', strategy: 'warm', targetsText: 'model-a' },
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

  it('builds selector/profile payload including spillover settings', () => {
    const payload = buildRoutingPayload(
      [
        {
          id: 'busy',
          strategy: 'spillover',
          targetsText: 'a, b',
          spillover: 4,
          name: 'Busy',
          description: 'desc',
        },
        { id: '', strategy: 'pin', targetsText: 'ignored' },
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
})
