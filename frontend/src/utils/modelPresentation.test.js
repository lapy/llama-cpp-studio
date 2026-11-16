import { describe, expect, it } from 'vitest'
import { AUDIO_PACKAGE } from './audioModel'
import { groupRuntimeState, modelNeedsSetup, modelRuntimeState } from './modelPresentation'

describe('runtime presentation', () => {
  it.each(['stale', 'unreachable', 'unknown'])(
    'never presents a %s observation as ready',
    (quality) => {
      const state = modelRuntimeState({
        status: 'ready',
        is_active: true,
        runtime_quality: quality,
      })
      expect(state.actionable).toBe(false)
      expect(state.label).toBe('Status unavailable')
      expect(state.severity).toBe('warn')
    },
  )
  it('surfaces incomplete observations ahead of a ready variant in a group', () => {
    expect(
      groupRuntimeState({
        quantizations: [{ status: 'ready', is_active: true }, { runtime_quality: 'unreachable' }],
      }).key,
    ).toBe('unknown')
  })
  it('uses one Running label for a ready server state and an active process', () => {
    expect(modelRuntimeState({ status: 'ready', is_active: true })).toMatchObject({
      key: 'running',
      label: 'Running',
      actionable: true,
    })
    expect(modelRuntimeState({ status: 'ready', is_active: false }).label).toBe('Running')
    expect(modelRuntimeState({ is_active: true }).label).toBe('Running')
  })
  it('uses one Starting label for loading and in-flight starts', () => {
    expect(modelRuntimeState({ status: 'loading', is_active: true })).toMatchObject({
      key: 'starting',
      label: 'Starting',
      actionable: false,
    })
    expect(modelRuntimeState({ status: 'stopped' }, { isStarting: true })).toMatchObject({
      key: 'starting',
      label: 'Starting',
    })
    expect(modelRuntimeState({ is_active: true }, { isStopping: true }).label).toBe('Stopping')
  })
  it('distinguishes loading, running and a stopped model', () => {
    expect(modelRuntimeState({ status: 'loading', is_active: true }).actionable).toBe(false)
    expect(modelRuntimeState({ status: 'ready', is_active: true }).actionable).toBe(true)
    expect(modelRuntimeState({ is_active: false }).label).toBe('Stopped')
  })
  it('carries the failure detail used by the inline error and tooltip', () => {
    expect(modelRuntimeState({ status: 'failed', last_error: 'CUDA out of memory' })).toMatchObject(
      {
        key: 'error',
        label: 'Failed',
        detail: 'CUDA out of memory',
      },
    )
  })
})

describe('setup presentation', () => {
  it('only asks for setup on an unreviewed stopped text model', () => {
    const stopped = { status: 'stopped', is_active: false, config_reviewed: false }
    expect(modelNeedsSetup(stopped)).toBe(true)
    expect(modelNeedsSetup({ ...stopped, config_reviewed: true })).toBe(false)
    // Older payloads and search results without provenance do not claim setup is needed.
    expect(modelNeedsSetup({ status: 'stopped', is_active: false })).toBe(false)
  })

  it('asks for setup when an inactive unreviewed model state is unknown or stale', () => {
    const unreviewed = { config_reviewed: false, is_active: false }
    expect(modelNeedsSetup({ ...unreviewed, runtime_quality: 'unknown' })).toBe(true)
    expect(modelNeedsSetup({ ...unreviewed, runtime_quality: 'stale' })).toBe(true)
    expect(modelNeedsSetup({ ...unreviewed, runtime_quality: 'unreachable' })).toBe(true)
    expect(modelNeedsSetup({ ...unreviewed, runtime_quality: 'unknown', is_active: true })).toBe(
      false,
    )
  })

  it('leaves running, starting and failed models to their state primary', () => {
    const unreviewed = { config_reviewed: false }
    expect(modelNeedsSetup({ ...unreviewed, status: 'ready', is_active: true })).toBe(false)
    expect(modelNeedsSetup({ ...unreviewed, status: 'loading' })).toBe(false)
    expect(modelNeedsSetup({ ...unreviewed, status: 'failed' })).toBe(false)
  })

  it('keeps audio packages on the audio path', () => {
    expect(
      modelNeedsSetup({
        status: 'stopped',
        config_reviewed: false,
        format: AUDIO_PACKAGE,
      }),
    ).toBe(false)
  })
})
