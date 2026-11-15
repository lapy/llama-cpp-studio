import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { eventSourceFactory, useProgressStore } from './progress.js'
import {
  OFFLINE_AFTER_ATTEMPTS,
  RECONNECT_BASE_DELAY_MS,
  RECONNECT_JITTER_RATIO,
  RECONNECT_MAX_DELAY_MS,
  STALE_CHECK_INTERVAL_MS,
  STALE_CONNECTION_MS,
  reconnectDelay,
} from '@/utils/liveUpdates'

class FakeEventSource {
  constructor(url) {
    this.url = url
    this.readyState = 0
    this.closed = false
    this.handlers = new Map()
    instances.push(this)
  }

  addEventListener(type, handler) {
    this.handlers.set(type, handler)
  }

  close() {
    this.closed = true
    this.readyState = 2
  }

  emit(type, payload) {
    this.handlers.get(type)?.({ data: JSON.stringify(payload) })
  }

  open() {
    this.readyState = 1
    this.onopen?.()
  }

  fail() {
    this.readyState = 2
    this.onerror?.({ type: 'error' })
  }
}

let instances
let openSpy

function baseDelayFor(attempt) {
  return Math.min(RECONNECT_MAX_DELAY_MS, RECONNECT_BASE_DELAY_MS * 2 ** (attempt - 1))
}

describe('progress store connection model', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localStorage.removeItem('llama-studio.activity.dismissed')
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-11-15T12:00:00Z'))
    instances = []
    openSpy = vi
      .spyOn(eventSourceFactory, 'open')
      .mockImplementation((url) => new FakeEventSource(url))
  })

  afterEach(() => {
    vi.useRealTimers()
    openSpy?.mockRestore()
  })

  it('opens exactly one EventSource even if connect is called repeatedly', () => {
    const store = useProgressStore()

    store.connect()
    store.connect()
    expect(instances).toHaveLength(1)

    instances[0].open()
    store.connect()
    expect(instances).toHaveLength(1)
    expect(store.connectionState).toBe('live')
    expect(store.isConnected).toBe(true)
    expect(store.reconnectAttempt).toBe(0)
    expect(store.lastConnectedAt).toBe(Date.now())
  })

  it('backs off 1s, 2s, 4s ... with jitter and resets on a successful open', async () => {
    vi.spyOn(Math, 'random').mockReturnValue(0)
    const store = useProgressStore()
    store.connect()

    const expectedDelays = [800, 1600, 3200, 6400, 12800, 24000, 24000]
    for (let attempt = 1; attempt <= expectedDelays.length; attempt += 1) {
      instances[attempt - 1].fail()
      expect(store.reconnectAttempt).toBe(attempt)
      expect(store.connectionState).toBe(
        attempt >= OFFLINE_AFTER_ATTEMPTS ? 'offline' : 'connecting',
      )

      const delay = expectedDelays[attempt - 1]
      await vi.advanceTimersByTimeAsync(delay - 1)
      expect(instances).toHaveLength(attempt)
      await vi.advanceTimersByTimeAsync(1)
      expect(instances).toHaveLength(attempt + 1)
    }

    instances.at(-1).open()
    expect(store.reconnectAttempt).toBe(0)
    expect(store.connectionState).toBe('live')
    expect(store.disconnectedSince).toBeNull()
  })

  it('keeps every retry delay inside its jitter-bounded and capped window', () => {
    for (const attempt of [1, 2, 3, 4, 5, 6, 10, 50]) {
      const base = baseDelayFor(attempt)
      const floor = Math.round(base * (1 - RECONNECT_JITTER_RATIO))
      const ceiling = Math.min(
        RECONNECT_MAX_DELAY_MS,
        Math.round(base * (1 + RECONNECT_JITTER_RATIO)),
      )
      for (const random of [0, 0.25, 0.5, 0.75, 1]) {
        const delay = reconnectDelay(attempt, () => random)
        expect(delay).toBeGreaterThanOrEqual(floor)
        expect(delay).toBeLessThanOrEqual(ceiling)
        expect(delay).toBeLessThanOrEqual(RECONNECT_MAX_DELAY_MS)
      }
    }

    expect(reconnectDelay(1, () => 0)).toBe(800)
    expect(reconnectDelay(1, () => 1)).toBe(1200)
    expect(reconnectDelay(2, () => 0)).toBe(1600)
    expect(reconnectDelay(6, () => 0)).toBe(24000)
    expect(reconnectDelay(6, () => 1)).toBe(RECONNECT_MAX_DELAY_MS)
    expect(reconnectDelay(50, () => 0.5)).toBe(RECONNECT_MAX_DELAY_MS)
  })

  it('retryConnection cancels the pending timer and reconnects immediately', async () => {
    vi.spyOn(Math, 'random').mockReturnValue(0)
    const store = useProgressStore()
    store.connect()
    instances[0].fail()
    expect(store.reconnectAttempt).toBe(1)

    store.retryConnection()

    expect(instances).toHaveLength(2)
    expect(instances[0].closed).toBe(true)
    expect(store.reconnectAttempt).toBe(0)
    expect(vi.getTimerCount()).toBe(0)

    await vi.advanceTimersByTimeAsync(10_000)
    expect(instances).toHaveLength(2)

    instances[1].open()
    expect(store.connectionState).toBe('live')
  })

  it('reconnects after the stale threshold while tasks are still active', async () => {
    vi.spyOn(Math, 'random').mockReturnValue(0)
    const store = useProgressStore()
    store.handleEvent('task_created', {
      task_id: 'build_1',
      type: 'build',
      status: 'running',
      progress: 10,
    })
    store.connect()
    instances[0].open()
    expect(store.hasActiveTasks).toBe(true)

    await vi.advanceTimersByTimeAsync(STALE_CONNECTION_MS - 1)
    expect(instances).toHaveLength(1)

    // The 45s watchdog tick declares the stream stale, closes it, and starts
    // the normal first retry delay.
    await vi.advanceTimersByTimeAsync(1)
    expect(instances[0].closed).toBe(true)
    expect(store.reconnectAttempt).toBe(1)

    await vi.advanceTimersByTimeAsync(800)
    expect(instances).toHaveLength(2)
    expect(store.connectionState).toBe('connecting')
  })

  it('leaves an idle stream alone even after the stale threshold', async () => {
    const store = useProgressStore()
    store.connect()
    instances[0].open()

    await vi.advanceTimersByTimeAsync(STALE_CONNECTION_MS + 2 * STALE_CHECK_INTERVAL_MS)

    expect(instances).toHaveLength(1)
    expect(instances[0].closed).toBe(false)
    expect(store.connectionState).toBe('live')
  })

  it('updates lastEventAt for framed events and named heartbeats', async () => {
    const store = useProgressStore()
    store.connect()
    instances[0].open()
    const openedAt = store.lastEventAt

    await vi.advanceTimersByTimeAsync(5_000)
    store.handleEvent('task_updated', { task_id: 'task_1', status: 'running' })
    expect(store.lastEventAt).toBe(openedAt + 5_000)

    await vi.advanceTimersByTimeAsync(3_000)
    instances[0].emit('heartbeat', { type: 'heartbeat' })
    expect(store.lastEventAt).toBe(openedAt + 8_000)
  })

  it('disconnect stops pending retries and the stale watchdog', async () => {
    const store = useProgressStore()
    store.connect()
    instances[0].open()
    store.handleEvent('task_created', {
      task_id: 'build_2',
      type: 'build',
      status: 'running',
    })
    store.disconnect()

    await vi.advanceTimersByTimeAsync(STALE_CONNECTION_MS * 3)

    expect(instances).toHaveLength(1)
    expect(store.isConnected).toBe(false)
    expect(store.connectionState).toBe('offline')
    expect(vi.getTimerCount()).toBe(0)
  })
})
