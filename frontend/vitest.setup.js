import { afterEach, beforeEach, vi } from 'vitest'

// Dropped DOM events when timers change
//
// Vue records the time a listener is attached and ignores a later DOM event
// whose `_vts` is not newer than that time. Vue Test Utils sets `_vts` to
// `Date.now() + 1` inside `trigger()`. That is normally enough. It is not
// enough when the listener was attached while fake timers were set ahead of
// the clock used for the event.
//
// The observed failure is a handler that never runs. Assertions then report
// zero calls, as with the audio-upload change event and a HeartMuLa generate
// click. `trigger()` itself does not throw. Mounting under `vi.useFakeTimers()`
// plus `vi.setSystemTime()` in the future, then calling `vi.useRealTimers()`
// and `trigger()`, reproduces it. A future `cachedNow` inside Vue can also
// stamp the next synchronous mount, so each test waits one microtask after
// restoring real timers before mounting.
//
// happy-dom overrides `dispatchEvent` on some prototypes. Those overrides call
// the original parent method, so wrapping `EventTarget.prototype` alone does
// not see the event. Wrap each override. Events that do not already carry
// `_vts` are left alone, which keeps Vue's protection against a bubble that
// attached the listener.
//
// Do not remove this when adjusting fake timers. A test that only passes with
// real timers, or only when run by itself, is the same failure.
function keepAttachedListenersReachable() {
  const doc = globalThis.document
  if (!doc?.createElement) return
  for (const tag of ['div', 'button', 'input', 'textarea', 'select']) {
    let proto = Object.getPrototypeOf(doc.createElement(tag))
    while (proto && proto !== Object.prototype) {
      const desc = Object.getOwnPropertyDescriptor(proto, 'dispatchEvent')
      if (desc && typeof desc.value === 'function' && !desc.value.__studioPatched) {
        const native = desc.value
        const wrapped = function dispatchEvent(event) {
          if (event && typeof event._vts === 'number' && event._vts < Number.MAX_SAFE_INTEGER) {
            event._vts = Number.MAX_SAFE_INTEGER
          }
          return native.call(this, event)
        }
        wrapped.__studioPatched = true
        Object.defineProperty(proto, 'dispatchEvent', {
          configurable: desc.configurable,
          enumerable: desc.enumerable,
          writable: desc.writable,
          value: wrapped,
        })
      }
      proto = Object.getPrototypeOf(proto)
    }
  }
}

// Fake timers are process-global and can leak across files in a reused worker.
// Vue also caches Date.now() until a microtask. A render under a future fake
// clock can make the next synchronous mount ignore DOM events, so flush that
// cache before each test mounts anything.
beforeEach(async () => {
  vi.useRealTimers()
  await Promise.resolve()
  keepAttachedListenersReachable()
})

afterEach(() => {
  vi.useRealTimers()
})
