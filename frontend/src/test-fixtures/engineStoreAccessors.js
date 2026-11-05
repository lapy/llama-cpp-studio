/**
 * R99: the synchronous store READ accessors that rendered components call.
 *
 * `SglangEnginePanel` (and any future panel) reads `versionsForEngine()` and
 * `statusForEngine()` during render. A store STUB that omits them does not fail
 * an assertion — it throws inside Vue's render and surfaces only as an
 * "Unhandled Rejection" while the test file still reports green (that is exactly
 * what happened in R98's sweep). Every EnginesView stub now spreads this fixture,
 * and `views/tests/engineStoreAccessorContract.test.js` fails if a component
 * calls an accessor this fixture does not provide.
 */
export function engineStoreAccessors() {
  return {
    engineStatusById: {},
    engineVersionsById: {},
    statusForEngine(engineId) {
      return this.engineStatusById?.[engineId] ?? {}
    },
    versionsForEngine(engineId) {
      return this.engineVersionsById?.[engineId] ?? []
    },
  }
}
