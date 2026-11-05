import { describe, expect, it } from 'vitest'
import { summarizeFrontendRun } from '../frontendGate'

const PASSING_RUN = `
 Test Files  71 passed (71)
      Tests  630 passed (630)
   Duration  55.00s
`

const ANSI_PASSING_RUN = `
\u001b[2m Test Files \u001b[22m \u001b[1m\u001b[32m71 passed\u001b[39m\u001b[22m (71)
\u001b[2m      Tests \u001b[22m \u001b[1m\u001b[32m630 passed\u001b[39m\u001b[22m (630)
`

const FALSE_GREEN_RUN = `
 Test Files  70 passed (70)
      Tests  629 passed (629)
     Errors  14 errors
\u2500 Unhandled Rejection \u2500
TypeError: store.versionsForEngine is not a function
`

const UNHANDLED_ONLY = `
 Tests  629 passed (629)
\u2500 Unhandled Errors \u2500
`

describe('summarizeFrontendRun', () => {
  it('accepts a clean run', () => {
    const summary = summarizeFrontendRun(PASSING_RUN, 0)
    expect(summary).toMatchObject({
      clean: true,
      exitCode: 0,
      unhandled: false,
      errorCount: false,
      passed: 630,
    })
  })

  it('reads counts through ANSI colour codes', () => {
    expect(summarizeFrontendRun(ANSI_PASSING_RUN, 0)).toMatchObject({
      clean: true,
      passed: 630,
    })
  })

  it('rejects the R98 false-green shape (tests passed but errors present)', () => {
    const summary = summarizeFrontendRun(FALSE_GREEN_RUN, 1)
    expect(summary.clean).toBe(false)
    expect(summary.unhandled).toBe(true)
    expect(summary.errorCount).toBe(true)
    expect(summary.passed).toBe(629)
  })

  it('rejects unhandled output even when the exit code is 0', () => {
    expect(summarizeFrontendRun(UNHANDLED_ONLY, 0).clean).toBe(false)
  })

  it('rejects a non-zero exit even without error text', () => {
    expect(summarizeFrontendRun(PASSING_RUN, 1).clean).toBe(false)
  })

  it('rejects a run where no test executed', () => {
    const summary = summarizeFrontendRun('Test Files  no tests\n', 0)
    expect(summary.clean).toBe(false)
    expect(summary.reason).toContain('no tests detected')
  })
})
