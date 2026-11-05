/**
 * R100: the frontend gate's REPORT must not hide broken execution.
 *
 * Vitest exits non-zero when a suite raises unhandled rejections, but a run can
 * still print "Tests N passed" (that is how R98 shipped a false-green frontend
 * claim). This module turns the runner output into an explicit verdict so the
 * gate fails on the exit code, on any unhandled error / error count, and when no
 * test actually ran.
 */

// eslint-disable-next-line no-control-regex -- ANSI escapes are control characters by definition
const ANSI = /\u001b\[[0-9;]*m/g
const UNHANDLED = /Unhandled(?: Rejection| Error| Errors)?/i
const ERROR_COUNT = /\bErrors?\s+\d+\s+error/i
const PASSED_COUNT = /\bTests\s+(\d+)\s+passed/i

function stripAnsi(text) {
  return String(text || '').replace(ANSI, '')
}

export function summarizeFrontendRun(output, exitCode) {
  const text = stripAnsi(output)
  const unhandled = UNHANDLED.test(text)
  const errorCount = ERROR_COUNT.test(text)
  const match = PASSED_COUNT.exec(text)
  const passed = match ? Number(match[1]) : 0
  const noTests = passed <= 0
  const clean = exitCode === 0 && !unhandled && !errorCount && !noTests
  return {
    clean,
    exitCode,
    unhandled,
    errorCount,
    passed,
    reason: clean
      ? `exit 0, ${passed} tests passed, no unhandled errors`
      : [
          exitCode !== 0 ? `exit ${exitCode}` : '',
          unhandled ? 'unhandled rejection/error in output' : '',
          errorCount ? 'error section in output' : '',
          noTests ? 'no tests detected' : '',
        ]
          .filter(Boolean)
          .join('; '),
  }
}

/** Formats the verdict for the terminal (used by scripts/frontend-gate.mjs). */
export function formatVerdict(summary) {
  return summary.clean
    ? `frontend gate: CLEAN (${summary.reason})`
    : `frontend gate: FAILED (${summary.reason})`
}
