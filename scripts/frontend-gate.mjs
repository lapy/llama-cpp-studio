#!/usr/bin/env node
/**
 * R100: frontend gate runner.
 *
 * Runs the full Vitest suite and fails on the EXIT CODE **and** on any unhandled
 * error or error section — the assertion count alone is not a verdict (R98).
 *
 * The suite is launched through the current Node executable with Vitest's own
 * entry script rather than an `npx` shim: Node refuses to spawn `.cmd`
 * wrappers without a shell (EINVAL on Windows), so the old `npx.cmd` command
 * could never produce a verdict on a Windows host.
 *
 * Usage: node scripts/frontend-gate.mjs
 */

import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

import { formatVerdict, summarizeFrontendRun } from '../frontend/src/utils/frontendGate.js'

const vitestEntry = fileURLToPath(new URL('../node_modules/vitest/vitest.mjs', import.meta.url))
const configPath = fileURLToPath(new URL('../frontend/vite.config.mjs', import.meta.url))
if (!existsSync(vitestEntry)) {
  console.error(`vitest entry not found: ${vitestEntry} (run npm install first)`)
  process.exit(1)
}

const args = [vitestEntry, 'run', '--config', configPath, ...process.argv.slice(2)]

const child = spawn(process.execPath, args, { stdio: ['ignore', 'pipe', 'pipe'] })
let output = ''

child.stdout.on('data', (chunk) => {
  output += chunk.toString()
  process.stdout.write(chunk)
})
child.stderr.on('data', (chunk) => {
  output += chunk.toString()
  process.stderr.write(chunk)
})

child.on('close', (code) => {
  const summary = summarizeFrontendRun(output, code ?? 1)
  const verdict = formatVerdict(summary)
  console.log(verdict)
  process.exit(summary.clean ? 0 : 1)
})
