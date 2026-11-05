/**
 * R99: components read the engine store through SYNC ACCESSORS during render.
 *
 * A store stub that omits one does not fail a test assertion — Vue throws inside
 * the render and Vitest reports an "Unhandled Rejection" while the file still
 * counts as passed (R98 shipped 629 green tests + 14 unhandled errors that way).
 *
 * This pin scans the sources for `store.<accessor>ForEngine(...)` calls and fails
 * when the shared stub fixture does not provide that accessor, so adding a new
 * accessor to a panel forces the fixture (and therefore every stub) to follow.
 */

import { describe, expect, it } from 'vitest'
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { engineStoreAccessors } from '@/test-fixtures/engineStoreAccessors'

const SRC = join(process.cwd(), 'frontend/src')

function vueFilesUnder(dir) {
  const out = []
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry)
    if (statSync(path).isDirectory()) {
      if (entry === 'tests' || entry === 'node_modules') continue
      out.push(...vueFilesUnder(path))
    } else if (entry.endsWith('.vue')) {
      out.push(path)
    }
  }
  return out
}

const ACCESSOR_CALL = /\bstore\.([A-Za-z]\w*ForEngine)\s*\(/g

describe('engine store accessor contract', () => {
  it('provides every accessor components call during render', () => {
    const fixture = engineStoreAccessors()
    const missing = new Set()
    for (const dir of ['components', 'views']) {
      for (const file of vueFilesUnder(join(SRC, dir))) {
        const text = readFileSync(file, 'utf8')
        for (const match of text.matchAll(ACCESSOR_CALL)) {
          const name = match[1]
          if (typeof fixture[name] !== 'function') {
            missing.add(`${name} (called in ${file.replace(SRC, '')})`)
          }
        }
      }
    }
    expect([...missing]).toEqual([])
  })
})
