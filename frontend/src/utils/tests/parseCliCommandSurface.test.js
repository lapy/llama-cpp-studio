/**
 * R97 pin: the CLI-import module's PUBLIC SURFACE is deliberate.
 *
 * The module used to export ~20 internal helpers (regexes, coercers, index
 * builders) that no consumer imported. They are module-local now; this pin keeps
 * the surface from drifting back (adding an export requires updating this list,
 * i.e. a reviewed diff).
 */

import { describe, expect, it } from 'vitest'
import * as cli from '../parseCliCommand'

const EXPECTED_EXPORTS = [
  'STUDIO_RESERVED_KEYS',
  'buildEnvImportPreview',
  'buildImportPreview',
  'delimitedEnumSeparator',
  'formatCliValue',
  'isDelimitedEnumParam',
  'joinCliTokens',
  'parseCliCommand',
  'studioEnvSkipReason',
  'tokenizeCli',
  'valuesEqual',
]

describe('parseCliCommand public surface', () => {
  it('exports only the consumed API', () => {
    expect(Object.keys(cli).sort()).toEqual(EXPECTED_EXPORTS)
  })
})
