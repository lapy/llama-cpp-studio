// @vitest-environment node
/**
 * Unit-level contract check for the payloads the browser fixture serves.
 *
 * `frontend/e2e/fixtures/payloads.mjs` is imported by both Playwright and this
 * test, so a payload cannot drift without one of them noticing. For every
 * mocked response we resolve the committed OpenAPI operation and validate the
 * payload against its response schema.
 *
 * The committed document leaves most control-plane responses untyped
 * (`"schema": {}`), so for those endpoints `UI_SHAPES` pins the subset of
 * fields the fixture-driven journeys read. These slices are intentionally
 * shallow and are not a claim of full JSON-Schema compliance; when a route
 * grows a response model the OpenAPI schema takes over and the slice can go.
 */

import { describe, expect, it } from 'vitest'
import openapi from './openapi.json'
import {
  accessState,
  audioModelConfig,
  configHistory,
  createdInferenceKey,
  inferenceKeys,
  inferenceKeysResponse,
  inferenceSettings,
  invalidSession,
  llamaModelConfig,
  modelCatalog,
  paramRegistry,
  proxyRequests,
  proxySnapshot,
  referenceAudio,
  runtimePlanState,
  safetensorsModels,
  workspaceStatus,
} from '../../e2e/fixtures/payloads.mjs'

// ---------------------------------------------------------------------------
// Minimal schema validator (deliberately smaller than JSON Schema).
// Supports local $ref, allOf, anyOf/oneOf (at least one branch), type arrays,
// nullable-as-anyOf, arrays, required object fields, and bounded recursion.
// ---------------------------------------------------------------------------

const MAX_DEPTH = 4 // suffices for the nested fixture payloads; avoids cycles

function resolveRef(schema) {
  if (!schema?.$ref) return schema
  const target = schema.$ref
    .replace(/^#\//, '')
    .split('/')
    .map((part) => part.replaceAll('~1', '/').replaceAll('~0', '~'))
    .reduce((node, part) => node?.[part], openapi)
  if (!target) throw new Error(`cannot resolve OpenAPI $ref ${schema.$ref}`)
  return resolveRef(target)
}

function isEmptySchema(schema) {
  return Object.keys(resolveRef(schema) ?? {}).length === 0
}

function matchesType(value, type) {
  switch (type) {
    case 'null':
      return value === null
    case 'array':
      return Array.isArray(value)
    case 'object':
      return value !== null && typeof value === 'object' && !Array.isArray(value)
    case 'integer':
      return Number.isInteger(value)
    case 'number':
      return typeof value === 'number' && Number.isFinite(value)
    case 'string':
      return typeof value === 'string'
    case 'boolean':
      return typeof value === 'boolean'
    default:
      return true // unknown keyword: stay permissive, as documented above
  }
}

function describeValue(value) {
  if (value === null) return 'null'
  if (Array.isArray(value)) return 'array'
  if (typeof value === 'string') return `string ${JSON.stringify(value)}`
  if (typeof value === 'number') return `number ${value}`
  return typeof value
}

function validate(value, rawSchema, path = '$', depth = 0) {
  const schema = resolveRef(rawSchema ?? {})
  const errors = []
  if (schema.nullable === true && value === null) return errors

  const variants = schema.anyOf ?? schema.oneOf
  if (Array.isArray(variants)) {
    const matches = variants.some((variant) => validate(value, variant, path, depth).length === 0)
    if (!matches) errors.push(`${path}: does not match any of the allowed schemas`)
  }
  for (const part of schema.allOf ?? []) {
    errors.push(...validate(value, part, path, depth))
  }

  const types = schema.type === undefined ? [] : [].concat(schema.type)
  if (types.length && !types.some((type) => matchesType(value, type))) {
    errors.push(`${path}: expected ${types.join(' | ')}, got ${describeValue(value)}`)
    return errors
  }

  if (Array.isArray(value) && depth < MAX_DEPTH && schema.items) {
    value.forEach((item, index) => {
      errors.push(...validate(item, schema.items, `${path}[${index}]`, depth + 1))
    })
  }

  if (value !== null && typeof value === 'object' && !Array.isArray(value) && depth < MAX_DEPTH) {
    for (const key of schema.required ?? []) {
      if (!Object.hasOwn(value, key)) errors.push(`${path}.${key}: missing required property`)
    }
    for (const [key, propertySchema] of Object.entries(schema.properties ?? {})) {
      if (!Object.hasOwn(value, key)) continue
      errors.push(...validate(value[key], propertySchema, `${path}.${key}`, depth + 1))
    }
  }

  return errors
}

function assertContract(label, value, schema) {
  const errors = validate(value, schema)
  if (errors.length) {
    throw new Error(`${label} does not satisfy its contract:\n  ${errors.join('\n  ')}`)
  }
}

// ---------------------------------------------------------------------------
// OpenAPI lookup
// ---------------------------------------------------------------------------

function matchesTemplate(template, path) {
  const templateParts = template.split('/')
  const pathParts = path.split('/')
  return (
    templateParts.length === pathParts.length &&
    templateParts.every((part, index) => /^\{[^}]+\}$/.test(part) || part === pathParts[index])
  )
}

function findOperation(method, path) {
  const exact = openapi.paths[path]
  if (exact?.[method.toLowerCase()]) {
    return { template: path, operation: exact[method.toLowerCase()] }
  }
  for (const [template, item] of Object.entries(openapi.paths)) {
    if (!matchesTemplate(template, path)) continue
    const operation = item[method.toLowerCase()]
    if (operation) return { template, operation }
  }
  return null
}

function responseSchema(operation) {
  const responses = operation.responses ?? {}
  const success =
    responses['200'] ??
    responses['201'] ??
    responses['202'] ??
    responses['204'] ??
    Object.entries(responses).find(([status]) => /^2\d\d$/.test(status))?.[1] ??
    responses.default
  if (!success) throw new Error('OpenAPI operation has no success response')
  return success.content?.['application/json']?.schema ?? {}
}

// ---------------------------------------------------------------------------
// UI-consumed slices for responses the committed document leaves untyped.
// ---------------------------------------------------------------------------

const string = { type: 'string' }
const number = { type: 'number' }
const integer = { type: 'integer' }
const boolean = { type: 'boolean' }
const arrayOf = (items = {}) => ({ type: 'array', items })
const objectOf = (properties, required = Object.keys(properties)) => ({
  type: 'object',
  properties,
  required,
})

const proxyRequestRow = objectOf({
  id: string,
  started_at: number,
  completed_at: number,
  model: string,
  method: string,
  path: string,
  state: string,
  status: integer,
  latency_ms: integer,
  input_tokens: integer,
  output_tokens: integer,
  usage_source: string,
  key_name: string,
})

const UI_SHAPES = {
  'GET /api/access': objectOf({ mode: string, authenticated: boolean }),
  'GET /api/status': objectOf({
    proxy_status: objectOf({ backend: string, healthy: boolean }),
    persistence: objectOf({}),
    runtime_observation: objectOf({ quality: string }),
  }),
  // Model list consumers read these fields (stores/models.js renders groups
  // and quantizations; pickers use id/name/is_active).
  'GET /api/models': arrayOf(
    objectOf({
      base_model_name: string,
      huggingface_id: string,
      quantizations: arrayOf(
        objectOf({
          id: string,
          name: string,
          format: string,
          is_active: boolean,
          proxy_name: string,
        }),
      ),
    }),
  ),
  'GET /api/models/param-registry': objectOf({ sections: arrayOf() }),
  'GET /api/models/speech-model/reference-audio': arrayOf({ type: 'object' }),
  'GET /api/models/speech-model/config': objectOf({
    engine: string,
    family: string,
    task: string,
    model_alias: string,
    speech_defaults: objectOf({}),
  }),
  'GET /api/models/demo-model/config': objectOf({
    engine: string,
    engines: objectOf({ llama_cpp: objectOf({ ctx_size: integer }) }),
  }),
  // ProxyView reads every field below for its five tabs.
  'GET /api/proxy': objectOf({
    settings: objectOf({ backend: string, allow_anonymous: boolean, proxy_port: integer }),
    configured_backend: string,
    active_backend: string,
    running: arrayOf(
      objectOf({
        model: string,
        engine: string,
        state: string,
        ready: boolean,
        pid: integer,
        port: integer,
        inflight: integer,
        request_count: integer,
        client_error_count: integer,
        server_error_count: integer,
      }),
    ),
    catalog: arrayOf(objectOf({ id: string, studio_model_id: string })),
    requests: arrayOf(proxyRequestRow),
    request_total: integer,
    summary: objectOf({
      requests: integer,
      client_errors: integer,
      server_errors: integer,
      input_tokens: integer,
      output_tokens: integer,
      unknown_usage: integer,
    }),
    activity: arrayOf(),
  }),
  'GET /api/proxy/requests': objectOf({
    requests: arrayOf(proxyRequestRow),
    total: integer,
  }),
  // Access-keys tab and ConnectDialog.
  'GET /api/inference/keys': objectOf({
    keys: arrayOf(
      objectOf({
        id: string,
        name: string,
        prefix: string,
        scopes: arrayOf(string),
        models: arrayOf(),
      }),
    ),
  }),
  'POST /api/inference/keys': objectOf({ key: string }),
  'GET /api/runtime/stale': objectOf({
    applicable: boolean,
    pending: boolean,
    stale: boolean,
  }),
  'GET /api/runtime/pending': objectOf({
    applicable: boolean,
    pending: boolean,
    stale: boolean,
  }),
  'GET /api/settings/inference': objectOf({ public_inference_url: string }),
  'GET /api/models/safetensors': arrayOf(),
  // The remote-mode mock is a 401; OpenAPI only documents the 200 branch.
  'POST /api/session': objectOf({ detail: string }),
}

// ---------------------------------------------------------------------------
// Cases: one per response shape `installWorkspaceApi` serves.
//
// `installWorkspaceApi` has no branches for `/api/llama-versions/cuda-status`
// or `/api/engines` at this revision (both fall through to the generic `{}`),
// so there is no fixture payload to check. Their committed OpenAPI responses
// are untyped too, so adding routes alone would not create a schema contract.
// ---------------------------------------------------------------------------

const FIXED_NOW = 1_700_000_000

const CONTRACT_CASES = [
  { method: 'GET', path: '/api/access', payload: accessState() },
  { method: 'GET', path: '/api/status', payload: workspaceStatus() },
  { method: 'GET', path: '/api/models', payload: modelCatalog() },
  { method: 'GET', path: '/api/models/param-registry', payload: paramRegistry() },
  {
    method: 'GET',
    path: '/api/models/speech-model/reference-audio',
    payload: referenceAudio(),
  },
  { method: 'GET', path: '/api/models/speech-model/config', payload: audioModelConfig() },
  { method: 'GET', path: '/api/models/demo-model/config', payload: llamaModelConfig() },
  { method: 'GET', path: '/api/proxy', payload: proxySnapshot({ now: FIXED_NOW }) },
  { method: 'GET', path: '/api/proxy/requests', payload: proxyRequests({ now: FIXED_NOW }) },
  {
    method: 'GET',
    path: '/api/inference/keys',
    payload: inferenceKeysResponse(inferenceKeys()),
  },
  { method: 'POST', path: '/api/inference/keys', payload: createdInferenceKey('sk-studio-copy') },
  { method: 'GET', path: '/api/runtime/stale', payload: runtimePlanState() },
  { method: 'GET', path: '/api/runtime/pending', payload: runtimePlanState() },
  { method: 'GET', path: '/api/settings/inference', payload: inferenceSettings() },
  { method: 'GET', path: '/api/config-history', payload: configHistory() },
  { method: 'GET', path: '/api/models/safetensors', payload: safetensorsModels() },
  {
    method: 'POST',
    path: '/api/session',
    label: 'POST /api/session (401)',
    payload: invalidSession(),
  },
]

describe('e2e workspace payloads match the committed API contract', () => {
  for (const endpoint of CONTRACT_CASES) {
    const label = endpoint.label ?? `${endpoint.method} ${endpoint.path}`
    it(`${label} matches its OpenAPI or documented UI contract`, () => {
      const found = findOperation(endpoint.method, endpoint.path)
      expect(found, `${label}: no matching operation in frontend/src/api/openapi.json`).toBeTruthy()

      const schema = responseSchema(found.operation)
      const uiSchema = UI_SHAPES[`${endpoint.method} ${endpoint.path}`]
      if (isEmptySchema(schema) && !uiSchema) {
        throw new Error(
          `${label}: OpenAPI declares no response schema; add a UI_SHAPES entry ` +
            'for the fields this mock serves',
        )
      }
      if (!isEmptySchema(schema)) {
        assertContract(`${label} (OpenAPI ${found.template})`, endpoint.payload, schema)
      }
      if (uiSchema) assertContract(`${label} (UI-consumed fields)`, endpoint.payload, uiSchema)
    })
  }
})

describe('validator scope', () => {
  it('resolves $ref and reports every missing required field with its path', () => {
    const entrySchema = { $ref: '#/components/schemas/HistoryEntryResponse' }
    expect(
      validate(
        {
          id: 'history-1',
          created_at: '2026-01-01T00:00:00Z',
          document: 'preferences',
          reason: 'saved',
        },
        entrySchema,
      ),
    ).toEqual([])
    const errors = validate({}, entrySchema)
    expect(errors).toContain('$.id: missing required property')
    expect(errors).toContain('$.created_at: missing required property')
  })

  it('accepts nullable anyOf and every allOf branch, rejecting a wrong branch', () => {
    expect(validate(null, { anyOf: [string, { type: 'null' }] })).toEqual([])
    const schema = {
      allOf: [
        { type: 'object', required: ['a'], properties: { a: number } },
        { type: 'object', required: ['b'], properties: { b: string } },
      ],
    }
    expect(validate({ a: 1, b: 'x' }, schema)).toEqual([])
    expect(validate({ a: '1', b: 'x' }, schema)).toEqual(['$.a: expected number, got string "1"'])
  })

  it('recurses through arrays into nested objects', () => {
    const errors = validate([{ n: '1' }], {
      type: 'array',
      items: { type: 'object', required: ['n'], properties: { n: number } },
    })
    expect(errors).toEqual(['$[0].n: expected number, got string "1"'])
  })
})
