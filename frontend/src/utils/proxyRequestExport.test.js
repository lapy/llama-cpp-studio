import { describe, expect, it, vi } from 'vitest'
import {
  buildCurlCommand,
  buildHarExport,
  buildJsonExport,
  buildRequestUrl,
  downloadTextFile,
  exportFilename,
  hasCapturedPayload,
  requestBodyText,
  requestHeaders,
  shellQuote,
} from './proxyRequestExport'

const CONTINUATION = ' ' + String.fromCharCode(92) + '\n'

function captured(overrides = {}) {
  return {
    id: 'r1',
    state: 'completed',
    method: 'POST',
    path: '/v1/chat/completions',
    status: 200,
    http_status: 200,
    started_at: 1700000000.25,
    completed_at: 1700000000.262,
    latency_ms: 12,
    input_bytes: 56,
    output_bytes: 34,
    ...overrides,
  }
}

describe('proxy request export builders', () => {
  it('quotes values for a POSIX shell', () => {
    expect(shellQuote("it's")).toBe(`'it'"'"'s'`)
    expect(shellQuote('two words')).toBe(`'two words'`)
    expect(shellQuote('')).toBe(`''`)
    expect(shellQuote('$HOME `x`')).toBe("'$HOME `x`'")
  })

  it('normalizes captured headers from maps, pairs, and HAR objects', () => {
    expect(requestHeaders({ headers: { 'X-A': '1', 'x-b': ['2', '3'] } })).toEqual([
      { name: 'X-A', value: '1' },
      { name: 'x-b', value: '2, 3' },
    ])
    expect(requestHeaders({ headers: [['x-a', '1'], { name: 'x-b', value: '2' }, null] })).toEqual([
      { name: 'x-a', value: '1' },
      { name: 'x-b', value: '2' },
    ])
    expect(requestHeaders({ headers: 'not-a-header-list' })).toEqual([])
    expect(requestHeaders({})).toEqual([])
  })

  it('detects captured header and body payloads', () => {
    expect(hasCapturedPayload(captured())).toBe(false)
    expect(hasCapturedPayload(captured({ headers: { accept: '*/*' } }))).toBe(true)
    expect(hasCapturedPayload(captured({ body: '{"model":"demo"}' }))).toBe(true)
    expect(requestBodyText(captured({ body: { model: 'demo' } }))).toBe('{"model":"demo"}')
    expect(requestBodyText(captured({ body: '' }))).toBe(null)
    expect(requestBodyText(captured())).toBe(null)
  })

  it('builds an absolute URL from the current origin and keeps paths otherwise', () => {
    expect(buildRequestUrl(captured(), { origin: 'https://studio.example' })).toBe(
      'https://studio.example/v1/chat/completions',
    )
    expect(buildRequestUrl(captured(), { origin: 'https://studio.example/' })).toBe(
      'https://studio.example/v1/chat/completions',
    )
    expect(buildRequestUrl({ path: 'v1/models' }, { origin: 'http://127.0.0.1:8081' })).toBe(
      'http://127.0.0.1:8081/v1/models',
    )
    expect(
      buildRequestUrl(
        { path: 'https://upstream.example/v1/models' },
        { origin: 'https://studio.example' },
      ),
    ).toBe('https://upstream.example/v1/models')
    expect(buildRequestUrl({ path: '/v1/models' }, { origin: '' })).toBe('/v1/models')
    expect(buildRequestUrl({ path: '/v1/models' }, { origin: 'file:///tmp' })).toBe('/v1/models')
  })

  it('builds a method and URL command when no headers or body were captured', () => {
    const command = buildCurlCommand(captured({ method: 'get', path: '/v1/models' }), {
      origin: 'https://studio.example',
    })
    expect(command).toBe("curl --request GET 'https://studio.example/v1/models'")
    expect(command).not.toContain('--header')
    expect(command).not.toContain('--data')
  })

  it('includes captured headers and bodies with safe quoting', () => {
    const command = buildCurlCommand(
      captured({
        headers: { 'content-type': 'application/json', authorization: "Bearer a'b" },
        body: `{"prompt":"it's"}`,
      }),
      { origin: 'https://studio.example' },
    )
    expect(command).toBe(
      [
        "curl --request POST 'https://studio.example/v1/chat/completions'",
        `  --header 'content-type: application/json'`,
        `  --header 'authorization: Bearer a'"'"'b'`,
        `  --data '{"prompt":"it'"'"'s"}'`,
      ].join(CONTINUATION),
    )
    expect(
      buildCurlCommand(captured({ body: { model: 'demo' } }), {
        origin: 'https://studio.example',
      }),
    ).toContain(`--data '{"model":"demo"}'`)
  })

  it('pretty-prints the raw captured object as JSON', () => {
    const raw = captured({ nested: { keep: true } })
    const text = buildJsonExport(raw)
    expect(JSON.parse(text)).toEqual(raw)
    expect(text).toContain('\n  "id": "r1"')
    expect(text.endsWith('\n')).toBe(true)
    expect(buildJsonExport(null)).toBe('{}\n')
  })

  it('builds a valid single-entry HAR 1.2 document', () => {
    const har = buildHarExport(
      captured({
        path: '/v1/models?limit=2&tag=a%20b',
        headers: [['content-type', 'application/json']],
        body: '{"limit":2}',
      }),
      { origin: 'https://studio.example', appVersion: '9.9.9' },
    )
    expect(har.log.version).toBe('1.2')
    expect(har.log.creator).toEqual({ name: 'llama-cpp-studio', version: '9.9.9' })
    expect(har.log.entries).toHaveLength(1)
    const [entry] = har.log.entries
    expect(entry.startedDateTime).toBe(new Date(1700000000250).toISOString())
    expect(entry.time).toBe(12)
    expect(entry.request).toMatchObject({
      method: 'POST',
      url: 'https://studio.example/v1/models?limit=2&tag=a%20b',
      httpVersion: 'HTTP/1.1',
      headers: [{ name: 'content-type', value: 'application/json' }],
      queryString: [
        { name: 'limit', value: '2' },
        { name: 'tag', value: 'a b' },
      ],
      postData: { mimeType: 'application/json', text: '{"limit":2}' },
    })
    expect(entry.request.bodySize).toBe(11)
    expect(entry.response).toMatchObject({
      status: 200,
      statusText: '',
      httpVersion: 'HTTP/1.1',
      headers: [],
      content: { size: 34, mimeType: '' },
      redirectURL: '',
    })
    expect(entry.cache).toEqual({})
    expect(entry.timings).toEqual({ send: -1, wait: 12, receive: -1 })
    expect(JSON.parse(JSON.stringify(har))).toEqual(har)
  })

  it('keeps required HAR fields when captured metadata is missing', () => {
    const har = buildHarExport({}, { origin: 'https://studio.example' })
    const [entry] = har.log.entries
    expect(entry.request.method).toBe('GET')
    expect(entry.request.url).toBe('https://studio.example/')
    expect(entry.request.bodySize).toBe(0)
    expect(entry.request.postData).toBeUndefined()
    expect(entry.response.status).toBe(0)
    expect(entry.response.content).toEqual({ size: 0, mimeType: '' })
    expect(entry.time).toBe(0)
    expect(entry.timings).toEqual({ send: -1, wait: 0, receive: -1 })
    expect(entry.startedDateTime).toBe(new Date(0).toISOString())
  })

  it('falls back to the delivered status and timestamp fields that exist', () => {
    const har = buildHarExport(
      captured({ http_status: undefined, status: 502, started_at: undefined }),
      { origin: 'https://studio.example' },
    )
    const [entry] = har.log.entries
    expect(entry.response.status).toBe(502)
    expect(entry.startedDateTime).toBe(new Date(1700000000262).toISOString())
  })

  it('makes filesystem-safe request filenames', () => {
    expect(exportFilename('abc123', 'json')).toBe('request-abc123.json')
    expect(exportFilename('abc123', 'har')).toBe('request-abc123.har')
    expect(exportFilename('a/b:c d', 'json')).toBe('request-a_b_c_d.json')
    expect(exportFilename('../evil', 'json')).toBe('request-evil.json')
    expect(exportFilename('', 'json')).toBe('request-unknown.json')
    expect(exportFilename('id', 'exe')).toBe('request-id.json')
    expect(exportFilename('x'.repeat(200), 'har')).toHaveLength(
      'request-'.length + 96 + '.har'.length,
    )
  })

  it('downloads through a Blob object URL and revokes it', () => {
    const createObjectURL = vi.fn(() => 'blob:request-export')
    const revokeObjectURL = vi.fn()
    class TestURL extends URL {
      static createObjectURL(blob) {
        return createObjectURL(blob)
      }

      static revokeObjectURL(url) {
        revokeObjectURL(url)
      }
    }
    vi.stubGlobal('URL', TestURL)
    const click = vi.spyOn(HTMLElement.prototype, 'click').mockImplementation(() => {})

    downloadTextFile('{"id":"r1"}', 'request-r1.json')

    expect(createObjectURL).toHaveBeenCalledTimes(1)
    expect(createObjectURL.mock.calls[0][0]).toBeInstanceOf(Blob)
    expect(createObjectURL.mock.calls[0][0].type).toBe('application/json;charset=utf-8')
    expect(click).toHaveBeenCalledTimes(1)
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:request-export')
  })
})
