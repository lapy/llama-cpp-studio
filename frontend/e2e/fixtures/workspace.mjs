export async function installWorkspaceApi(
  page,
  { bridgeOrigin, remote = false, key = 'sk-studio-copy-probe' } = {},
) {
  const keys = [
    {
      id: 'production',
      name: 'Production',
      prefix: 'sk-studio-preview',
      scopes: ['inference', 'models:read'],
      models: [],
    },
  ]
  await page.route('**/*', async (route) => {
    const request = route.request(),
      url = new URL(request.url()),
      path = url.pathname
    if (!path.startsWith('/api/')) {
      if (bridgeOrigin && url.origin === bridgeOrigin) {
        const response = await route.fetch({
          url: `http://127.0.0.1:4179${path}${url.search}`,
          // Firefox preserves the original Host header when route.fetch changes URL.
          headers: { ...request.headers(), host: '127.0.0.1:4179' },
        })
        return route.fulfill({ response })
      }
      return route.continue()
    }
    const now = Date.now() / 1000
    const rows = [
      {
        id: 'request-1',
        started_at: now - 30,
        completed_at: now - 29,
        model: 'demo-model',
        method: 'POST',
        path: '/v1/chat/completions',
        state: 'completed',
        status: 200,
        latency_ms: 150,
        input_tokens: 10,
        output_tokens: 20,
        usage_source: 'reported',
        key_name: 'Production',
      },
    ]
    let data = {}
    if (path === '/api/session' && remote)
      return route.fulfill({
        status: 401,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'Invalid password' }),
      })
    if (path === '/api/access') data = { mode: remote ? 'remote' : 'local', authenticated: !remote }
    else if (path === '/api/status')
      data = {
        proxy_status: {
          backend: 'studio',
          healthy: true,
          health_observed_at: new Date().toISOString(),
        },
        persistence: {},
        runtime_observation: { quality: 'verified', observed_at: new Date().toISOString() },
      }
    else if (path === '/api/models')
      data = [
        {
          base_model_name: 'Demo',
          huggingface_id: 'org/demo',
          quantizations: [
            {
              id: 'demo-model',
              name: 'Demo',
              quantization: 'Q4_K_M',
              format: 'gguf',
              is_active: true,
              run_state: 'running',
              status: 'ready',
              runtime_quality: 'verified',
              config: { engine: 'llama_cpp' },
              proxy_name: 'demo-model',
            },
          ],
        },
        {
          base_model_name: 'Speech demo',
          huggingface_id: 'org/speech',
          quantizations: [
            {
              id: 'speech-model',
              name: 'Speech demo',
              display_name: 'Speech demo',
              format: 'audio_cpp',
              engine: 'audio_cpp',
              is_active: true,
              runtime_quality: 'verified',
              config: { engine: 'audio_cpp' },
              proxy_name: 'speech-model',
            },
          ],
        },
      ]
    else if (path === '/api/models/param-registry') data = { sections: [] }
    else if (path.endsWith('/reference-audio')) data = []
    else if (path === '/api/models/speech-model/config')
      data = {
        engine: 'audio_cpp',
        family: 'pocket-tts',
        task: 'tts',
        model_alias: 'speech-model',
        speech_defaults: {},
      }
    else if (path === '/api/models/demo-model/config')
      data = { engine: 'llama_cpp', engines: { llama_cpp: { ctx_size: 4096 } } }
    else if (path === '/api/proxy')
      data = {
        settings: { backend: 'studio', allow_anonymous: false, proxy_port: 2000 },
        configured_backend: 'studio',
        active_backend: 'studio',
        running: [
          {
            model: 'demo-model',
            engine: 'llama_cpp',
            state: 'ready',
            ready: true,
            pid: 42,
            port: 3000,
            inflight: 0,
            request_count: 1,
            client_error_count: 0,
            server_error_count: 0,
          },
        ],
        catalog: [{ id: 'demo-model', studio_model_id: 'demo-model' }],
        requests: rows,
        request_total: 1,
        summary: {
          requests: 1,
          client_errors: 0,
          server_errors: 0,
          input_tokens: 10,
          output_tokens: 20,
          unknown_usage: 0,
        },
        activity: [],
      }
    else if (path === '/api/proxy/requests') data = { requests: rows, total: 1 }
    else if (path === '/api/inference/keys') {
      if (request.method() === 'POST') {
        keys.push({ ...request.postDataJSON(), id: 'new-key', prefix: 'sk-studio-copy' })
        data = { key }
      } else data = { keys }
    } else if (path === '/api/llama-swap/stale' || path === '/api/llama-swap/pending')
      data = { applicable: false, pending: false, stale: false }
    else if (path === '/api/settings/inference') data = { public_inference_url: '' }
    else if (path === '/api/config-history') data = { entries: [] }
    else if (path === '/api/models/safetensors') data = []
    await route.fulfill({ contentType: 'application/json', body: JSON.stringify(data) })
  })
}
