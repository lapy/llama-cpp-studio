import {
  accessState,
  audioContract,
  AUDIO_CONTRACT_PATH,
  audioModelConfig,
  audioVoices,
  configHistory,
  createdInferenceKey,
  engineDescriptors,
  inferenceKeyFromRequest,
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
} from './payloads.mjs'

export async function installWorkspaceApi(
  page,
  { bridgeOrigin, remote = false, key = 'sk-studio-copy-probe', audio = false } = {},
) {
  const keys = inferenceKeys()
  await page.route('**/*', async (route) => {
    const request = route.request(),
      url = new URL(request.url()),
      path = url.pathname
    if (path.startsWith('/v1/')) {
      // The Audio workspace asks the native origin for voices; without this the
      // request falls through Vite's dev proxy to a backend that Playwright
      // does not start, producing ECONNREFUSED noise in CI logs.
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(audioVoices()),
      })
    }
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
    let data = {}
    if (path === '/api/session' && remote)
      return route.fulfill({
        status: 401,
        contentType: 'application/json',
        body: JSON.stringify(invalidSession()),
      })
    if (path === '/api/access') data = accessState({ remote })
    else if (path === '/api/status') data = workspaceStatus()
    // Opt-in only: descriptor-driven engine options would hide the llama.cpp
    // fixture from unrelated ModelConfig journeys, so keep this stub partial.
    else if (audio && path === '/api/engines') data = engineDescriptors()
    else if (audio && path === AUDIO_CONTRACT_PATH) data = audioContract()
    else if (path === '/api/models') data = modelCatalog()
    else if (path === '/api/models/param-registry') data = paramRegistry()
    else if (path.endsWith('/reference-audio')) data = referenceAudio()
    else if (path === '/api/models/speech-model/config') data = audioModelConfig()
    else if (path === '/api/models/demo-model/config') data = llamaModelConfig()
    else if (path === '/api/proxy') data = proxySnapshot({ now })
    else if (path === '/api/proxy/requests') data = proxyRequests({ now })
    else if (path === '/api/inference/keys') {
      if (request.method() === 'POST') {
        keys.push(inferenceKeyFromRequest(request.postDataJSON()))
        data = createdInferenceKey(key)
      } else data = inferenceKeysResponse(keys)
    } else if (path === '/api/runtime/stale' || path === '/api/runtime/pending')
      data = runtimePlanState()
    else if (path === '/api/settings/inference') data = inferenceSettings()
    else if (path === '/api/config-history') data = configHistory()
    else if (path === '/api/models/safetensors') data = safetensorsModels()
    await route.fulfill({ contentType: 'application/json', body: JSON.stringify(data) })
  })
}
