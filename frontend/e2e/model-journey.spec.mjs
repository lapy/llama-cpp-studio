import { expect, test } from '@playwright/test'

const SAVED_CONFIG = { engine: 'llama_cpp', engines: { llama_cpp: {} } }

function catalog(state) {
  if (!state.installed) return []
  return [{
    base_model_name: 'Demo',
    huggingface_id: 'org/demo',
    quantizations: [{
      id: 'demo-model',
      name: 'Demo',
      format: 'safetensors',
      is_active: state.running,
      runtime_quality: 'verified',
      config_reviewed: state.reviewed,
      config: { engine: 'llama_cpp' },
      llama_swap_id: 'demo-model',
      proxy_name: 'demo-model',
    }],
  }]
}

function pendingPlan(state) {
  if (!state.pending) {
    return {
      applicable: true,
      pending: false,
      launch_manifests: true,
      requires_proxy_reload: false,
      changes: [],
      models: [],
    }
  }
  return {
    applicable: true,
    pending: true,
    launch_manifests: true,
    requires_proxy_reload: false,
    plan_id: 'plan-1',
    changes: ['Saved settings changed'],
    models: [{
      catalog_id: 'demo-model',
      model_id: 'demo-model',
      action: 'publish_next_start',
      running: false,
      desired_revision: 'new',
      published_revision: 'old',
      requires_proxy_reload: false,
    }],
  }
}

function isStudioApi(url) {
  const path = url.pathname
  return path === '/api' || path.startsWith('/api/')
}

async function installApi(page, state) {
  await page.route(isStudioApi, async (route) => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    const method = request.method()

    if (path === '/api/events') {
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: 'event: task_snapshot\ndata: {"tasks":[]}\n\n',
      })
      return
    }

    if (path === '/api/engines' && method === 'GET') {
      state.enginesCalls += 1
      if (state.enginesMode === 'fail' || (state.enginesMode === 'fail-once' && state.enginesCalls === 1)) {
        await route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ detail: 'descriptors unavailable' }),
        })
        return
      }
      if (state.enginesGate && state.enginesCalls === 1) {
        state.enginesHeld()
        await state.enginesGate
      }
      await json(route, {
        engines: [{ id: 'llama_cpp', label: 'llama.cpp', runnable: true, enabled: true }],
      })
      return
    }

    if (path === '/api/models' && method === 'GET') {
      await json(route, catalog(state))
      return
    }
    if (path === '/api/models/demo-model/config' && method === 'GET') {
      await json(route, SAVED_CONFIG)
      return
    }
    if (path === '/api/models/safetensors' && method === 'GET') {
      await json(route, state.installed
        ? [{ id: 'demo-model', model_id: 'demo-model', huggingface_id: 'org/demo' }]
        : [])
      return
    }
    if (path === '/api/model-catalog/search' && method === 'POST') {
      await json(route, {
        items: [{
          id: 'org/demo',
          display_name: 'Demo',
          provider: 'huggingface',
          provider_item_id: 'org/demo',
          source: { id: 'org/demo' },
          artifact_format: 'safetensors',
          install_variants: [{
            id: 'default',
            label: 'Default',
            format: 'safetensors',
            installable: true,
            size_bytes: 1024,
            files: [{ filename: 'model.safetensors', size: 1024 }],
          }],
        }],
        total: 1,
        page: 1,
        has_more: false,
        facets: {},
        provider_status: {},
      })
      return
    }
    if (path === '/api/models/safetensors/download-bundle' && method === 'POST') {
      state.downloadAttempts = (state.downloadAttempts || 0) + 1
      if (state.downloadAttempts === 1) {
        await json(route, { detail: 'mirror offline' }, 500)
        return
      }
      state.installed = true
      await json(route, { status: 'started' })
      return
    }
    if (path === '/api/models/demo-model/config' && method === 'PUT') {
      state.saveAttempts = (state.saveAttempts || 0) + 1
      if (state.saveAttempts === 1) {
        await json(route, { detail: 'config store busy' }, 500)
        return
      }
      state.reviewed = true
      state.pending = true
      await json(route, SAVED_CONFIG)
      return
    }
    if (path === '/api/models/demo-model/runtime/apply' && method === 'POST') {
      state.pending = false
      await json(route, { status: 'succeeded' })
      return
    }
    if (path === '/api/models/demo-model/start' && method === 'POST') {
      state.running = true
      await json(route, { status: 'running' })
      return
    }
    if (path === '/api/models/demo-model/connect-test' && method === 'POST') {
      state.connectAttempts = (state.connectAttempts || 0) + 1
      if (state.connectAttempts === 1) {
        await json(route, { detail: 'endpoint offline' }, 500)
        return
      }
      await json(route, { status_code: 200, body: 'pong' })
      return
    }
    if (path === '/api/llama-swap/pending') {
      await json(route, pendingPlan(state))
      return
    }
    if (path === '/api/llama-swap/stale') {
      await json(route, { applicable: true, stale: state.pending })
      return
    }
    if (path === '/api/status') {
      await json(route, {
        proxy_status: { healthy: true, port: 2000, public_inference_url: '' },
      })
      return
    }
    if (path === '/api/access') {
      await json(route, { mode: 'local', authenticated: true })
      return
    }
    if (path === '/api/models/huggingface-token') {
      await json(route, { has_token: true, token_preview: '', from_environment: false })
      return
    }
    if (path === '/api/gpu-info' || path === '/api/gpu-list') {
      await json(route, { gpus: [], cpu_threads: 4, device_count: 0 })
      return
    }
    if (path === '/api/models/param-registry') {
      await json(route, { sections: [] })
      return
    }
    if (path === '/api/settings/inference' && method === 'PUT') {
      await json(route, { public_inference_url: '' })
      return
    }

    await json(route, method === 'GET' ? {} : { ok: true })
  })
}

function json(route, body, status = 200) {
  return route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}

async function openLibrary(page, state) {
  const errors = []
  page.on('pageerror', (error) => errors.push(String(error)))
  await page.addInitScript(() => {
    localStorage.removeItem('llama-studio.setup-checklist.dismissed')
  })
  await installApi(page, state)
  await page.goto('/models')
  return errors
}

async function expectNoHorizontalOverflow(page) {
  const overflow = await page.evaluate(() => (
    document.documentElement.scrollWidth - document.documentElement.clientWidth
  ))
  expect(overflow).toBeLessThanOrEqual(1)
}

test('withholds the checklist while descriptors are delayed, then advances', async ({ page }) => {
  let releaseEngines = () => {}
  let enginesHeld = () => {}
  const enginesGate = new Promise((resolve) => {
    releaseEngines = resolve
  })
  const held = new Promise((resolve) => {
    enginesHeld = resolve
  })
  const errors = await openLibrary(page, {
    enginesMode: 'delay',
    enginesCalls: 0,
    enginesGate,
    enginesHeld,
    installed: false,
  })
  await held
  await expect(page.getByRole('heading', { name: /Next:/ })).toHaveCount(0)
  releaseEngines()
  await expect(page.getByRole('heading', { name: 'Next: Download a model' })).toBeVisible()
  await expectNoHorizontalOverflow(page)
  expect(errors).toEqual([])
})

test('keeps engine preparation incomplete when descriptors fail, then recovers on reload', async ({ page }) => {
  const state = { enginesMode: 'fail-once', enginesCalls: 0, installed: false }
  const errors = await openLibrary(page, state)
  await expect(page.getByRole('heading', { name: 'Next: Prepare a runnable engine' })).toBeVisible()
  await page.reload()
  await expect(page.getByRole('heading', { name: 'Next: Download a model' })).toBeVisible()
  await expectNoHorizontalOverflow(page)
  expect(errors).toEqual([])
})

test('install, configure, apply, start, and connect under fixture control', async ({ page }) => {
  const state = {
    enginesMode: 'ready',
    enginesCalls: 0,
    installed: false,
    reviewed: false,
    running: false,
    pending: false,
    downloadAttempts: 0,
    saveAttempts: 0,
    connectAttempts: 0,
  }
  const errors = await openLibrary(page, state)
  await expect(page.getByRole('heading', { name: 'Next: Download a model' })).toBeVisible()
  await page.getByRole('button', { name: 'Search and download' }).click()
  await page.getByLabel('Search models').fill('demo')
  await page.getByRole('button', { name: 'Search', exact: true }).click()
  const download = page.getByRole('button', { name: 'Download' })
  await download.click()
  await expect(page.getByText('Download failed')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Configure' })).toHaveCount(0)
  await download.click()
  await expect(page.getByRole('button', { name: 'Configure' })).toBeVisible()

  await page.goto('/models')
  await expect(page.getByRole('heading', { name: 'Next: Review configuration' })).toBeVisible()
  await page.getByRole('button', { name: 'Configure Demo' }).click()
  const save = page.getByRole('button', { name: 'Save Configuration' })
  await expect(save).toBeVisible()
  await save.click()
  await expect(page.getByText('Save failed')).toBeVisible()
  await save.click()
  await page.getByRole('button', { name: 'Use on next start' }).click()
  const applyDialog = page.getByRole('dialog', { name: 'Apply saved settings' })
  await applyDialog.getByRole('button', { name: 'Use on next start' }).click()
  await expect(page.getByText('The model is stopped, so they are used the next time it starts.')).toBeVisible()

  await page.goto('/models')
  await page.getByRole('button', { name: 'Start Demo' }).click()
  await page.getByRole('button', { name: 'Connect Demo' }).click()
  const connect = page.getByRole('dialog', { name: 'Connect' })
  const send = connect.getByRole('button', { name: 'Send test' })
  await send.click()
  await expect(connect.getByRole('status')).toHaveText('endpoint offline')
  await send.click()
  await expect(connect.getByRole('status')).toHaveText('pong')
  await expectNoHorizontalOverflow(page)
  expect(errors).toEqual([])
})
