import { expect, test } from '@playwright/test'

const INITIAL_CONFIG = {
  engine: 'llama_cpp',
  engines: { llama_cpp: { ctx_size: 4096 } },
}

const PARAM_REGISTRY = {
  sections: [{
    id: 'inference',
    label: 'Inference',
    params: [{
      key: 'ctx_size',
      label: 'Context Size',
      type: 'int',
      scalar_type: 'int',
      value_kind: 'scalar',
      default: 0,
      primary_flag: '--ctx-size',
      flags: ['--ctx-size'],
      supported: true,
    }],
  }],
}

function savedConfig(state) {
  return state.config || INITIAL_CONFIG
}

function catalog(state) {
  if (!state.installed) return []
  const phase = state.runPhase ?? (state.running ? 'running' : 'stopped')
  const run_state = phase === 'loading' || phase === 'running' ? phase : null
  const runtime_quality = state.runtimeQuality
    || (phase === 'unreachable' || phase === 'stale' ? phase : 'verified')
  const is_active = Object.hasOwn(state, 'isActive') ? state.isActive : run_state != null
  return [{
    base_model_name: 'Demo',
    huggingface_id: 'org/demo',
    quantizations: [{
      id: 'demo-model',
      name: 'Demo',
      format: 'safetensors',
      is_active,
      run_state,
      status: run_state,
      runtime_quality,
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
      action: state.running ? 'restart_now' : 'publish_next_start',
      running: Boolean(state.running),
      desired_revision: 'new',
      published_revision: 'old',
      requires_proxy_reload: false,
      consequence: state.running ? 'Restart this model' : 'Use on next start',
      differences: [{
        field: 'ctx_size',
        saved: String(state.config?.engines?.llama_cpp?.ctx_size ?? 4096),
        published: '4096',
        running: state.running ? '4096' : 'unknown',
      }],
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
      const next = state.sseEvents?.shift()
      const body = next
        ? `event: ${next.event}\ndata: ${JSON.stringify(next.data)}\n\n`
        : 'event: task_snapshot\ndata: {"tasks":[]}\n\n'
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body,
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
      if (state.failConfigGetsAfterSave && state.saveAttempts > 0) {
        await json(route, { detail: 'reload failed' }, 500)
        return
      }
      await json(route, savedConfig(state))
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
      if (state.failDownloadOnce && state.downloadAttempts === 1) {
        await json(route, { detail: 'mirror offline' }, 500)
        return
      }
      if (state.installOnDownload === false) {
        await json(route, { status: 'started', task_id: state.downloadTaskId || 'download-demo' })
        return
      }
      state.installed = true
      await json(route, { status: 'started', task_id: state.downloadTaskId || 'download-demo' })
      return
    }
    if (path === '/api/models/demo-model/config' && method === 'PUT') {
      state.saveAttempts = (state.saveAttempts || 0) + 1
      if (state.saveFailure === 'queue' && state.saveAttempts === 1) {
        await json(route, {
          code: 'STORE_QUEUE_FULL',
          committed: false,
          detail: 'Persistence is busy.',
        }, 503)
        return
      }
      if (state.saveFailure === 'replaced') {
        state.config = JSON.parse(request.postData() || '{}')
        await json(route, {
          code: 'STORE_WRITE_FAILED',
          committed: true,
          detail: 'The document was replaced, but acknowledgement failed.',
        }, 500)
        return
      }
      if (state.failSaveOnce && state.saveAttempts === 1) {
        await json(route, { detail: 'config store busy' }, 500)
        return
      }
      state.config = JSON.parse(request.postData() || '{}')
      state.reviewed = true
      state.pending = true
      await json(route, state.config)
      return
    }
    if (path === '/api/operations/recovery' && method === 'GET') {
      if (state.recoveryUnknown) {
        await json(route, {
          outcome: 'unknown',
          committed: 'unknown',
          detail: 'The recovery outcome could not be established. Refresh before trying again.',
        })
        return
      }
    }
    if (path === '/api/operations/reconcile' && method === 'POST') {
      state.reconcilePosts = (state.reconcilePosts || 0) + 1
      await json(route, {
        code: 'RECONCILE_UNKNOWN',
        outcome: 'unknown',
        committed: 'unknown',
        detail: 'The recovery outcome could not be established. Refresh before trying again.',
      }, 500)
      return
    }
    if (path === '/api/models/demo-model/runtime/apply' && method === 'POST') {
      state.applyAttempts = (state.applyAttempts || 0) + 1
      if (state.failApplyOnce && state.applyAttempts === 1) {
        await json(route, { detail: 'proxy rejected the plan' }, 500)
        return
      }
      state.pending = false
      await json(route, { status: 'succeeded' })
      return
    }
    if (path === '/api/models/demo-model/start' && method === 'POST') {
      if (state.startMode === 'hold') {
        await json(route, { status: 'accepted' })
        return
      }
      state.running = true
      state.runPhase = 'running'
      state.runtimeQuality = 'verified'
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
        proxy_status: {
          healthy: true,
          port: 2000,
          public_inference_url: '',
          health_observed_at: new Date().toISOString(),
        },
        runtime_observation: {
          quality: 'unreachable',
          observed_at: null,
          detail: 'No successful running-model observation yet.',
        },
        persistence: state.persistence || { saturated: false, latest_failure: null },
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
      await json(route, PARAM_REGISTRY)
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

function contextInput(page) {
  return page.locator('#basic-ctx_size input')
}

async function advanceClockUntil(page, predicate) {
  for (let step = 0; step < 40; step += 1) {
    if (await predicate()) return
    await page.clock.fastForward(1000)
  }
  expect(await predicate()).toBe(true)
}

async function saveDistinctiveContext(page, value = '12321') {
  await page.goto('/models/demo-model/config')
  const input = contextInput(page).first()
  await expect(input).toBeVisible()
  await input.fill(String(value))
  await input.press('Tab')
  const saved = page.waitForResponse((response) => (
    response.url().includes('/api/models/demo-model/config')
    && response.request().method() === 'PUT'
  ))
  await page.getByRole('button', { name: 'Save Configuration' }).click()
  expect((await saved).status()).toBe(200)
  await expect(page.getByText('These settings stay pending until you apply them.')).toBeVisible()
  await page.goto('/models')
  await page.goto('/models/demo-model/config')
  await expect(contextInput(page).first()).toHaveAttribute('aria-valuenow', String(value))
}

function installedState(extra = {}) {
  return {
    enginesMode: 'ready',
    enginesCalls: 0,
    installed: true,
    reviewed: true,
    running: false,
    pending: false,
    sseEvents: [],
    ...extra,
  }
}

function downloadTask(taskId, status, huggingfaceId = 'org/demo') {
  return {
    event: 'task_updated',
    data: {
      task_id: taskId,
      type: 'download',
      status,
      message: status === 'cancelled' ? 'stopped by user' : 'connection reset',
      metadata: { huggingface_id: huggingfaceId },
    },
  }
}

async function openSearch(page, state) {
  const errors = await openLibrary(page, state)
  await page.getByRole('button', { name: 'Search and download' }).click()
  await page.getByLabel('Search models').fill('demo')
  await page.getByRole('button', { name: 'Search', exact: true }).click()
  return errors
}

async function deliverDownloadEvent(page, state, task) {
  state.sseEvents.push(task)
  await page.clock.fastForward(3500)
}

test('install, configure, apply, start, and connect under fixture control', async ({ page }) => {
  const state = {
    enginesMode: 'ready',
    enginesCalls: 0,
    installed: false,
    reviewed: false,
    running: false,
    pending: false,
    failDownloadOnce: true,
    failSaveOnce: true,
    downloadAttempts: 0,
    saveAttempts: 0,
    connectAttempts: 0,
    sseEvents: [],
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
  await expect(page.getByText('Model started')).toBeVisible()
  await expectNoHorizontalOverflow(page)
  expect(errors).toEqual([])
})

test('keeps apply available after a failed publish and preserves the saved context size', async ({ page }) => {
  const state = installedState({ pending: true, failApplyOnce: true })
  const errors = await openLibrary(page, state)
  await saveDistinctiveContext(page, '12321')
  const apply = page.locator('#demo-model').getByRole('button', { name: 'Use on next start' })
  await apply.click()
  const dialog = page.getByRole('dialog', { name: 'Apply saved settings' })
  await dialog.getByRole('button', { name: 'Use on next start' }).click()
  await expect(page.getByText('Apply failed')).toBeVisible()
  await expect(dialog).toBeHidden()
  await expect(apply).toBeEnabled()
  await page.goto('/models')
  await page.goto('/models/demo-model/config')
  await expect(contextInput(page).first()).toHaveAttribute('aria-valuenow', '12321')
  await apply.click()
  await dialog.getByRole('button', { name: 'Use on next start' }).click()
  await expect(page.getByText('The model is stopped, so they are used the next time it starts.')).toBeVisible()
  expect(errors).toEqual([])
})

test('a stopped model says the next start uses the saved context size', async ({ page }) => {
  const errors = await openLibrary(page, installedState({ pending: true }))
  await saveDistinctiveContext(page, '12321')
  await page.getByRole('button', { name: 'Use on next start' }).click()
  const dialog = page.getByRole('dialog', { name: 'Apply saved settings' })
  await expect(dialog).toContainText('Use these saved settings the next time this model starts')
  await expect(dialog).toContainText('ctx_size')
  await expect(dialog).toContainText('saved 12321')
  await expect(dialog).toContainText('running unknown')
  expect(errors).toEqual([])
})

test('a running model says a restart is required for the saved context size', async ({ page }) => {
  const errors = await openLibrary(page, installedState({ pending: true, running: true, runPhase: 'running' }))
  await saveDistinctiveContext(page, '12321')
  await page.getByRole('button', { name: 'Restart this model' }).click()
  const dialog = page.getByRole('dialog', { name: 'Apply saved settings' })
  await expect(dialog).toContainText('Restart this model')
  await expect(dialog).toContainText('ctx_size')
  await expect(dialog).toContainText('saved 12321')
  await expect(dialog).toContainText('running 4096')
  expect(errors).toEqual([])
})

test('shows that a loading model has not started', async ({ page }) => {
  await page.clock.install({ time: new Date('2026-10-05T12:00:00Z') })
  const state = installedState({ startMode: 'hold' })
  const errors = await openLibrary(page, state)
  const started = page.waitForResponse((response) => (
    response.url().includes('/api/models/demo-model/start') && response.request().method() === 'POST'
  ))
  await page.getByRole('button', { name: 'Start Demo' }).click()
  await started
  state.runPhase = 'loading'
  state.isActive = true
  state.runtimeQuality = 'verified'
  await advanceClockUntil(page, () => page.getByText('Model is starting').isVisible())
  await expect(page.getByText('Model started')).toHaveCount(0)
  await expect(page.locator('button[aria-busy="true"]').first()).toBeVisible()
  expect(errors).toEqual([])
})

test('confirms start only for a verified running model', async ({ page }) => {
  const errors = await openLibrary(page, installedState())
  await page.getByRole('button', { name: 'Start Demo' }).click()
  await expect(page.getByText('Model started')).toBeVisible()
  await expect(page.getByText('Startup not confirmed')).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Connect Demo' })).toBeVisible()
  expect(errors).toEqual([])
})

test('does not treat a stale or unreachable observation as started', async ({ page }) => {
  await page.clock.install({ time: new Date('2026-10-05T12:00:00Z') })
  const state = installedState({ startMode: 'hold' })
  const errors = await openLibrary(page, state)
  const started = page.waitForResponse((response) => (
    response.url().includes('/api/models/demo-model/start') && response.request().method() === 'POST'
  ))
  await page.getByRole('button', { name: 'Start Demo' }).click()
  await started
  state.runPhase = 'running'
  state.runtimeQuality = 'stale'
  state.isActive = true
  await advanceClockUntil(page, () => page.getByText('Startup not confirmed').isVisible())
  await expect(page.getByText('Model started')).toHaveCount(0)
  expect(errors).toEqual([])
})

test('does not treat an unreachable observation as started', async ({ page }) => {
  await page.clock.install({ time: new Date('2026-10-05T12:00:00Z') })
  const state = installedState({ startMode: 'hold' })
  const errors = await openLibrary(page, state)
  const started = page.waitForResponse((response) => (
    response.url().includes('/api/models/demo-model/start') && response.request().method() === 'POST'
  ))
  await page.getByRole('button', { name: 'Start Demo' }).click()
  await started
  state.runPhase = 'stopped'
  state.runtimeQuality = 'unreachable'
  state.isActive = true
  await advanceClockUntil(page, () => page.getByText('Startup not confirmed').isVisible())
  await expect(page.getByText('Model started')).toHaveCount(0)
  expect(errors).toEqual([])
})

test('reports startup as unconfirmed when no ready observation arrives', async ({ page }) => {
  await page.clock.install({ time: new Date('2026-10-05T12:00:00Z') })
  const state = installedState({ startMode: 'hold' })
  const errors = await openLibrary(page, state)
  await saveDistinctiveContext(page, '22222')
  await page.goto('/models')
  const started = page.waitForResponse((response) => (
    response.url().includes('/api/models/demo-model/start') && response.request().method() === 'POST'
  ))
  await page.getByRole('button', { name: 'Start Demo' }).click()
  await started
  await advanceClockUntil(page, () => page.getByText('Startup not confirmed').isVisible())
  await expect(page.getByText('Model started')).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Start Demo' })).toBeEnabled()
  await page.goto('/models/demo-model/config')
  await expect(contextInput(page).first()).toHaveAttribute('aria-valuenow', '22222')
  expect(errors).toEqual([])
})

test('retries a proxy outage without dropping the saved context size', async ({ page }) => {
  const state = installedState({ runtimeQuality: 'unreachable' })
  const errors = await openLibrary(page, state)
  await expect(page.getByText('The inference proxy could not be reached. Running models are not shown as stopped.')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Retry' })).toBeVisible()
  await page.goto('/models/demo-model/config')
  await expect(page.getByText('Saved settings are on this page, but the proxy could not be reached.')).toBeVisible()
  await saveDistinctiveContext(page, '33333')
  await page.goto('/models')
  const outage = page.getByText('The inference proxy could not be reached. Running models are not shown as stopped.')
  await expect(outage).toBeVisible()
  state.runtimeQuality = 'verified'
  state.runPhase = 'stopped'
  await page.getByRole('button', { name: 'Retry' }).click()
  await expect(outage).toHaveCount(0)
  await expect(page.getByText('Unreachable')).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Start Demo' })).toBeEnabled()
  expect(errors).toEqual([])
})

test('keeps an in-progress download when a different task fails, then installs after cancellation', async ({ page }) => {
  await page.clock.install({ time: new Date('2026-10-05T12:00:00Z') })
  const state = installedState({
    installed: false,
    reviewed: false,
    installOnDownload: false,
    downloadTaskId: 'download-demo',
  })
  const errors = await openSearch(page, state)
  const download = page.getByRole('button', { name: 'Download' })
  await download.click()
  await expect(download).toBeDisabled()
  await deliverDownloadEvent(page, state, downloadTask('other-task', 'failed'))
  await expect(download).toBeDisabled()
  await expect(page.getByText('Download cancelled')).toHaveCount(0)
  await deliverDownloadEvent(page, state, downloadTask('download-demo', 'cancelled'))
  await expect(page.getByText('Download cancelled')).toBeVisible()
  await expect(download).toBeEnabled()
  state.installOnDownload = true
  await download.click()
  await expect(page.getByRole('button', { name: 'Configure' })).toBeVisible()
  await saveDistinctiveContext(page, '44444')
  expect(errors).toEqual([])
})

test('describes a failed download and installs on the next click', async ({ page }) => {
  await page.clock.install({ time: new Date('2026-10-05T12:00:00Z') })
  const state = installedState({
    installed: false,
    reviewed: false,
    installOnDownload: false,
    downloadTaskId: 'download-demo',
  })
  const errors = await openSearch(page, state)
  const download = page.getByRole('button', { name: 'Download' })
  await download.click()
  await expect(download).toBeDisabled()
  await deliverDownloadEvent(page, state, downloadTask('other-task', 'cancelled'))
  await expect(download).toBeDisabled()
  await deliverDownloadEvent(page, state, downloadTask('download-demo', 'failed'))
  await expect(page.getByText('Download failed')).toBeVisible()
  await expect(page.getByText('Download cancelled')).toHaveCount(0)
  await expect(download).toBeEnabled()
  state.installOnDownload = true
  await download.click()
  await expect(page.getByRole('button', { name: 'Configure' })).toBeVisible()
  expect(errors).toEqual([])
})

test('retries a rejected configuration save without dropping the edit', async ({ page }) => {
  const state = installedState({ saveFailure: 'queue' })
  await openLibrary(page, state)
  await page.goto('/models/demo-model/config')
  const input = contextInput(page).first()
  await expect(input).toBeVisible()
  await input.fill('12321')
  await input.press('Tab')
  const rejected = page.waitForResponse((response) => (
    response.url().includes('/api/models/demo-model/config')
    && response.request().method() === 'PUT'
  ))
  await page.getByRole('button', { name: 'Save Configuration' }).click()
  expect((await rejected).status()).toBe(503)
  await expect(input).toHaveAttribute('aria-valuenow', '12321')
  await expect(page.locator('.persistence-notice').getByText('Save paused')).toBeVisible()
  const accepted = page.waitForResponse((response) => (
    response.url().includes('/api/models/demo-model/config')
    && response.request().method() === 'PUT'
  ))
  await page.getByRole('button', { name: 'Try again' }).click()
  expect((await accepted).status()).toBe(200)
  expect(state.saveAttempts).toBe(2)
})

test('keeps a replaced configuration when acknowledgement fails and does not save again', async ({ page }) => {
  const state = installedState({ saveFailure: 'replaced' })
  await openLibrary(page, state)
  await page.goto('/models/demo-model/config')
  const input = contextInput(page).first()
  await expect(input).toBeVisible()
  await input.fill('12321')
  await input.press('Tab')
  const replaced = page.waitForResponse((response) => (
    response.url().includes('/api/models/demo-model/config')
    && response.request().method() === 'PUT'
  ))
  await page.getByRole('button', { name: 'Save Configuration' }).click()
  expect((await replaced).status()).toBe(500)
  const notice = page.locator('.persistence-notice')
  await expect(notice.getByText('Saved, acknowledgement failed')).toBeVisible()
  await expect(notice.getByText('was replaced')).toBeVisible()
  await expect(input).toHaveAttribute('aria-valuenow', '12321')
  await expect(page.getByRole('button', { name: 'Try again' })).toHaveCount(0)
  expect(state.saveAttempts).toBe(1)
})

test('keeps the edit when a replaced configuration cannot be refreshed', async ({ page }) => {
  const state = installedState({
    saveFailure: 'replaced',
    failConfigGetsAfterSave: true,
  })
  await openLibrary(page, state)
  await page.goto('/models/demo-model/config')
  const input = contextInput(page).first()
  await expect(input).toBeVisible()
  await input.fill('12321')
  await input.press('Tab')
  await page.getByRole('button', { name: 'Save Configuration' }).click()
  await expect(page.getByText('could not be reloaded')).toBeVisible()
  await expect(input).toHaveAttribute('aria-valuenow', '12321')
  await expect(page.getByRole('button', { name: 'Save Configuration' })).toBeDisabled()
  await page.locator('.persistence-notice').getByRole('button', { name: 'Refresh' }).click()
  await expect(page.getByText('could not be reloaded')).toBeVisible()
  expect(state.saveAttempts).toBe(1)
})

test('maps a persistence failure code and hides exported exception text', async ({ page }) => {
  const errors = await openLibrary(page, {
    installed: false,
    enginesCalls: 0,
    persistence: {
      saturated: true,
      pending_store_writes: 32,
      max_pending_store_writes: 32,
      latest_failure: {
        code: 'STORE_WRITE_FAILED',
        committed: false,
        message: 'disk full hf_BROWSERSECRET',
        description: 'hf_BROWSERSECRET',
      },
    },
  })
  const footer = page.locator('.footer-diagnostics')
  await expect(footer.getByText('Queue full (32/32)')).toBeVisible()
  await expect(footer.getByText('The save was not stored. The previous state is unchanged.')).toBeVisible()
  await expect(footer.getByText('No running-model observation')).toBeVisible()
  await expect(page.getByText('hf_BROWSERSECRET')).toHaveCount(0)
  expect(errors).toEqual([])
})

test('keeps an unknown recovery when reconcile cannot be established', async ({ page }) => {
  const state = installedState({ recoveryUnknown: true })
  await openLibrary(page, state)
  await expect(page.getByText('could not be established')).toBeVisible()
  await page.locator('.activity-recovery').getByRole('button', { name: 'Refresh' }).click()
  await expect(page.getByText('could not be established')).toBeVisible()
  expect(state.reconcilePosts).toBe(1)
})
