import { expect, test } from '@playwright/test'

const NOTICE = 'Restoring configuration does not publish it or restart models.'

function backup(models = []) {
  return {
    schema_version: 1,
    kind: 'llama-cpp-studio-config-backup',
    application_version: '1.0.0',
    preferences: { public_inference_url: 'http://restored.example' },
    models,
    templates: [],
    routing: { profiles: {}, selectors: {} },
  }
}

async function installApi(page, state) {
  await page.route('**/*', async (route) => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    const method = request.method()
    if (!path.startsWith('/api')) {
      await route.continue()
      return
    }
    if (path === '/api/events') {
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: 'event: task_snapshot\ndata: {"tasks":[]}\n\n',
      })
      return
    }
    if (path === '/api/access') {
      await json(route, { mode: 'local', authenticated: true })
      return
    }
    if (path === '/api/status') {
      await json(route, {
        proxy_status: { healthy: true, health_observed_at: new Date().toISOString() },
        runtime_observation: { quality: 'unreachable', observed_at: null },
        persistence: { saturated: false, latest_failure: null },
      })
      return
    }
    if (path === '/api/gpu-info' || path === '/api/gpu-list') {
      await json(route, { gpus: [], cpu_threads: 4, device_count: 0 })
      return
    }
    if (path === '/api/llama-swap/stale' || path === '/api/llama-swap/pending') {
      await json(route, { applicable: false, pending: false, stale: false })
      return
    }
    if (path === '/api/models' && method === 'GET') {
      await json(route, [{
        base_model_name: 'Demo',
        quantizations: [{ id: 'org--model', name: 'Demo' }],
      }])
      return
    }
    if (path === '/api/config-backup' && method === 'GET') {
      state.downloads = (state.downloads || 0) + 1
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        headers: { 'content-disposition': 'attachment; filename="studio-config-backup.json"' },
        body: JSON.stringify(backup()),
      })
      return
    }
    if (path === '/api/config-backup/preview' && method === 'POST') {
      state.previews += 1
      const body = JSON.parse(request.postData() || '{}')
      if (state.rejectPreview) {
        await json(route, { code: 'BACKUP_MALFORMED', detail: 'This backup version is not supported.' }, 400)
        return
      }
      const mapped = body.mapping?.['huggingface:org/missing']
      const skipped = body.decisions?.models?.['huggingface:org/missing'] === 'skip'
      const unresolved = state.unresolved && !mapped && !skipped
      await json(route, {
        schema_version: 1,
        plan_id: unresolved ? null : `plan-${state.previews}`,
        applicable: !unresolved,
        notice: 'This restores saved settings only. It does not start, stop, or publish a model.',
        revisions: { 'settings.yaml': '1:2:3' },
        items: state.unresolved
          ? [{
            kind: 'model',
            id: 'huggingface:org/missing',
            action: unresolved ? 'unresolved' : (skipped ? 'skip' : 'keep'),
            reason: unresolved ? 'no local model' : undefined,
            local_id: mapped || undefined,
          }]
          : [{ kind: 'preference', id: 'public_inference_url', action: 'replace' }],
        limits: { includes: ['portable preferences'], excludes: ['credentials'] },
      })
      return
    }
    if (path === '/api/config-backup/apply' && method === 'POST') {
      state.applies += 1
      state.lastApply = JSON.parse(request.postData() || '{}')
      if (state.applyMode === 'stale') {
        await json(route, {
          code: 'BACKUP_STALE',
          detail: 'The configuration changed after the preview. Request a new preview.',
        }, 409)
        return
      }
      if (state.applyMode === 'uncertain') {
        await json(route, {
          code: 'BACKUP_INCOMPLETE',
          detail: 'The restore outcome could not be established. Restart before trying again.',
        }, 500)
        return
      }
      await json(route, {
        plan_id: 'plan-done',
        outcome: 'completed',
        notice: 'This restores saved settings only. It does not start, stop, or publish a model.',
      })
      return
    }
    if (path === '/api/config-backup/reconcile' && method === 'POST') {
      state.reconciles += 1
      await json(route, { outcome: 'pre_import' })
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

async function openRestore(page, state) {
  await page.addInitScript(() => {
    localStorage.removeItem('llama-studio.setup-checklist.dismissed')
  })
  await installApi(page, state)
  await page.goto('/restore')
  await expect(page).toHaveURL(/\/engines#config-backup$/)
  await expect(page.getByRole('heading', { name: 'Backup and restore' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Download backup' })).toBeVisible()
  await expect(page.getByText(NOTICE)).toBeVisible()
}

test('downloads the current configuration backup', async ({ page }) => {
  const state = { previews: 0, applies: 0, reconciles: 0, downloads: 0 }
  await openRestore(page, state)
  const download = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download backup' }).click()
  const file = await download
  expect(file.suggestedFilename()).toBe('studio-config-backup.json')
  expect(state.downloads).toBe(1)
  expect(state.applies).toBe(0)
})

test('rejects a backup that cannot be read', async ({ page }) => {
  const state = { previews: 0, applies: 0, reconciles: 0 }
  await openRestore(page, state)
  await page.locator('input[type="file"]').setInputFiles({
    name: 'backup.json',
    mimeType: 'application/json',
    buffer: Buffer.from('{'),
  })
  await expect(page.getByText('This backup could not be read.')).toBeVisible()
  expect(state.previews).toBe(0)
  expect(state.applies).toBe(0)
})

test('requires a new preview when the plan is stale', async ({ page }) => {
  const state = { previews: 0, applies: 0, reconciles: 0, applyMode: 'stale' }
  await openRestore(page, state)
  await page.locator('input[type="file"]').setInputFiles({
    name: 'backup.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(backup())),
  })
  await expect(page.locator('.restore-item-action', { hasText: 'Replacement' })).toBeVisible()
  await page.getByRole('button', { name: 'Restore saved settings' }).click()
  await expect(page.getByText('Update the preview before restoring.')).toBeVisible()
  expect(state.applies).toBe(1)
  await expect(page.getByRole('button', { name: 'Restore saved settings' })).toBeDisabled()
})

test('maps an unresolved model and restores once', async ({ page }) => {
  const state = { previews: 0, applies: 0, reconciles: 0, unresolved: true }
  await openRestore(page, state)
  await page.locator('input[type="file"]').setInputFiles({
    name: 'backup.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(backup([{ ref: 'huggingface:org/missing', config: {} }]))),
  })
  await expect(page.locator('.restore-item-action', { hasText: 'Unresolved model' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Restore saved settings' })).toBeDisabled()
  await page.getByLabel('Map huggingface:org/missing').selectOption('org--model')
  await page.getByRole('button', { name: 'Update preview' }).click()
  await expect(page.locator('.restore-item-action', { hasText: 'Keep existing' })).toBeVisible()
  await page.getByRole('button', { name: 'Restore saved settings' }).click()
  await expect(page.getByText('Restore completed.')).toBeVisible()
  await expect(page.getByText('were not restarted or published')).toBeVisible()
  expect(state.applies).toBe(1)
  expect(state.lastApply.mapping['huggingface:org/missing']).toBe('org--model')
  expect(state.lastApply.plan_id).toBe('plan-2')
})

test('completes a restore without publishing or restarting', async ({ page }) => {
  const state = { previews: 0, applies: 0, reconciles: 0 }
  await openRestore(page, state)
  await page.locator('input[type="file"]').setInputFiles({
    name: 'backup.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(backup())),
  })
  await page.getByRole('button', { name: 'Restore saved settings' }).click()
  await expect(page.getByText('Restore completed. Saved settings were updated.')).toBeVisible()
  await expect(page.getByText('Running models were not restarted or published.')).toBeVisible()
  expect(state.applies).toBe(1)
})

test('reconciles an uncertain restore without sending it again', async ({ page }) => {
  const state = { previews: 0, applies: 0, reconciles: 0, applyMode: 'uncertain' }
  await openRestore(page, state)
  await page.locator('input[type="file"]').setInputFiles({
    name: 'backup.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(backup())),
  })
  await page.getByRole('button', { name: 'Restore saved settings' }).click()
  await expect(page.getByText('could not be established')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Restore saved settings' })).toHaveCount(0)
  await page.getByRole('button', { name: 'Reconcile' }).click()
  await expect(page.getByText('pre-import state')).toBeVisible()
  expect(state.applies).toBe(1)
  expect(state.reconciles).toBe(1)
})
