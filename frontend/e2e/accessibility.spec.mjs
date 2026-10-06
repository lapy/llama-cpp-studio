import { expect, test } from '@playwright/test'

const NOTICE = 'Restoring configuration does not publish it or restart models.'

function backup() {
  return {
    schema_version: 1,
    kind: 'llama-cpp-studio-config-backup',
    application_version: '1.0.0',
    preferences: { public_inference_url: 'http://restored.example' },
    models: [],
    templates: [],
    routing: { profiles: {}, selectors: {} },
  }
}

async function json(route, body, status = 200) {
  await route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
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
    if (path === '/api/models' && method === 'GET') {
      await json(route, [{
        base_model_name: 'Demo',
        huggingface_id: 'org/demo',
        quantizations: [{
          id: 'demo-model',
          name: 'Demo',
          format: 'gguf',
          quantization: 'Q4_K_M',
          is_active: true,
          run_state: 'running',
          status: 'running',
          runtime_quality: 'verified',
          config: { engine: 'llama_cpp' },
          llama_swap_id: 'demo-model',
          proxy_name: 'demo-model',
        }],
      }])
      return
    }
    if (path === '/api/models/demo-model/config' && method === 'GET') {
      await json(route, { engine: 'llama_cpp', engines: { llama_cpp: { ctx_size: 4096 } } })
      return
    }
    if (path === '/api/models/param-registry') {
      await json(route, { sections: [] })
      return
    }
    if (path === '/api/llama-swap/stale') {
      await json(route, { applicable: true, stale: true, pending: false })
      return
    }
    if (path === '/api/config-backup/preview' && method === 'POST') {
      await json(route, {
        schema_version: 1,
        plan_id: 'plan-a11y',
        applicable: true,
        notice: 'This restores saved settings only. It does not start, stop, or publish a model.',
        revisions: { 'settings.yaml': '1:2:3' },
        items: [{ kind: 'preference', id: 'public_inference_url', action: 'replace' }],
        limits: { includes: ['portable preferences'], excludes: ['credentials'] },
      })
      return
    }
    if (path === '/api/config-backup/apply' && method === 'POST') {
      state.applies += 1
      await json(route, {
        plan_id: 'plan-a11y',
        outcome: 'completed',
        notice: 'This restores saved settings only. It does not start, stop, or publish a model.',
      })
      return
    }
    await json(route, method === 'GET' ? {} : { ok: true })
  })
}

async function horizontalOverflow(page) {
  return page.evaluate(() => {
    const main = document.getElementById('main-content')
    return main.scrollWidth - main.clientWidth
  })
}

test('keyboard restore can preview, confirm, and cancel', async ({ page }) => {
  const state = { applies: 0 }
  await installApi(page, state)
  await page.goto('/restore')
  await expect(page.getByRole('heading', { name: 'Backup and restore' })).toBeVisible()
  await expect(page.getByText(NOTICE)).toBeVisible()
  const file = page.locator('#restore-backup-file')
  await file.setInputFiles({
    name: 'studio-config-backup.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(backup())),
  })
  const restore = page.getByRole('button', { name: 'Restore saved settings' })
  await expect(restore).toBeEnabled()
  await file.focus()
  for (let step = 0; step < 12 && await restore.evaluate((node) => document.activeElement !== node); step += 1) {
    await page.keyboard.press('Tab')
  }
  await expect(restore).toBeFocused()
  await page.keyboard.press('Enter')
  await expect(page.locator('.restore-status')).toContainText('Restore completed')
  expect(state.applies).toBe(1)

  await file.setInputFiles({
    name: 'studio-config-backup.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(backup())),
  })
  const cancel = page.getByRole('button', { name: 'Cancel restore' })
  await expect(cancel).toBeVisible()
  await cancel.focus()
  await page.keyboard.press('Enter')
  await expect(restore).toHaveCount(0)
  await expect(file).toBeFocused()
  expect(await horizontalOverflow(page)).toBeLessThanOrEqual(1)
})

test('connect and apply dialogs return focus and announce a bad URL once', async ({ page }) => {
  await installApi(page, { applies: 0 })
  await page.goto('/models')
  await page.getByRole('button', { name: 'List' }).click()
  const connect = page.getByRole('button', { name: 'Connect Demo' })
  await expect(connect).toBeVisible()
  await connect.focus()
  await page.keyboard.press('Enter')
  const dialog = page.getByRole('dialog', { name: 'Connect' })
  await expect(dialog).toBeVisible()
  await expect(dialog.getByRole('button', { name: 'Copy model ID' })).toBeVisible()
  await expect(dialog.getByRole('button', { name: 'Copy endpoint' })).toBeVisible()
  const url = dialog.getByLabel('Public URL')
  await url.fill('ftp://files.example/model')
  await url.press('Tab')
  const alert = dialog.getByRole('alert')
  await expect(alert).toHaveCount(1)
  await expect(alert).toHaveText('Use an http or https URL without a username or password.')
  await page.keyboard.press('Escape')
  await expect(dialog).toBeHidden()
  await expect(connect).toBeFocused()

  await page.goto('/models/demo-model/config')
  const apply = page.getByRole('button', { name: 'Reload proxy' })
  await expect(apply).toBeVisible()
  await apply.focus()
  await page.keyboard.press('Enter')
  const impact = page.getByRole('dialog', { name: 'Apply saved settings' })
  await expect(impact).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(impact).toBeHidden()
  await expect(apply).toBeFocused()
})

test('diagnostics, restore, and connect stay reachable at 200% zoom', async ({ page }) => {
  await installApi(page, { applies: 0 })
  await page.goto('/models')
  await page.getByRole('button', { name: 'List' }).click()
  await page.evaluate(() => { document.documentElement.style.zoom = '2' })
  const connect = page.getByRole('button', { name: 'Connect Demo' })
  await connect.scrollIntoViewIfNeeded()
  await expect(connect).toBeVisible()
  const diagnostics = page.getByRole('link', { name: 'Download diagnostics' })
  await diagnostics.scrollIntoViewIfNeeded()
  await expect(diagnostics).toBeVisible()
  await page.goto('/restore')
  await page.evaluate(() => { document.documentElement.style.zoom = '2' })
  const download = page.getByRole('button', { name: 'Download backup' })
  await download.scrollIntoViewIfNeeded()
  await expect(download).toBeVisible()
  const file = page.locator('#restore-backup-file')
  await file.scrollIntoViewIfNeeded()
  await expect(file).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Backup and restore' })).toBeVisible()
})
