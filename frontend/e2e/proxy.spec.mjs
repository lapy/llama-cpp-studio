import { expect, test } from '@playwright/test'

for (const width of [1440, 390]) {
  test(`proxy tabs keep the footer visible at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 })
    await page.route('**/api/**', async (route) => {
      const path = new URL(route.request().url()).pathname
      if (!path.startsWith('/api/')) return route.continue()
      const data =
        path === '/api/proxy' || path === '/api/proxy/requests'
          ? {
              total: 100,
              settings: { backend: 'studio', proxy_port: 2000 },
              configured_backend: 'studio',
              active_backend: 'studio',
              running: [],
              catalog: [],
              activity: [{ type: 'ready', model: 'demo', created_at: 1, timestamp: 1000 }],
              requests: Array.from({ length: 100 }, (_, index) => ({
                id: String(index),
                model: 'demo',
                completed_at: 1000 + index,
                method: 'POST',
                path: '/v1/chat/completions',
                status: 200,
                latency_ms: 10,
                output_tokens: 20,
              })),
            }
          : path === '/api/inference/keys'
            ? { keys: [] }
            : {}
      await route.fulfill({ contentType: 'application/json', body: JSON.stringify(data) })
    })
    await page.goto('/proxy')
    await expect(page.getByRole('tab', { name: 'Requests', exact: true })).toBeVisible()
    await page.getByRole('tab', { name: 'Requests', exact: true }).click()
    await expect(page.locator('tbody tr')).toHaveCount(100)
    await page.getByRole('button', { name: 'Details', exact: true }).first().click()
    await expect(page.getByRole('dialog')).toContainText('/v1/chat/completions')
    await expect(
      page.getByRole('dialog').getByText('Time to first byte', { exact: true }),
    ).toHaveCount(0)
    await page.getByRole('dialog').getByLabel('Show unavailable fields').check()
    await expect(
      page.getByRole('dialog').getByText('Time to first byte', { exact: true }),
    ).toBeVisible()
    await page.keyboard.press('Escape')
    for (const name of ['Events', 'Settings & routing', 'Overview']) {
      await page.getByRole('tab', { name, exact: true }).click()
      const footer = await page.locator('footer').boundingBox()
      expect(footer.y + footer.height).toBeLessThanOrEqual(901)
      expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
        width,
      )
    }
  })
}

for (const width of [1440, 390]) {
  test(`key usage and policy editing at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 })
    let savedPolicy = null
    const key = {
      id: 'production',
      name: 'Production',
      prefix: 'sk-studio-preview',
      scopes: ['inference', 'models:read'],
      models: [],
      daily_token_budget: 1000,
      monthly_token_budget: 10000,
      max_output_tokens: 64,
      usage: {
        daily_tokens: 250,
        monthly_tokens: 1500,
        daily_remaining: 750,
        monthly_remaining: 8500,
        day_resets_at: 1900000000,
        month_resets_at: 1901000000,
        requests_per_minute: 2,
        input_tokens_per_minute: 50,
        output_tokens_per_minute: 20,
        concurrent_requests: 1,
        estimated_requests: 1,
      },
    }
    await page.route('**/api/**', async (route) => {
      const path = new URL(route.request().url()).pathname
      if (!path.startsWith('/api/')) return route.continue()
      if (path === '/api/inference/keys/production' && route.request().method() === 'PATCH') {
        savedPolicy = route.request().postDataJSON()
        Object.assign(key, savedPolicy)
      }
      await route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify(
          path === '/api/proxy'
            ? {
                settings: { backend: 'studio' },
                configured_backend: 'studio',
                active_backend: 'studio',
                running: [],
                catalog: [],
              }
            : path === '/api/inference/keys'
              ? { keys: [key] }
              : {},
        ),
      })
    })
    await page.goto('/proxy')
    await page.getByRole('tab', { name: 'Access keys', exact: true }).click()
    await expect(page.locator('.key-usage')).toContainText('750 remaining')
    await expect(page.locator('.key-usage')).toContainText('1 requests charged by estimate')
    await page.getByRole('button', { name: 'Edit policy', exact: true }).click()
    const dialog = page.getByRole('dialog')
    await dialog.getByLabel('Token accounting', { exact: true }).selectOption('exact')
    await dialog.getByLabel('Daily token budget (UTC)', { exact: true }).fill('2000')
    await dialog.getByLabel('Maximum output tokens per request', { exact: true }).fill('128')
    await dialog.getByRole('button', { name: 'Save policy', exact: true }).click()
    await expect(dialog).not.toBeVisible()
    expect(savedPolicy.token_accounting).toBe('exact')
    expect(savedPolicy.daily_token_budget).toBe(2000)
    expect(savedPolicy.max_output_tokens).toBe(128)
    expect(savedPolicy.scopes).toEqual(['inference', 'models:read'])
    const footer = await page.locator('footer').boundingBox()
    expect(footer.y + footer.height).toBeLessThanOrEqual(901)
  })
}

for (const width of [1440, 390]) {
  test(`diagnostics use available width and System tools stay out of the footer at ${width}px`, async ({
    page,
  }) => {
    await page.setViewportSize({ width, height: 900 })
    let requestQuery = ''
    await page.route('**/api/**', async (route) => {
      const url = new URL(route.request().url())
      if (!url.pathname.startsWith('/api/')) return route.continue()
      if (url.pathname === '/api/proxy/requests') requestQuery = url.search
      const data =
        url.pathname === '/api/proxy'
          ? {
              settings: { backend: 'studio' },
              active_backend: 'studio',
              configured_backend: 'studio',
              summary: { requests: 277, client_errors: 271, server_errors: 0 },
              running: [
                {
                  model: 'demo-model',
                  state: 'ready',
                  ready: true,
                  pid: 42,
                  port: 3010,
                  request_count: 277,
                  client_error_count: 271,
                  server_error_count: 0,
                },
              ],
              catalog: [],
            }
          : url.pathname === '/api/inference/keys'
            ? { keys: [] }
            : {}
      await route.fulfill({ contentType: 'application/json', body: JSON.stringify(data) })
    })
    await page.goto('/proxy')
    const runtime = page.locator('.runtime')
    await expect(runtime).toBeVisible()
    if (width === 390) {
      await expect(page.locator('.overview-analysis')).not.toHaveAttribute('open')
      await page.locator('.overview-analysis > summary').click()
      await expect(page.locator('.traffic-chart')).toBeVisible()
      await page.locator('.overview-analysis > summary').click()
    }
    const bounds = await runtime.boundingBox()
    const panel = await page.locator('.proxy-panels').boundingBox()
    expect(bounds.width / panel.width).toBeGreaterThan(0.85)
    await page.getByRole('button', { name: /Inspect failed requests for demo-model/ }).click()
    await expect(page.getByRole('tab', { name: 'Requests', exact: true })).toHaveAttribute(
      'aria-selected',
      'true',
    )
    await expect.poll(() => requestQuery).toContain('outcome=errors')
    expect(requestQuery).toContain('model=demo-model')
    await page.getByRole('link', { name: 'System', exact: true }).first().click()
    await expect(page.getByRole('heading', { name: 'Service health' })).toBeVisible()
    await page.getByRole('link', { name: 'Backup & restore', exact: true }).click()
    await expect(
      page.getByRole('heading', { name: 'Backup and restore', exact: true }),
    ).toBeVisible()
    await expect(page.locator('footer')).not.toContainText('Download backup')
    expect((await page.locator('footer').boundingBox()).height).toBeLessThan(65)
    await page.getByRole('link', { name: 'Change history', exact: true }).click()
    await expect(page.getByRole('heading', { name: 'Recent changes', exact: true })).toBeVisible()
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
      width,
    )
  })
}

for (const width of [1440, 390]) {
  test(`workspace navigator and focused key creation work at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 })
    const now = Date.now() / 1000
    const keys = [
      {
        id: 'production',
        name: 'Production',
        prefix: 'sk-studio-preview',
        scopes: ['inference'],
        models: [],
      },
    ]
    await page.route('**/api/**', async (route) => {
      const path = new URL(route.request().url()).pathname
      if (!path.startsWith('/api/')) return route.continue()
      let data = {}
      if (path === '/api/proxy')
        data = {
          settings: { backend: 'studio', allow_anonymous: false },
          configured_backend: 'studio',
          active_backend: 'studio',
          running: [],
          catalog: [],
          summary: { requests: 3, client_errors: 1, server_errors: 1, unknown_usage: 0 },
          requests: [200, 404, 503].map((status, index) => ({
            id: String(index),
            started_at: now - 10,
            status,
          })),
        }
      if (path === '/api/inference/keys') {
        if (route.request().method() === 'POST') {
          const policy = route.request().postDataJSON()
          keys.push({ ...policy, id: 'operator', prefix: 'sk-studio-test' })
          data = { key: 'sk-studio-test-secret' }
        } else data = { keys }
      }
      await route.fulfill({ contentType: 'application/json', body: JSON.stringify(data) })
    })
    await page.goto('/proxy')
    await expect(page.locator('.traffic-chart title').first()).toContainText(
      '3 retained HTTP requests',
    )
    const launcher = page.getByRole('button', { name: /^Search workspace/ })
    await launcher.click()
    await expect(page.getByRole('dialog', { name: 'Go to workspace' })).toBeVisible()
    await page.keyboard.press('Escape')
    await expect(page.getByRole('dialog', { name: 'Go to workspace' })).toBeHidden()
    await expect(launcher).toBeFocused()
    await page.keyboard.press('Control+k')
    let dialog = page.getByRole('dialog', { name: 'Go to workspace' })
    const search = dialog.getByRole('combobox')
    await expect(search).toBeFocused()
    await search.fill('backup')
    await expect(dialog.getByRole('option')).toHaveCount(1)
    await search.press('Enter')
    await expect(page).toHaveURL(/\/system#config-backup$/)
    await expect(page.locator('#main-content')).toBeFocused()
    await expect(dialog).toBeHidden()
    await page.getByRole('button', { name: /^Search workspace/ }).click()
    dialog = page.getByRole('dialog', { name: 'Go to workspace' })
    await dialog.getByRole('combobox').fill('proxy')
    await dialog.getByRole('combobox').press('Enter')
    await expect(page).toHaveURL(/\/proxy$/)
    await expect(dialog).toBeHidden()
    await page.getByRole('tab', { name: 'Access keys', exact: true }).click()
    await expect(page.locator('.key-list')).toContainText('Production')
    await expect(page.getByLabel('Key name', { exact: true })).toHaveCount(0)
    await page.getByRole('button', { name: 'New key', exact: true }).click()
    const creation = page.getByRole('dialog', { name: 'Create inference key' })
    await expect(creation.getByLabel('Key name', { exact: true })).toBeFocused()
    await creation.getByLabel('Key name', { exact: true }).fill('Operator')
    await creation.getByRole('button', { name: 'Create key', exact: true }).click()
    const secret = page.getByRole('dialog', { name: 'Save your inference key' })
    await expect(secret.getByLabel('Inference key', { exact: true })).toHaveValue(
      'sk-studio-test-secret',
    )
    await secret.getByRole('button', { name: 'Done', exact: true }).click()
    await expect(secret).toBeHidden()
    await expect(page.locator('.key-list')).toContainText('Operator')
    await expect(page.getByText('sk-studio-test-secret', { exact: true })).toHaveCount(0)
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
      width,
    )
  })
}
