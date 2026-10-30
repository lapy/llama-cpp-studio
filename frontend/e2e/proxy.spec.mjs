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
    await page.keyboard.press('Escape')
    for (const name of ['Events', 'Settings & routing', 'Engine diagnostics']) {
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
