import { expect, test } from '@playwright/test'
import { installWorkspaceApi } from './fixtures/workspace.mjs'

for (const width of [390, 1440]) {
  test(`Hardware navigation at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 })
    await installWorkspaceApi(page)
    await page.goto('/engines')
    await expect(page.locator('.logo-copy')).toHaveText('Studio')
    await expect(page.locator('.brand-mark')).toBeVisible()
    await expect(page.getByRole('heading', { name: 'CUDA Toolkit' })).toHaveCount(0)
    await page.goto('/models')
    await page.goto('/engines#system')
    await expect(page).toHaveURL(/\/system#hardware$/)
    await expect(page.getByRole('heading', { name: 'Hardware resources' })).toBeVisible()
    for (const link of await page
      .getByRole('navigation', { name: 'System sections' })
      .getByRole('link')
      .all()) {
      expect(await link.evaluate((node) => node.scrollWidth <= node.clientWidth)).toBe(true)
    }
    await expect(page.getByRole('heading', { name: 'CUDA Toolkit' })).toBeVisible()
    await page.getByRole('button', { name: 'Install CUDA', exact: true }).click()
    await expect(page.getByRole('dialog')).toBeVisible()
    await page.keyboard.press('Escape')
    await expect(page.getByRole('dialog')).toBeHidden()
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
      width,
    )
    await page.screenshot({ path: `/tmp/studio-system-${width}.png`, fullPage: true })
  })
}

test('command import retains all list values and field help has one description', async ({
  page,
}) => {
  await installWorkspaceApi(page)
  let saved
  const description = 'Batch sizes used when decoding CUDA graphs.'
  await page.route('**/api/models/param-registry*', (route) =>
    route.fulfill({
      json: {
        sections: [
          {
            id: 'runtime',
            label: 'Runtime',
            params: [
              {
                key: 'cuda_graph_bs_decode',
                label: 'Decode batch sizes',
                type: 'list',
                supported: true,
                flags: ['--cuda-graph-bs-decode'],
                description,
                default: '1',
              },
            ],
          },
        ],
      },
    }),
  )
  await page.route('**/api/models/demo-model/config', (route) => {
    if (route.request().method() === 'PUT') saved = route.request().postDataJSON()
    return route.fulfill({ json: saved || { engine: 'llama_cpp', engines: { llama_cpp: {} } } })
  })
  await page.goto('/models/demo-model/config')
  await page.getByText('More actions', { exact: true }).click()
  await page.getByRole('button', { name: 'Import command', exact: true }).click()
  const dialog = page.getByRole('dialog')
  await dialog.getByLabel('Command text').fill('--cuda-graph-bs-decode 1 2 3')
  await expect(dialog.locator('.parse-preview-values strong')).toHaveText('1, 2, 3')
  await expect(dialog.getByText('Unknown options', { exact: true })).toHaveCount(0)
  await dialog.getByRole('button', { name: /^Apply 1 parameter/ }).click()
  await page.getByRole('button', { name: 'Save changes', exact: true }).click()
  await expect.poll(() => saved?.engines?.llama_cpp?.cuda_graph_bs_decode).toEqual(['1', '2', '3'])
  await expect(page.getByText(description, { exact: true })).toHaveCount(0)
  await expect(page.locator('[aria-label="About Decode batch sizes"]').first()).toBeAttached()
})

test('request parameter aliases save, reload, validate and remove without losing false values', async ({
  page,
}) => {
  await installWorkspaceApi(page)
  let saved = { engine: 'llama_cpp', engines: { llama_cpp: { model_alias: 'Assistant Name' } } }
  await page.route('**/api/models/demo-model/config', (route) => {
    if (route.request().method() === 'PUT') saved = route.request().postDataJSON()
    return route.fulfill({ json: saved })
  })
  await page.goto('/models/demo-model/config')
  await page.getByRole('tab', { name: 'Launch', exact: true }).click()
  await page.getByRole('button', { name: 'Add alias', exact: true }).click()
  const params = page.getByLabel('Request overrides (JSON object)', { exact: true })
  await params.fill('{broken')
  await expect(page.getByRole('button', { name: 'Save changes', exact: true })).toBeDisabled()
  await params.fill('{"chat_template_kwargs.enable_thinking": false, "thinking_effort": "low"}')
  await expect(page.getByText('assistant-name:thinking', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Save changes', exact: true }).click()
  await expect
    .poll(() => saved.engines.llama_cpp.set_params_by_id)
    .toEqual([
      {
        sub_id: 'thinking',
        params: { 'chat_template_kwargs.enable_thinking': false, thinking_effort: 'low' },
      },
    ])
  await page.reload()
  await page.getByRole('tab', { name: 'Launch', exact: true }).click()
  await expect(params).toHaveValue(/"thinking_effort": "low"/)
  await page.getByRole('button', { name: 'Remove alias 1', exact: true }).click()
  await page.getByRole('button', { name: 'Save changes', exact: true }).click()
  await expect.poll(() => saved.engines.llama_cpp.set_params_by_id).toEqual([])
})

test('native routing aliases retain multiple targets and save without backend selection', async ({
  page,
}) => {
  await installWorkspaceApi(page)
  let settings = { proxy_port: 2000, routing_aliases: { balanced: ['demo-model'] } }
  let saved
  await page.route('**/api/proxy', (route) =>
    route.fulfill({
      json: {
        settings,
        configured_backend: 'studio',
        active_backend: 'studio',
        running: [],
        catalog: [],
        requests: [],
        activity: [],
      },
    }),
  )
  await page.route('**/api/proxy/settings', (route) => {
    saved = route.request().postDataJSON()
    settings = saved
    return route.fulfill({
      json: { settings, restart_required: false, applied_live_fields: ['routing_aliases'] },
    })
  })
  await page.goto('/proxy')
  await page.getByRole('tab', { name: 'Settings & routing', exact: true }).click()
  await expect(page.getByLabel('Alias name', { exact: true })).toHaveValue('balanced')
  await page.getByLabel('Alias targets', { exact: true }).fill('demo-model, speech-model')
  await page.getByRole('button', { name: 'Save settings', exact: true }).click()
  await expect
    .poll(() => saved?.routing_aliases)
    .toEqual({ balanced: ['demo-model', 'speech-model'] })
  expect(saved).not.toHaveProperty('backend')
  await expect(page.getByLabel('Alias targets', { exact: true })).toHaveValue(
    'demo-model, speech-model',
  )
})
