import { expect, test } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'
import { installWorkspaceApi } from './fixtures/workspace.mjs'

const registry = {
  sections: [
    {
      id: 'runtime',
      label: 'Runtime',
      params: [
        {
          key: 'ctx_size',
          label: 'Context size',
          type: 'int',
          default: 4096,
          supported: true,
          flags: ['--ctx-size'],
        },
        ...Array.from({ length: 15 }, (_, index) => ({
          key: `option_${index}`,
          label: `Runtime option ${index}`,
          type: 'int',
          default: 1,
          supported: true,
        })),
      ],
    },
  ],
}

async function installConfiguration(page) {
  await page.route('**/api/models/param-registry*', (route) => route.fulfill({ json: registry }))
  await page.route('**/api/models/demo-model/config', (route) =>
    route.fulfill({
      json: {
        engine: 'llama_cpp',
        engines: {
          llama_cpp: {
            ctx_size: 8192,
            ...Object.fromEntries(Array.from({ length: 15 }, (_, index) => [`option_${index}`, 2])),
          },
        },
      },
    }),
  )
}

test('pages reflow and configuration actions remain above mobile navigation in both themes', async ({
  page,
}) => {
  test.setTimeout(120000)
  await installWorkspaceApi(page)
  await installConfiguration(page)
  for (const theme of ['dark', 'light']) {
    await page.addInitScript((value) => localStorage.setItem('theme', value), theme)
    for (const width of [320, 390, 900, 1440]) {
      await page.setViewportSize({ width, height: 900 })
      for (const path of [
        '/models',
        '/audio',
        '/engines',
        '/proxy',
        '/search',
        '/system',
        '/models/demo-model/config',
      ]) {
        await page.goto(path)
        await expect(page.locator('main h1').first()).toBeVisible()
        expect(
          await page.evaluate(() => document.documentElement.scrollWidth),
          `${theme} ${width} ${path}`,
        ).toBeLessThanOrEqual(width)
      }
      const save = page.getByRole('button', { name: 'Save changes', exact: true })
      await save.scrollIntoViewIfNeeded()
      const bounds = await save.boundingBox()
      const nav = await page.getByRole('navigation', { name: 'Main', exact: true }).boundingBox()
      if (width <= 900) expect(bounds.y + bounds.height).toBeLessThanOrEqual(nav.y)
      expect(
        await save.evaluate((node) => {
          const box = node.getBoundingClientRect()
          return node.contains(
            document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2),
          )
        }),
      ).toBe(true)
      const input = page.getByRole('spinbutton', { name: /Context size/ })
      await expect(input).toHaveAttribute('id', 'basic-ctx_size')
      await expect(input).toHaveAttribute('aria-describedby', 'basic-help-ctx_size')
    }
  }
})

test('library filters recover cleanly and single-model cards expose their main actions', async ({
  page,
}) => {
  await installWorkspaceApi(page)
  await page.goto('/models')
  await expect(page.getByRole('button', { name: 'Connect Demo', exact: true })).toBeVisible()
  await page.getByRole('button', { name: /^Filters/ }).click()
  await page.getByLabel('Engine', { exact: true }).selectOption('audio_cpp')
  await expect(page.getByRole('status').filter({ hasText: '1 of 2 models' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Engine: audio.cpp ×', exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Clear filters', exact: true }).click()
  await expect(page.getByRole('status').filter({ hasText: '2 of 2 models' })).toBeVisible()
  await page.getByRole('button', { name: 'List', exact: true }).click()
  await expect(page.getByRole('table', { name: 'Model library' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Connect Demo', exact: true })).toBeVisible()
})

test('Connect separates unavailable keys from policy, saves URLs deliberately and guards secrets', async ({
  page,
}) => {
  await installWorkspaceApi(page)
  let failKeys = true
  await page.route('**/api/inference/keys', async (route) => {
    if (route.request().method() === 'POST')
      return route.fulfill({ json: { key: 'sk-one-time-ui-test' } })
    return failKeys
      ? route.fulfill({ status: 503, json: { detail: 'offline' } })
      : route.fulfill({ json: { keys: [] } })
  })
  let saves = 0
  await page.route('**/api/settings/inference', (route) => {
    if (route.request().method() === 'PUT') saves++
    return route.fulfill({
      json: {
        public_inference_url:
          route.request().method() === 'PUT'
            ? route.request().postDataJSON().public_inference_url
            : '',
      },
    })
  })
  await page.goto('/models')
  await page.getByRole('button', { name: 'Connect Demo', exact: true }).click()
  const dialog = page.getByRole('dialog', { name: 'Connect', exact: true })
  await expect(dialog).toContainText('Could not load access keys')
  await expect(dialog).toContainText('An access key is required')
  await expect(dialog).not.toContainText('No access keys configured')
  await dialog.getByLabel('Public URL', { exact: true }).fill('https://inference.example.test')
  await dialog.getByLabel('Public URL', { exact: true }).press('Tab')
  expect(saves).toBe(0)
  await dialog.getByRole('button', { name: 'Save public URL', exact: true }).click()
  await expect(dialog.getByText('Public URL saved.', { exact: true })).toBeVisible()
  expect(saves).toBe(1)
  await expect(dialog.getByRole('button', { name: 'Copy cURL', exact: true })).toBeVisible()
  failKeys = false
  await dialog.getByRole('button', { name: 'Retry keys', exact: true }).click()
  await expect(dialog).toContainText('No access keys configured')
  await dialog.getByLabel('New access key name').fill('Desktop')
  await dialog.getByRole('button', { name: 'Create key', exact: true }).click()
  await expect(dialog).toContainText('sk-one-time-ui-test')
  await page.keyboard.press('Escape')
  const confirmation = page.getByRole('alertdialog', { name: 'Close without saving this key?' })
  await expect(confirmation).toBeVisible()
  await confirmation.getByRole('button', { name: 'Keep key open', exact: true }).click()
  await dialog.getByRole('button', { name: 'I saved this key', exact: true }).click()
  await page.keyboard.press('Escape')
  await expect(dialog).toBeHidden()
})

test('Audio gives a retryable error when the catalog fails rather than claiming no models are installed', async ({
  page,
}) => {
  await installWorkspaceApi(page)
  await page.route('**/api/models', (route) =>
    route.fulfill({ status: 503, json: { detail: 'catalog offline' } }),
  )
  await page.goto('/audio')
  await expect(page.getByRole('heading', { name: 'Could not load audio models' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Retry', exact: true })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'No audio.cpp models installed' })).toHaveCount(0)
})

test('Activity opens within the viewport and Escape returns focus to its header control', async ({
  page,
}) => {
  await installWorkspaceApi(page)
  await page.setViewportSize({ width: 320, height: 900 })
  await page.goto('/models')
  const toggle = page.getByRole('button', { name: 'Activity', exact: true })
  await toggle.click()
  const panel = page.getByRole('region', { name: 'Activity', exact: true })
  await expect(panel).toBeVisible()
  const box = await panel.boundingBox()
  expect(box.x).toBeGreaterThanOrEqual(0)
  expect(box.x + box.width).toBeLessThanOrEqual(320)
  await panel.getByRole('button', { name: 'Close activity', exact: true }).focus()
  await page.keyboard.press('Escape')
  await expect(panel).toBeHidden()
  await expect(toggle).toBeFocused()
})

test('Connect request formats and error states pass automated accessibility checks in both themes', async ({
  page,
}) => {
  await installWorkspaceApi(page)
  await page.setViewportSize({ width: 1440, height: 1000 })
  for (const theme of ['dark', 'light']) {
    await page.addInitScript((value) => localStorage.setItem('theme', value), theme)
    await page.goto('/models')
    await page.getByRole('button', { name: 'Connect Demo', exact: true }).click()
    const dialog = page.getByRole('dialog', { name: 'Connect', exact: true })
    await expect(dialog.getByRole('button', { name: 'Copy cURL', exact: true })).toBeVisible()
    await dialog.getByRole('button', { name: 'JSON body', exact: true }).click()
    await expect(dialog.getByRole('button', { name: 'Copy JSON body', exact: true })).toBeVisible()
    await dialog.getByLabel('Public URL', { exact: true }).fill('ftp://invalid.example')
    await dialog.getByRole('button', { name: 'Save public URL', exact: true }).click()
    await expect(dialog.getByLabel('Public URL', { exact: true })).toHaveAttribute(
      'aria-invalid',
      'true',
    )
    const audit = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa', 'best-practice'])
      .analyze()
    expect(audit.violations).toEqual([])
  }
})

test('library table keeps status, file sizes and actions in consistent columns', async ({
  page,
}) => {
  await installWorkspaceApi(page)
  await page.route('**/api/models', (route) =>
    route.fulfill({
      json: [
        {
          base_model_name: 'Running model with a long descriptive name',
          huggingface_id: 'org/running',
          quantizations: [
            {
              id: 'running',
              name: 'Running model',
              quantization: 'Q4_K_M',
              format: 'gguf',
              file_size: 4800000000,
              is_active: true,
              status: 'ready',
              config: { engine: 'llama_cpp' },
            },
          ],
        },
        {
          base_model_name: 'Stopped model',
          huggingface_id: 'org/stopped',
          quantizations: [
            {
              id: 'stopped',
              name: 'Stopped model',
              quantization: 'Q8_0',
              format: 'gguf',
              file_size: 12700000000,
              is_active: false,
              config: { engine: 'llama_cpp' },
            },
          ],
        },
        {
          base_model_name: 'Multiple variants',
          huggingface_id: 'org/variants',
          quantizations: [
            {
              id: 'variant-small',
              quantization: 'Q4_K_M',
              format: 'gguf',
              file_size: 3500000000,
              is_active: false,
              config: { engine: 'llama_cpp' },
            },
            {
              id: 'variant-large',
              quantization: 'Q8_0',
              format: 'gguf',
              file_size: 7000000000,
              is_active: false,
              config: { engine: 'llama_cpp' },
            },
          ],
        },
      ],
    }),
  )
  for (const width of [1440, 768, 390, 320]) {
    await page.setViewportSize({ width, height: 1000 })
    await page.goto('/models?layout=list')
    const table = page.getByRole('table', { name: 'Model library' })
    await expect(table).toBeVisible()
    await expect(table.getByText('12.7 GB', { exact: true })).toBeVisible()
    const sizes = await table.locator('.library-list__size').evaluateAll((cells) =>
      cells.map((cell) => {
        const box = cell.getBoundingClientRect()
        return { right: box.right, left: box.left }
      }),
    )
    expect(new Set(sizes.map((box) => Math.round(box.right))).size).toBe(1)
    const rows = table.locator('.library-list__row')
    for (const row of await rows.all()) {
      const size = await row.locator('.library-list__size').boundingBox()
      const actions = await row.locator('.library-list__actions').boundingBox()
      if (width > 1100) expect(size.x + size.width).toBeLessThan(actions.x)
      else expect(size.y + size.height).toBeLessThanOrEqual(actions.y)
    }
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
      width,
    )
    if (width <= 1100)
      await table.evaluate((node) => {
        const headerHeight = document.querySelector('.layout-header').getBoundingClientRect().height
        window.scrollTo(0, window.scrollY + node.getBoundingClientRect().top - headerHeight - 16)
      })
    await page.screenshot({ path: `docs/ui-review/table-${width}.png`, fullPage: width === 1440 })
  }
})

test('shared headers stay compact and page guidance opens on demand', async ({ page }) => {
  await installWorkspaceApi(page)
  const paths = [
    '/models',
    '/audio',
    '/search',
    '/proxy',
    '/engines',
    '/system',
    '/models/demo-model/config',
  ]
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: 900 })
    for (const path of paths) {
      await page.goto(path)
      await expect(page.locator('main h1').first()).toBeVisible()
      const header = page.locator('main .page-header').first()
      const bounds = await header.boundingBox()
      if (width === 1440) expect(bounds.height, path).toBeLessThanOrEqual(64)
      const appHeader = page.locator('.layout-header')
      const appBounds = await appHeader.boundingBox()
      expect(appBounds.height, `${width} global header`).toBeLessThanOrEqual(57)
      await expect
        .poll(async () =>
          page.evaluate(() =>
            Number.parseFloat(
              document.documentElement.style.getPropertyValue('--app-header-height'),
            ),
          ),
        )
        .toBe(appBounds.height)
      expect(
        await page.evaluate(() => document.documentElement.scrollWidth),
        `${width} ${path}`,
      ).toBeLessThanOrEqual(width)
      const description = header.locator('.page-header__description')
      if (await description.count()) {
        await expect(description).toBeHidden()
        const help = header.locator('summary')
        await help.click()
        await expect(description).toBeVisible()
        await help.press('Escape')
        await expect(description).toBeHidden()
        await expect(help).toBeFocused()
      }
      if (path === '/models' || (path === '/proxy' && width === 1440)) {
        await page.screenshot({ path: `docs/ui-review/compact-${path.slice(1)}-${width}.png` })
      }
    }
  }
})
