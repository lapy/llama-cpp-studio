import { expect, test } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'
import fs from 'node:fs/promises'
import { installWorkspaceApi } from './fixtures/workspace.mjs'

async function audit(page, testInfo, state) {
  const result = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa', 'best-practice'])
    .analyze()
  const reportPath = testInfo.outputPath(`${state}-accessibility.json`)
  await fs.mkdir(testInfo.outputDir, { recursive: true })
  await fs.writeFile(
    reportPath,
    JSON.stringify(
      {
        url: page.url(),
        violations: result.violations,
        incomplete: result.incomplete,
        passedRules: result.passes.length,
      },
      null,
      2,
    ),
  )
  await testInfo.attach(`${state}-accessibility.json`, {
    path: reportPath,
    contentType: 'application/json',
  })

  expect
    .soft(
      result.violations,
      `${state}: ${result.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(' ')).join(', ')}`).join('; ')}`,
    )
    .toEqual([])
}

for (const theme of ['dark', 'light']) {
  test(`workspace follows automated WCAG rules in ${theme} mode`, async ({ page }, testInfo) => {
    test.setTimeout(120000)
    await page.setViewportSize({ width: 1440, height: 1000 })
    await page.emulateMedia({ reducedMotion: 'reduce' })
    await page.addInitScript((value) => localStorage.setItem('theme', value), theme)
    await installWorkspaceApi(page)
    for (const path of [
      '/models',
      '/search',
      '/audio',
      '/engines',
      '/models/demo-model/config',
      '/system',
      '/system#config-backup',
      '/system#history',
    ]) {
      await page.goto(path)
      await expect(page.locator('main h1').first()).toBeVisible()
      if (path === '/audio') {
        const modelPicker = page.getByRole('combobox', { name: 'Audio model', exact: true })
        await expect(modelPicker).toBeVisible()
        await expect(
          page.getByRole('combobox', { name: 'Voice preset', exact: true }),
        ).toBeVisible()
        await expect(
          page.getByRole('combobox', { name: 'Reference audio', exact: true }),
        ).toBeVisible()
        await modelPicker.press('ArrowDown')
        await expect(modelPicker).toHaveAttribute('aria-expanded', 'true')
        const controls = await modelPicker.getAttribute('aria-controls')
        await expect(page.locator(`[id="${controls}"]`)).toBeVisible()
        await modelPicker.press('Escape')
        await expect(modelPicker).toHaveAttribute('aria-expanded', 'false')
        await expect(page.locator(`[id="${controls}"]`)).toBeHidden()
      }
      await audit(page, testInfo, `${theme}-${path.replaceAll('/', '-').replace('#', '-')}`)
    }
    await page.goto('/proxy')
    for (const tab of ['Overview', 'Requests', 'Events', 'Access keys', 'Settings & routing']) {
      await page.getByRole('tab', { name: tab, exact: true }).click()
      await audit(page, testInfo, `${theme}-proxy-${tab}`)
    }
    await page.getByRole('tab', { name: 'Access keys', exact: true }).click()
    await page.getByRole('button', { name: 'Revoke', exact: true }).click()
    await expect(page.getByRole('alertdialog', { name: /Revoke/ })).toBeVisible()
    await audit(page, testInfo, `${theme}-revoke-confirmation`)
    await page.keyboard.press('Escape')
    await expect(page.getByRole('alertdialog', { name: /Revoke/ })).toBeHidden()
    await page.getByRole('button', { name: 'New key', exact: true }).click()
    await audit(page, testInfo, `${theme}-key-dialog`)
    const form = page.getByRole('dialog', { name: 'Create inference key' })
    for (let step = 0; step < 25; step++) {
      await page.keyboard.press('Tab')
      expect(await form.evaluate((element) => element.contains(document.activeElement))).toBe(true)
      await expect(page.locator('[aria-hidden="true"]:focus')).toHaveCount(0)
    }
    await form.getByLabel('Key name', { exact: true }).fill('Accessibility fixture')
    await form.getByRole('button', { name: 'Create key', exact: true }).click()
    await expect(page.getByRole('dialog', { name: 'Save your inference key' })).toBeVisible()
    await audit(page, testInfo, `${theme}-secret-dialog`)
    await page.keyboard.press('Escape')
    await page.getByRole('button', { name: /^Search workspace/ }).click()
    await expect(page.getByRole('dialog', { name: 'Go to workspace' })).toBeVisible()
    await audit(page, testInfo, `${theme}-navigator`)
  })
}

test('page navigation has meaningful titles and moves keyboard focus without stealing query edits', async ({
  page,
}) => {
  await installWorkspaceApi(page)
  await page.goto('/models')
  await expect(page).toHaveTitle('Models — llama.cpp Studio')
  await page
    .getByRole('navigation', { name: 'Main', exact: true })
    .getByRole('link', { name: 'Search', exact: true })
    .focus()
  await page.keyboard.press('Enter')
  await expect(page).toHaveTitle('Search models — llama.cpp Studio')
  await expect(page.locator('#main-content')).toBeFocused()
  expect(
    await page
      .locator('#main-content')
      .evaluate((node) => window.getComputedStyle(node).outlineWidth),
  ).toBe('2px')
  await page
    .getByRole('navigation', { name: 'Main', exact: true })
    .getByRole('link', { name: 'System', exact: true })
    .click()
  await page.getByRole('link', { name: 'Backup & restore', exact: true }).click()
  await expect(page).toHaveTitle('Backup & restore — llama.cpp Studio')
})

test('access password form exposes its label and error association', async ({ page }, testInfo) => {
  await installWorkspaceApi(page, { remote: true })
  await page.goto('/models')
  await expect(page.getByRole('heading', { name: 'External access is locked' })).toBeVisible()
  const password = page.getByLabel('Access password', { exact: true })
  await expect(password).toBeFocused()
  await page.getByRole('button', { name: 'Show password', exact: true }).focus()
  await page.keyboard.press('Space')
  await expect(password).toHaveAttribute('type', 'text')
  await page.keyboard.press('Space')
  await expect(password).toHaveAttribute('type', 'password')
  await password.fill('invalid-fixture-password')
  await page.getByRole('button', { name: 'Unlock Studio', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('not accepted')
  await expect(password).toHaveAttribute('aria-invalid', 'true')
  await expect(password).toHaveAttribute('aria-describedby', 'access-gate-help access-gate-error')
  await audit(page, testInfo, 'remote-access-gate')
  await page.route('**/api/session', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: '{}' }),
  )
  await password.fill('valid-fixture-password')
  await page.getByRole('button', { name: 'Unlock Studio', exact: true }).click()
  await expect(page.locator('main h1').first()).toBeVisible()
  await expect(page.locator('#main-content')).toBeFocused()
})

test('touch layouts retain minimum target sizes and reflow at 320 CSS pixels', async ({
  browser,
}, testInfo) => {
  test.setTimeout(120000)
  const context = await browser.newContext({
    viewport: { width: 320, height: 900 },
    isMobile: true,
    hasTouch: true,
  })
  const page = await context.newPage()
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await installWorkspaceApi(page)
  try {
    for (const theme of ['dark', 'light']) {
      await page.addInitScript((value) => localStorage.setItem('theme', value), theme)
      for (const path of ['/models', '/audio', '/proxy', '/system#config-backup']) {
        await page.goto(`http://127.0.0.1:4179${path}`)
        await expect(page.locator('main h1').first()).toBeVisible()
        expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
          320,
        )
        const undersized = await page.locator('button:visible:enabled').evaluateAll((buttons) =>
          buttons
            .filter((button) => {
              const { width, height } = button.getBoundingClientRect()
              return width < 24 || height < 24
            })
            .map((button) => ({
              label: button.getAttribute('aria-label') || button.textContent,
              width: button.getBoundingClientRect().width,
              height: button.getBoundingClientRect().height,
            })),
        )
        expect(undersized).toEqual([])
        await audit(page, testInfo, `touch-${theme}-${path.replaceAll('/', '-').replace('#', '-')}`)
        if (path === '/proxy') {
          await page.getByRole('tab', { name: 'Overview', exact: true }).focus()
          for (const label of ['Requests', 'Events', 'Access keys', 'Settings & routing']) {
            await page.keyboard.press('ArrowRight')
            const tab = page.getByRole('tab', { name: label, exact: true })
            await expect(tab).toBeFocused()
            const bounds = await tab.boundingBox()
            const strip = await page.getByRole('tablist', { name: 'Proxy sections' }).boundingBox()
            expect(bounds.x).toBeGreaterThanOrEqual(strip.x - 1)
            expect(bounds.x + bounds.width).toBeLessThanOrEqual(strip.x + strip.width + 1)
          }
        }
      }
    }
  } finally {
    await context.close()
  }
})
