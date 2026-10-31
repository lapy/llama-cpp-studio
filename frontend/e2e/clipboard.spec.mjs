import { expect, test } from '@playwright/test'
import { installWorkspaceApi } from './fixtures/workspace.mjs'

async function createKey(page, origin, key) {
  await installWorkspaceApi(page, {
    bridgeOrigin: origin.includes('studio.test') ? origin : undefined,
    key,
  })
  await page.goto(`${origin}/proxy`)
  await page.getByRole('tab', { name: 'Access keys', exact: true }).click()
  await page.getByRole('button', { name: 'New key', exact: true }).click()
  const form = page.getByRole('dialog', { name: 'Create inference key' })
  await form.getByLabel('Key name', { exact: true }).fill('Clipboard verification')
  await form.getByRole('button', { name: 'Create key', exact: true }).click()
  const secret = page.getByRole('dialog', { name: 'Save your inference key' })
  await expect(secret.getByLabel('Inference key', { exact: true })).toHaveValue(key)
  return secret
}

for (const [name, origin] of [
  ['secure', 'http://127.0.0.1:4179'],
  ['HTTP LAN', 'http://studio.test:4179'],
]) {
  test(`copies the actual secret and pastes it correctly on ${name}`, async ({ page }) => {
    const key = `sk-studio-${name === 'secure' ? 'secure' : 'http'}-clipboard-verification`
    const secret = await createKey(page, origin, key)
    expect(await page.evaluate(() => window.isSecureContext)).toBe(name === 'secure')
    if (name !== 'secure')
      expect(await page.evaluate(() => typeof navigator.clipboard)).toBe('undefined')
    await secret.getByRole('button', { name: 'Copy', exact: true }).click()
    await expect(secret.getByRole('status')).toHaveText('Key copied to clipboard.')
    await secret.evaluate((element) => {
      const probe = document.createElement('textarea')
      probe.setAttribute('aria-label', 'Clipboard verification')
      element.append(probe)
    })
    const paste = secret.getByLabel('Clipboard verification', { exact: true })
    await paste.focus()
    await page.keyboard.press('Control+v')
    await expect(paste).toHaveValue(key)
    await secret.getByRole('button', { name: 'Done', exact: true }).click()
    await expect(secret).toBeHidden()
    await expect(page.getByLabel('Inference key', { exact: true })).toHaveCount(0)
  })
}

test('reports blocked copying honestly and selects the key for manual copy', async ({ page }) => {
  await page.addInitScript(() => {
    document.execCommand = () => false
  })
  const key = 'sk-studio-manual-copy-verification'
  const secret = await createKey(page, 'http://studio.test:4179', key)
  await secret.getByRole('button', { name: 'Copy', exact: true }).click()
  await expect(secret.getByRole('status')).toContainText('Automatic copy is blocked')
  await expect(secret.getByRole('button', { name: 'Copied', exact: true })).toHaveCount(0)
  const field = secret.getByLabel('Inference key', { exact: true })
  await expect(field).toBeFocused()
  expect(await field.evaluate((node) => node.selectionEnd - node.selectionStart)).toBe(key.length)
  await expect(field).toHaveValue(key)
  await page.keyboard.press('Control+c')
  await secret.evaluate((element) => {
    const probe = document.createElement('textarea')
    probe.setAttribute('aria-label', 'Manual paste verification')
    element.append(probe)
  })
  const paste = secret.getByLabel('Manual paste verification')
  await paste.focus()
  await page.keyboard.press('Control+v')
  await expect(paste).toHaveValue(key)
})
