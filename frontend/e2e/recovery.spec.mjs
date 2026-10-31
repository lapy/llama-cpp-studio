import { expect, test } from '@playwright/test'
import { installWorkspaceApi } from './fixtures/workspace.mjs'

test.beforeEach(async ({ page }) => {
  await installWorkspaceApi(page)
})

test('proxy refresh preserves edits and navigation offers Stay and Discard', async ({ page }) => {
  await page.goto('/proxy?tab=settings')
  const port = page.getByLabel(/Inference port/)
  await port.fill('2333')
  await page.getByRole('button', { name: 'Refresh proxy status' }).click()
  await expect(port).toHaveValue('2333')
  await page.getByRole('link', { name: 'Models', exact: true }).first().click()
  await expect(page.getByRole('dialog', { name: 'Unsaved configuration' })).toBeVisible()
  await page.getByRole('button', { name: 'Stay', exact: true }).click()
  await expect(port).toHaveValue('2333')
  await page.getByRole('link', { name: 'Models', exact: true }).first().click()
  await page.getByRole('button', { name: 'Discard', exact: true }).click()
  await expect(page).toHaveURL(/\/models/)
})

test('proxy tab and request inspector survive reload and browser history', async ({ page }) => {
  await page.goto('/proxy?tab=requests&request=request-1')
  await expect(page.getByRole('dialog')).toBeVisible()
  await page.getByRole('button', { name: 'Close', exact: true }).last().click()
  await expect(page).not.toHaveURL(/request=request-1/)
  await page.reload()
  await expect(page.getByRole('tab', { name: 'Requests', exact: true })).toHaveAttribute(
    'aria-selected',
    'true',
  )
  await page.goto('/proxy?tab=keys')
  await page.goBack()
  await expect(page.getByRole('tab', { name: 'Requests', exact: true })).toHaveAttribute(
    'aria-selected',
    'true',
  )
})

test('models and engines retain filters in shared URLs', async ({ page }) => {
  await page.goto('/models?status=running&layout=list&q=Demo')
  await expect(page.locator('#library-status')).toHaveValue('running')
  await page.reload()
  await expect(page.locator('#library-status')).toHaveValue('running')
  await page.goto('/engines?task=audio&hardware=cpu')
  await expect(page.locator('#engine-task-filter')).toHaveValue('audio')
  await expect(page.locator('#engine-hardware-filter')).toHaveValue('cpu')
})

test('unknown addresses retain workspace navigation and offer a useful destination', async ({
  page,
}) => {
  await page.goto('/missing-workspace')
  await expect(page.getByRole('heading', { name: 'Page not found' })).toBeVisible()
  await page.getByRole('link', { name: 'Return to Models' }).click()
  await expect(page).toHaveURL(/\/models/)
})

test('failed live reads preserve data and recover through an explicit read retry', async ({
  page,
}) => {
  let fail = false
  await page.route('**/api/proxy', async (route) => {
    if (fail) return route.fulfill({ status: 503, json: { detail: 'Temporary read failure' } })
    return route.fallback()
  })
  await page.goto('/proxy')
  await expect(page.getByText('Live', { exact: true })).toBeVisible()
  fail = true
  await page.getByRole('button', { name: 'Refresh proxy status' }).click()
  await expect(page.getByText('Connection interrupted · retrying', { exact: true })).toBeVisible()
  await expect(page.getByRole('tab', { name: 'Overview', exact: true })).toBeVisible()
  fail = false
  await page.getByRole('button', { name: 'Refresh proxy status' }).click()
  await expect(page.getByText('Live', { exact: true })).toBeVisible()
})

test('a failed route module offers recovery while keeping workspace navigation available', async ({
  page,
}) => {
  const modulePath = '**/src/views/ProxyView.vue*'
  await page.route(modulePath, (route) => route.abort())
  await page.goto('/models')
  await page.getByRole('link', { name: 'Proxy', exact: true }).first().click()
  await expect(page.getByRole('heading', { name: 'Unable to open this view' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Retry navigation' })).toBeVisible()
  await page.unroute(modulePath)
  await page.getByRole('button', { name: 'Reload Studio' }).click()
  await expect(page.getByRole('heading', { name: 'Model library', exact: true })).toBeVisible()
  await page.getByRole('link', { name: 'Proxy', exact: true }).first().click()
  await expect(page.getByRole('heading', { name: 'Proxy', exact: true })).toBeVisible()
})
