/**
 * Production navigation measurements against a built server.
 *
 * This is not the compressed asset-size budget and not a vite preview.
 * The measured page is /models with the seeded safetensors model "Demo".
 * scripts/measure-production-navigation.sh starts that fixture. A bare URL
 * still has to be serving that same library view.
 *
 * Document timing comes from PerformanceNavigationTiming after the load
 * event. Transfer bytes are the document plus same-origin assets, not API
 * responses. Ceilings sit above the spread of several uncached runs.
 */
import { chromium } from '@playwright/test'

const baseUrl = (process.env.STUDIO_NAVIGATION_URL || '').replace(/\/$/, '')
const runs = Number(process.env.STUDIO_NAVIGATION_RUNS || 5)
const documentCeilingMs = 800
const transferCeilingBytes = 400_000
const usableCeilingMs = 1_000
// Spread from five uncached runs on 2026-10-05 against the built server:
// document 121–262 ms, document-plus-asset transfer 232,616 bytes,
// time to the Demo control 252–437 ms.
const libraryPath = '/models'
const controlName = /^(Start|Configure|Connect) Demo\b/

if (!baseUrl) {
  throw new Error('STUDIO_NAVIGATION_URL is required and must serve the built studio')
}

function median(values) {
  const ordered = [...values].sort((a, b) => a - b)
  const mid = Math.floor(ordered.length / 2)
  if (ordered.length % 2) return ordered[mid]
  return (ordered[mid - 1] + ordered[mid]) / 2
}

const browser = await chromium.launch()
const samples = []
try {
  for (let index = 0; index < runs; index += 1) {
    const context = await browser.newContext()
    const page = await context.newPage()
    const session = await context.newCDPSession(page)
    await session.send('Network.enable')
    await session.send('Network.setCacheDisabled', { cacheDisabled: true })
    await page.goto(`${baseUrl}${libraryPath}`, { waitUntil: 'load' })
    const control = page.getByRole('button', { name: controlName })
    await control.first().waitFor({ timeout: usableCeilingMs })
    const measured = await page.evaluate(() => {
      const navigation = performance.getEntriesByType('navigation')[0]
      const resources = performance.getEntriesByType('resource')
      const assetBytes = resources
        .filter((entry) => {
          const url = new URL(entry.name)
          if (url.origin !== location.origin) return false
          if (url.pathname.startsWith('/api')) return false
          return url.pathname.startsWith('/assets/')
            || /\.(?:js|css|woff2?|svg|ico|png|webp)$/.test(url.pathname)
        })
        .reduce((sum, entry) => sum + (entry.transferSize || 0), 0)
      return {
        documentMs: navigation ? navigation.duration : null,
        loadEventEnd: navigation ? navigation.loadEventEnd : 0,
        transferSize: (navigation ? navigation.transferSize || 0 : 0) + assetBytes,
      }
    })
    if (measured.documentMs == null || measured.loadEventEnd <= 0) {
      throw new Error('document navigation had not finished loading')
    }
    if (measured.transferSize <= 0) {
      throw new Error('document and asset transfer size was empty')
    }
    samples.push({
      documentMs: measured.documentMs,
      transferSize: measured.transferSize,
      usableMs: await page.evaluate(() => performance.now()),
    })
    await context.close()
  }
} finally {
  await browser.close()
}

function spread(pick) {
  const values = samples.map(pick)
  return { min: Math.min(...values), median: median(values), max: Math.max(...values) }
}

const documentSpread = spread((sample) => sample.documentMs)
const transferSpread = spread((sample) => sample.transferSize)
const usableSpread = spread((sample) => sample.usableMs)
console.log('production navigation measurements')
console.log(`document ms ${JSON.stringify(documentSpread)} ceiling ${documentCeilingMs}`)
console.log(`transfer bytes ${JSON.stringify(transferSpread)} ceiling ${transferCeilingBytes}`)
console.log(`usable ms ${JSON.stringify(usableSpread)} ceiling ${usableCeilingMs}`)

const failures = []
if (documentSpread.max > documentCeilingMs) failures.push('document timing')
if (transferSpread.max > transferCeilingBytes) failures.push('transfer size')
if (usableSpread.max > usableCeilingMs) failures.push('time to control')
if (failures.length) {
  throw new Error(`production navigation measurements exceeded: ${failures.join(', ')}`)
}
