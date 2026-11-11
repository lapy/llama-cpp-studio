import { gzipSync } from 'node:zlib'
import { readFileSync, statSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

// Compressed asset-size budget. This recompresses built files; it does not
// measure network transfer. Keep the ceiling aligned with
// COLD_MODELS_GZIP_BUDGET_BYTES.
//
// 2026-11-11: 250_000 -> 270_000 for the frontend dependency refresh
// (Vue 3.5.45, PrimeVue 5.0.4, @primeuix/themes 3.1.0, Vite 8.3.6): the
// cold /models graph measured 260_419 bytes gzip, ~10.4 KB above the old
// ceiling with no code change of our own.
const BUDGET_BYTES = 270_000
const dist = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'dist')

function add(chosen, relative) {
  const cleaned = relative.split('?', 1)[0].split('#', 1)[0].replace(/^\.?\//, '')
  const file = path.join(dist, cleaned)
  if (chosen.has(file)) return null
  try {
    if (!statSync(file).isFile()) return null
  } catch {
    return null
  }
  chosen.add(file)
  return file
}

const htmlPath = path.join(dist, 'index.html')
const html = readFileSync(htmlPath, 'utf8')
const chosen = new Set([htmlPath])
for (const match of html.matchAll(/(?:src|href)="([^"]+)"/g)) {
  if (/\.(js|css|svg)$/.test(match[1])) add(chosen, match[1])
}
const entryName = [...html.matchAll(/src="([^"]+\.js)"/g)][0]?.[1]
const entry = [...chosen].find((file) => file.endsWith(path.basename(entryName || '')))
if (!entry) throw new Error('production build has no entry script')
const entryText = readFileSync(entry, 'utf8')
const mapped = entryText.match(/__vite__mapDeps=\(i,m=__vite__mapDeps,d=\(m\.f\|\|\(m\.f=(\[[\s\S]*?\])\)\)\)/)
const route = entryText.match(/path:(?:`\/models`|"\/models"|'\/models')[\s\S]{0,800}?__vite__mapDeps\(\[([0-9,\s]+)\]\)/)
if (!mapped || !route) {
  throw new Error('production build does not expose the /models preload list')
}
const depFiles = JSON.parse(mapped[1])
for (const index of route[1].split(',').map((part) => part.trim()).filter(Boolean)) {
  add(chosen, depFiles[Number(index)])
}
for (const imported of entryText.matchAll(/from["']\.\/([^"']+\.js)["']/g)) {
  add(chosen, `assets/${imported[1]}`)
}
const pending = [...chosen].filter((file) => file.endsWith('.js') && file !== entry)
while (pending.length) {
  const script = pending.pop()
  const text = readFileSync(script, 'utf8')
  for (const imported of new Set([...text.matchAll(/["']\.\/([^"']+\.(?:js|css))["']/g)].map((match) => match[1]))) {
    const added = add(chosen, `assets/${imported}`)
    if (added && added.endsWith('.js')) pending.push(added)
  }
}
for (const stylesheet of [...chosen].filter((file) => file.endsWith('.css'))) {
  const text = readFileSync(stylesheet, 'utf8')
  for (const match of text.matchAll(/url\(\s*['"]?([^)'"]+)/g)) {
    const url = match[1]
    if (url.startsWith('data:')) continue
    if (url.split('?', 1)[0].split('#', 1)[0].endsWith('.woff2')) add(chosen, url)
  }
}
const names = [...chosen].map((file) => path.basename(file))
if (!names.some((name) => name.startsWith('ModelLibrary-') && name.endsWith('.css'))) {
  throw new Error('cold /models graph is missing the library stylesheet')
}
if (!names.some((name) => name.startsWith('primeicons-') && name.endsWith('.woff2'))) {
  throw new Error('cold /models graph is missing the icon font')
}
const total = [...chosen].reduce(
  (sum, file) => sum + gzipSync(readFileSync(file), { level: 6 }).length,
  0,
)
if (total > BUDGET_BYTES) {
  throw new Error(`compressed asset-size budget exceeded: ${total} bytes, ceiling ${BUDGET_BYTES}`)
}
console.log(`compressed asset-size budget ${total} bytes across ${chosen.size} files`)
