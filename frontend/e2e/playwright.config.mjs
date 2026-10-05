import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { defineConfig } from '@playwright/test'

const frontend = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

export default defineConfig({
  testDir: path.dirname(fileURLToPath(import.meta.url)),
  testMatch: /.*\.spec\.mjs/,
  timeout: 90_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: 1,
  use: {
    baseURL: 'http://127.0.0.1:4179',
    viewport: { width: 390, height: 844 },
  },
  webServer: {
    command: 'npx vite --port 4179 --strictPort --host 127.0.0.1',
    cwd: frontend,
    url: 'http://127.0.0.1:4179',
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
})
