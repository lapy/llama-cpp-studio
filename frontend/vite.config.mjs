/// <reference types="vitest/config" />
import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve, dirname } from 'path'
import { fileURLToPath } from 'url'
import { readFileSync } from 'fs'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)
const pkg = JSON.parse(readFileSync(resolve(__dirname, '../package.json'), 'utf-8'))

function nonEmptyEnv(value) {
  return typeof value === 'string' && value.trim() ? value.trim() : ''
}

function resolvePrimeUiLicense(mode) {
  const fromProcess = nonEmptyEnv(process.env.VITE_PRIMEUI_LICENSE)
  if (fromProcess) return fromProcess

  // A blank VITE_* process env overrides .env files. Drop it so Docker builds
  // without --build-arg can still pick up frontend/.env.local.
  delete process.env.VITE_PRIMEUI_LICENSE

  const fromFiles = {
    ...loadEnv(mode, resolve(__dirname, '..'), 'VITE_'),
    ...loadEnv(mode, resolve(__dirname, '.'), 'VITE_'),
  }
  return nonEmptyEnv(fromFiles.VITE_PRIMEUI_LICENSE)
}

export default defineConfig(({ command, mode }) => {
  const primeUiLicense = resolvePrimeUiLicense(mode)
  if (primeUiLicense) {
    process.env.VITE_PRIMEUI_LICENSE = primeUiLicense
  } else if (command === 'build') {
    throw new Error(
      'VITE_PRIMEUI_LICENSE is required for production builds. ' +
        'Add it to frontend/.env.local or pass --build-arg VITE_PRIMEUI_LICENSE=...'
    )
  }

  return {
    plugins: [vue()],
    test: {
      environment: 'happy-dom',
      globals: true,
      include: ['src/**/*.test.js'],
      setupFiles: ['./vitest.setup.js'],
      restoreMocks: true,
      // Vue + happy-dom integration tests share timer-heavy module state.
      // Serialize files to prevent intermittent timeouts and missed DOM events.
      pool: 'forks',
      maxWorkers: 1,
      fileParallelism: false,
      testTimeout: 10000,
    },
    root: resolve(__dirname, '.'),
    define: {
      __APP_VERSION__: JSON.stringify(pkg.version || '0.0.0'),
    },
    resolve: {
      alias: {
        '@': resolve(__dirname, 'src'),
      },
    },
    server: {
      port: 5173,
      // Fail fast if 5173 is taken so the dev URL stays predictable (avoids "browser won't load" on 5173 while Vite is on 5174).
      strictPort: true,
      host: true,        // listen on 0.0.0.0 so reachable from host (e.g. WSL → Windows browser)
      watch: {
        usePolling: true,
      },
      proxy: {
        '/api': {
          // Use IPv4 loopback so Node on Windows does not hit ::1 while Uvicorn is on IPv4 only.
          target: 'http://127.0.0.1:8081',
          changeOrigin: true,
        },
        // Approach A: Studio OpenAI audio proxy (ASR convert + speech passthrough)
        '/v1': {
          target: 'http://127.0.0.1:8081',
          changeOrigin: true,
        },
        '/audio-cpp-ui': {
          target: 'http://127.0.0.1:8081',
          changeOrigin: true,
        },
      },
    },
    build: {
      outDir: 'dist',
      assetsDir: 'assets',
      rollupOptions: {
        output: {
          entryFileNames: `assets/[name]-${Date.now()}.js`,
          chunkFileNames: `assets/[name]-${Date.now()}.js`,
          assetFileNames: `assets/[name]-${Date.now()}.[ext]`
        }
      }
    },
  }
})
