/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve, dirname } from 'path'
import { fileURLToPath } from 'url'
import { readFileSync } from 'fs'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)
const pkg = JSON.parse(readFileSync(resolve(__dirname, '../package.json'), 'utf-8'))

export default defineConfig({
  plugins: [vue()],
  test: {
    environment: 'happy-dom',
    globals: true,
    include: ['src/**/*.test.js'],
    restoreMocks: true,
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
})
