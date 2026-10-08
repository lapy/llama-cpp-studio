import js from '@eslint/js'
import pluginVue from 'eslint-plugin-vue'

const browserGlobals = {
  AbortController: 'readonly',
  AbortSignal: 'readonly',
  atob: 'readonly',
  Blob: 'readonly',
  btoa: 'readonly',
  Buffer: 'readonly',
  clearInterval: 'readonly',
  clearTimeout: 'readonly',
  console: 'readonly',
  CSS: 'readonly',
  CustomEvent: 'readonly',
  document: 'readonly',
  Event: 'readonly',
  EventSource: 'readonly',
  fetch: 'readonly',
  File: 'readonly',
  FileReader: 'readonly',
  FormData: 'readonly',
  Headers: 'readonly',
  HTMLElement: 'readonly',
  HTMLAnchorElement: 'readonly',
  location: 'readonly',
  localStorage: 'readonly',
  MediaRecorder: 'readonly',
  navigator: 'readonly',
  performance: 'readonly',
  queueMicrotask: 'readonly',
  requestAnimationFrame: 'readonly',
  sessionStorage: 'readonly',
  setInterval: 'readonly',
  setTimeout: 'readonly',
  URL: 'readonly',
  window: 'readonly',
  __APP_VERSION__: 'readonly',
}

export default [
  { ignores: ['frontend/dist/**', 'frontend/src/api/openapi.json'] },
  js.configs.recommended,
  ...pluginVue.configs['flat/essential'],
  {
    files: ['frontend/src/**/*.{js,vue}', 'frontend/e2e/**/*.mjs'],
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
      globals: browserGlobals,
    },
    rules: {
      'no-empty': ['error', { allowEmptyCatch: true }],
      'no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
        },
      ],
      'no-useless-assignment': 'error',
      'vue/multi-word-component-names': 'off',
    },
  },
  {
    files: ['frontend/**/*.test.js', 'frontend/e2e/**/*.mjs', 'frontend/vitest.setup.js'],
    languageOptions: {
      globals: {
        ...browserGlobals,
        afterEach: 'readonly',
        beforeEach: 'readonly',
        describe: 'readonly',
        expect: 'readonly',
        it: 'readonly',
        process: 'readonly',
        test: 'readonly',
        vi: 'readonly',
      },
    },
  },
]
