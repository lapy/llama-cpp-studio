import { createApp } from 'vue'
import { createPinia } from 'pinia'
import PrimeVue from 'primevue/config'
import { definePreset } from '@primeuix/themes'
import Lara from '@primeuix/themes/lara'
import ConfirmationService from 'primevue/confirmationservice'
import ToastService from 'primevue/toastservice'
import Tooltip from 'primevue/tooltip'

import App from './App.vue'
import router from './router'
import './api/client.js'

import 'primeicons/primeicons.css'
import './styles/index.css'

const StudioPreset = definePreset(Lara, {
  semantic: {
    primary: {
      color: 'light-dark({indigo.600}, {indigo.400})',
      hoverColor: 'light-dark({indigo.700}, {indigo.300})',
      activeColor: 'light-dark({indigo.800}, {indigo.200})',
      50: '{indigo.50}',
      100: '{indigo.100}',
      200: '{indigo.200}',
      300: '{indigo.300}',
      400: '{indigo.400}',
      500: '{indigo.500}',
      600: '{indigo.600}',
      700: '{indigo.700}',
      800: '{indigo.800}',
      900: '{indigo.900}',
      950: '{indigo.950}',
    },
  },
  components: {
    message: {
      warn: { color: 'light-dark({yellow.800}, {yellow.300})' },
      error: { color: 'light-dark({red.800}, {red.300})' },
      success: { color: 'light-dark({green.800}, {green.300})' },
      info: { color: 'light-dark({blue.800}, {blue.300})' },
    },
    button: {
      outlined: { secondary: { color: 'light-dark({surface.600}, {surface.400})' } },
      text: {
        warn: { color: 'light-dark({orange.800}, {orange.300})' },
        danger: { color: 'light-dark({red.700}, {red.300})' },
        success: { color: 'light-dark({green.700}, {green.300})' },
        info: { color: 'light-dark({sky.700}, {sky.300})' },
      },
    },
  },
})

const app = createApp(App)

app.use(createPinia())
app.use(router)
app.use(PrimeVue, {
  license: import.meta.env.VITE_PRIMEUI_LICENSE,
  theme: {
    preset: StudioPreset,
    options: {
      darkModeSelector: '[data-theme="dark"]',
      cssLayer: false,
    },
  },
})
app.use(ConfirmationService)
app.use(ToastService)
app.directive('tooltip', Tooltip)

app.mount('#app')
