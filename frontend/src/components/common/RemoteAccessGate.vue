<template>
  <div v-if="pending" class="access-gate" role="status" aria-live="polite">
    <p class="access-gate__pending">Checking how this Studio is reached…</p>
  </div>
  <div
    v-else
    class="access-gate"
    role="dialog"
    aria-modal="true"
    aria-labelledby="access-gate-title"
    aria-describedby="access-gate-help"
  >
    <form class="access-gate__panel" @submit.prevent="submit">
      <div class="access-gate__copy">
        <div class="access-gate__mark" aria-hidden="true">
          <i class="pi pi-lock" />
        </div>
        <p class="access-gate__eyebrow">llama.cpp Studio</p>
        <h1 id="access-gate-title">External access is locked</h1>
        <p id="access-gate-help" class="access-gate__lead">
          This Studio is set to external access. The management interface stays closed until you
          enter the access password.
        </p>
        <p class="access-gate__notes">
          The password is the <code>STUDIO_API_TOKEN</code> value. If that variable is unset, Studio
          saved a generated password as <code>studio_access.token</code> in
          <code>data/config</code> (<code>/app/data/config</code> in Docker). Set
          <code>STUDIO_ACCESS_MODE=local</code> and restart to open this machine without a password.
        </p>
      </div>

      <div class="access-gate__form">
        <div class="access-gate__field">
          <label for="studio-access-password">Access password</label>
          <div class="access-password-control">
            <input
              id="studio-access-password"
              ref="passwordInput"
              v-model="password"
              class="p-inputtext p-component access-password-input"
              :type="passwordVisible ? 'text' : 'password'"
              autocomplete="current-password"
              required
              placeholder="Enter the access password"
              :disabled="submitting"
              :aria-invalid="Boolean(error)"
              :aria-describedby="error ? 'access-gate-help access-gate-error' : 'access-gate-help'"
            />
            <button
              type="button"
              class="password-toggle"
              :aria-label="passwordVisible ? 'Hide password' : 'Show password'"
              :aria-pressed="passwordVisible"
              :disabled="submitting"
              @click="passwordVisible = !passwordVisible"
            >
              <i :class="['pi', passwordVisible ? 'pi-eye-slash' : 'pi-eye']" aria-hidden="true" />
            </button>
          </div>
        </div>

        <p v-if="error" id="access-gate-error" class="access-gate__error" role="alert">
          {{ error }}
        </p>

        <Button
          type="submit"
          class="access-gate__submit"
          label="Unlock Studio"
          icon="pi pi-unlock"
          :loading="submitting"
          :disabled="submitting || !password.trim()"
        />
      </div>
    </form>
  </div>
</template>

<script setup>
import { nextTick, ref, watch } from 'vue'
import Button from 'primevue/button'

const props = defineProps({
  pending: { type: Boolean, default: false },
  error: { type: String, default: '' },
  submitting: { type: Boolean, default: false },
})

const emit = defineEmits(['submit'])
const password = ref('')

const passwordInput = ref(null),
  passwordVisible = ref(false)
watch(
  () => props.pending,
  async (pending) => {
    if (!pending) {
      await nextTick()
      passwordInput.value?.focus()
    }
  },
  { immediate: true },
)

function submit() {
  const token = password.value.trim()
  if (!token) return
  emit('submit', token)
}
</script>

<style scoped>
.access-password-control {
  position: relative;
}
.access-password-input {
  width: 100%;
  padding-right: 3rem;
}
.password-toggle {
  position: absolute;
  inset: 0.15rem 0.15rem 0.15rem auto;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 2.5rem;
  min-height: 2.5rem;
  border: 0;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-secondary);
  cursor: pointer;
}

.access-gate {
  position: fixed;
  inset: 0;
  z-index: 20000;
  display: flex;
  align-items: stretch;
  justify-content: center;
  overflow: hidden;
  background:
    radial-gradient(ellipse at top, rgba(34, 211, 238, 0.12), transparent 46%), var(--bg-primary);
}

.access-gate__pending {
  margin: auto;
  color: var(--text-secondary);
  font-size: 1rem;
}

.access-gate__panel {
  width: min(46rem, 100%);
  height: 100%;
  min-height: 0;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 1.25rem;
  padding: clamp(1.25rem, 4vw, 3rem);
  background: transparent;
}

.access-gate__copy {
  min-height: 0;
  overflow: auto;
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
}

.access-gate__form {
  flex: 0 0 auto;
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
}

.access-gate__mark {
  width: 3rem;
  height: 3rem;
  display: grid;
  place-items: center;
  border-radius: 999px;
  background: var(--accent-cyan-soft);
  color: var(--accent-cyan);
  font-size: 1.25rem;
}

.access-gate__eyebrow {
  margin: 0;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--accent-cyan);
}

.access-gate__panel h1 {
  margin: 0;
  font-size: clamp(1.75rem, 3vw, 2.25rem);
}

.access-gate__lead,
.access-gate__notes {
  margin: 0;
  color: var(--text-secondary);
  font-size: 1.05rem;
  line-height: 1.55;
}

.access-gate__notes code {
  color: var(--text-primary);
  font-size: 0.88em;
}

.access-gate__field {
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
}

.access-gate__field label {
  font-weight: 650;
  color: var(--text-primary);
}

.access-gate__error {
  margin: 0;
  padding: 0.75rem 0.9rem;
  border-radius: var(--radius-md);
  background: var(--status-error-soft);
  color: var(--status-error);
  border: 1px solid color-mix(in srgb, var(--status-error) 35%, transparent);
}

.access-gate__submit {
  width: 100%;
  justify-content: center;
  min-height: 2.75rem;
}

.access-gate :deep(.p-password) {
  width: 100%;
}
</style>
