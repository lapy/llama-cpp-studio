<template>
  <Transition name="swap-banner">
    <div
      v-if="show"
      class="swap-banner"
      role="status"
      aria-live="polite"
    >
      <div class="swap-banner__glow" aria-hidden="true" />
      <div class="swap-banner__inner">
        <div class="swap-banner__icon-wrap">
          <i class="pi pi-file-edit" aria-hidden="true" />
        </div>
        <div class="swap-banner__body">
          <div class="swap-banner__title-row">
            <span class="swap-banner__title">llama-swap config is out of date</span>
            <span class="swap-banner__badge">pending</span>
          </div>
          <p class="swap-banner__lead">
            The file on disk does not match your current models and engine settings.
          </p>
          <ul v-if="changes.length" class="swap-banner__changes">
            <li v-for="(line, idx) in changes" :key="idx">{{ line }}</li>
          </ul>
          <p class="swap-banner__warn">
            <i class="pi pi-exclamation-triangle" aria-hidden="true" />
            Applying writes <code>llama-swap-config.yaml</code> and reloads the proxy — all loaded models will stop.
          </p>
        </div>
        <div class="swap-banner__actions">
          <Button
            label="Apply configuration"
            icon="pi pi-check"
            severity="success"
            size="small"
            :loading="applying"
            :disabled="applying"
            @click="onApply"
          />
        </div>
      </div>
    </div>
  </Transition>
</template>

<script setup>
import { ref, computed } from 'vue'
import Button from 'primevue/button'
import { useToast } from 'primevue/usetoast'
import { useEnginesStore } from '@/stores/engines'

const enginesStore = useEnginesStore()
const toast = useToast()
const applying = ref(false)

const show = computed(() => {
  const s = enginesStore.swapConfigPending
  return Boolean(s?.applicable && s?.pending)
})

const changes = computed(() => enginesStore.swapConfigPending?.changes ?? [])

function formatErr(e) {
  const d = e?.response?.data?.detail
  if (Array.isArray(d)) return d.map((x) => x.msg || String(x)).join(' ')
  if (typeof d === 'string') return d
  return e?.message || 'Unknown error'
}

async function onApply() {
  applying.value = true
  try {
    await enginesStore.applySwapConfig()
    toast.add({
      severity: 'success',
      summary: 'Configuration applied',
      detail: 'llama-swap config was regenerated.',
      life: 3500,
    })
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'Apply failed',
      detail: formatErr(e),
      life: 5000,
    })
  } finally {
    applying.value = false
  }
}
</script>

<style scoped>
.swap-banner-enter-active,
.swap-banner-leave-active {
  transition: opacity 0.2s ease, transform 0.2s ease;
}
.swap-banner-enter-from,
.swap-banner-leave-to {
  opacity: 0;
  transform: translateY(-6px);
}

.swap-banner {
  position: relative;
  margin: 0 clamp(0.75rem, 2vw, 1.5rem);
  margin-top: 0.75rem;
  border-radius: var(--radius-lg, 12px);
  border: 1px solid var(--border-primary);
  background: linear-gradient(
    135deg,
    color-mix(in srgb, var(--accent-cyan, #22d3ee) 8%, var(--bg-card, #0f1419)) 0%,
    var(--bg-card, #0f1419) 55%
  );
  box-shadow:
    0 0 0 1px color-mix(in srgb, var(--accent-cyan, #22d3ee) 25%, transparent),
    0 12px 24px rgba(0, 0, 0, 0.25);
  overflow: hidden;
}

.swap-banner__glow {
  position: absolute;
  inset: 0;
  pointer-events: none;
  background: radial-gradient(
    ellipse 70% 50% at 10% 0%,
    color-mix(in srgb, var(--accent-cyan, #22d3ee) 18%, transparent),
    transparent 55%
  );
  opacity: 0.9;
}

.swap-banner__inner {
  position: relative;
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  padding: 1rem 1.15rem;
  flex-wrap: wrap;
}

.swap-banner__icon-wrap {
  flex-shrink: 0;
  width: 2.5rem;
  height: 2.5rem;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: color-mix(in srgb, var(--accent-cyan, #22d3ee) 15%, transparent);
  border: 1px solid color-mix(in srgb, var(--accent-cyan, #22d3ee) 35%, transparent);
  color: var(--accent-cyan, #22d3ee);
  font-size: 1.15rem;
}

.swap-banner__body {
  flex: 1;
  min-width: min(100%, 320px);
}

.swap-banner__title-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin-bottom: 0.35rem;
}

.swap-banner__title {
  font-weight: 700;
  font-size: 0.95rem;
  letter-spacing: -0.01em;
  color: var(--text-primary);
}

.swap-banner__badge {
  font-size: 0.65rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding: 0.15rem 0.45rem;
  border-radius: 999px;
  background: color-mix(in srgb, var(--accent-amber, #f59e0b) 22%, transparent);
  border: 1px solid color-mix(in srgb, var(--accent-amber, #f59e0b) 45%, transparent);
  color: var(--accent-amber, #fbbf24);
}

.swap-banner__lead {
  margin: 0 0 0.45rem;
  font-size: 0.82rem;
  font-weight: 500;
  color: var(--text-secondary);
  line-height: 1.45;
}

.swap-banner__changes {
  margin: 0 0 0.45rem;
  padding-left: 1.1rem;
  font-size: 0.8rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;
  color: var(--text-primary);
  line-height: 1.5;
  max-height: 5.5rem;
  overflow-y: auto;
}

.swap-banner__changes li {
  margin-bottom: 0.15rem;
}

.swap-banner__warn {
  margin: 0;
  display: flex;
  align-items: flex-start;
  gap: 0.4rem;
  font-size: 0.75rem;
  line-height: 1.45;
  color: color-mix(in srgb, var(--accent-amber, #f59e0b) 90%, var(--text-secondary));
}

.swap-banner__warn i {
  margin-top: 0.1rem;
  flex-shrink: 0;
}

.swap-banner__warn code {
  font-size: 0.72rem;
  padding: 0.05em 0.35em;
  border-radius: 4px;
  background: var(--bg-surface, rgba(0, 0, 0, 0.25));
  border: 1px solid var(--border-primary);
}

.swap-banner__actions {
  flex-shrink: 0;
  width: 100%;
  display: flex;
  justify-content: flex-end;
  align-items: flex-start;
}

@media (min-width: 720px) {
  .swap-banner__actions {
    width: auto;
    margin-left: auto;
  }
}
</style>
