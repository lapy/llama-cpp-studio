<template>
  <PageSection
    id="config-names"
    title="Names &amp; routing"
    description="Every name this model can answer to, where it is stored, and the control that changes it."
  >
    <dl class="names-summary">
      <div class="names-summary__row">
        <dt class="names-summary__term">Catalog identity</dt>
        <dd class="names-summary__detail">
          <span class="names-summary__value">{{ displayName }}</span>
          <code v-if="catalogId" class="names-summary__code">{{ catalogId }}</code>
          <p class="names-summary__hint">
            Selects this model in Studio, its URLs, and management APIs. Inference clients never
            send this name.
          </p>
          <RouterLink class="names-summary__link" to="/models">Open in Models</RouterLink>
        </dd>
      </div>

      <div class="names-summary__row">
        <dt class="names-summary__term">Stable runtime id</dt>
        <dd class="names-summary__detail">
          <code v-if="stableModelId" class="names-summary__code">{{ stableModelId }}</code>
          <span v-else class="names-summary__value names-summary__value--muted"
            >Assigned the first time this model starts</span
          >
          <p class="names-summary__hint">
            Identifies the engine process, launch manifests, and runtime state. Studio manages it;
            it cannot be edited here.
          </p>
        </dd>
      </div>

      <div class="names-summary__row">
        <dt class="names-summary__term">Primary API name</dt>
        <dd class="names-summary__detail">
          <code v-if="primaryName" class="names-summary__code">{{ primaryName }}</code>
          <span v-else class="names-summary__value names-summary__value--muted"
            >Not set — clients use the stable runtime id</span
          >
          <p class="names-summary__hint">{{ primaryHint }}</p>
          <div class="names-summary__actions">
            <Button
              :label="primaryName ? 'Edit primary name' : 'Add primary name'"
              icon="pi pi-pencil"
              size="small"
              text
              @click="$emit('edit-section', 'launch')"
            />
          </div>
        </dd>
      </div>

      <div class="names-summary__row">
        <dt class="names-summary__term">Ordinary aliases</dt>
        <dd class="names-summary__detail">
          <span
            v-if="!ordinaryAliases.length"
            class="names-summary__value names-summary__value--muted"
            >No additional ordinary aliases are saved</span
          >
          <span v-else class="names-summary__chips">
            <code v-for="alias in ordinaryAliases" :key="alias" class="names-summary__chip">{{
              alias
            }}</code>
          </span>
          <p class="names-summary__hint">
            Additional names for the same process. Studio stores them as
            <code>swap_aliases</code> on the engine section and publishes them lower-case.
          </p>
          <div class="names-summary__actions">
            <Button
              label="Edit ordinary aliases"
              icon="pi pi-bookmark"
              size="small"
              text
              @click="$emit('manage-templates')"
            />
          </div>
        </dd>
      </div>

      <div class="names-summary__row">
        <dt class="names-summary__term">Request presets</dt>
        <dd class="names-summary__detail">
          <span
            v-if="!requestPresets.length"
            class="names-summary__value names-summary__value--muted"
            >No request presets are saved</span
          >
          <span v-else class="names-summary__chips">
            <code v-for="preset in requestPresets" :key="preset.key" class="names-summary__chip">
              {{ preset.name }}
              <small>{{ preset.count }} override{{ preset.count === 1 ? '' : 's' }}</small>
            </code>
          </span>
          <p class="names-summary__hint">{{ presetHint }}</p>
          <div class="names-summary__actions">
            <Button
              label="Edit request presets"
              icon="pi pi-sliders-h"
              size="small"
              text
              @click="$emit('edit-section', 'launch')"
            />
          </div>
        </dd>
      </div>

      <div class="names-summary__row">
        <dt class="names-summary__term">Gateway routing aliases</dt>
        <dd class="names-summary__detail">
          <p class="names-summary__hint">
            Proxy routing aliases map a client-facing name to one or more targets, including this
            model's names or request presets. They are stored with Proxy settings, not with this
            model; multi-target aliases dispatch round-robin.
          </p>
          <p v-if="routableNames.length" class="names-summary__hint">
            Names the gateway can route here:
            <span class="names-summary__chips names-summary__chips--inline">
              <code v-for="name in routableNames" :key="name" class="names-summary__chip">{{
                name
              }}</code>
            </span>
          </p>
          <p v-else class="names-summary__hint">
            No routable name is published yet. Add a primary API name or start the model to assign
            its stable runtime id.
          </p>
          <div class="names-summary__actions">
            <RouterLink
              class="names-summary__link"
              :to="{ path: '/proxy', query: { tab: 'settings' } }"
              >Open routing settings</RouterLink
            >
          </div>
        </dd>
      </div>
    </dl>
  </PageSection>
</template>

<script setup>
import { computed } from 'vue'
import { RouterLink } from 'vue-router'
import Button from 'primevue/button'

import PageSection from '@/components/common/PageSection.vue'
import { normalizeApiName } from '@/utils/modelPresentation'

const props = defineProps({
  model: { type: Object, required: true },
  config: { type: Object, default: () => ({}) },
  stableModelId: { type: String, default: '' },
})

defineEmits(['edit-section', 'manage-templates'])

const displayName = computed(
  () => props.model?.display_name || props.model?.base_model_name || 'This model',
)
const catalogId = computed(() => String(props.model?.id || '').trim())
const stableRuntimeId = computed(() => String(props.stableModelId || '').trim())

const primarySaved = computed(() => String(props.config?.model_alias || '').trim())
const primaryName = computed(() => normalizeApiName(primarySaved.value))
const primaryHint = computed(() => {
  if (!primaryName.value) {
    return 'Clients use the stable runtime id until a primary API name is saved. Set one in Launch.'
  }
  if (primaryName.value !== primarySaved.value) {
    return `Saved as "${primarySaved.value}"; Studio publishes the lower-case name. Clients send it as the model field.`
  }
  return 'Clients send this as the model field; the gateway resolves it to the stable runtime id.'
})

const ordinaryAliases = computed(() => {
  const raw = Array.isArray(props.config?.swap_aliases) ? props.config.swap_aliases : []
  const seen = new Set()
  const aliases = []
  for (const entry of raw) {
    const name = normalizeApiName(entry)
    if (!name || seen.has(name)) continue
    seen.add(name)
    aliases.push(name)
  }
  return aliases
})

const requestPresets = computed(() => {
  const base = primaryName.value || stableRuntimeId.value
  const rows = Array.isArray(props.config?.set_params_by_id) ? props.config.set_params_by_id : []
  const seen = new Set()
  const presets = []
  for (const row of rows) {
    const suffix = String(row?.sub_id || '').trim()
    const name = suffix ? `${base || 'unset-model'}:${suffix}` : base || 'unset-model'
    if (seen.has(name)) continue
    seen.add(name)
    presets.push({
      key: name,
      name,
      count: Object.keys(row?.params && typeof row.params === 'object' ? row.params : {}).length,
    })
  }
  return presets
})

const presetHint = computed(() => {
  if (!primaryName.value && !stableRuntimeId.value) {
    return 'Add a primary API name, or start the model to assign its stable runtime id, before request presets can be addressed.'
  }
  return 'Each preset exposes the same process under a name:suffix id with request overrides; edit them in Launch → Request parameter aliases.'
})

const routableNames = computed(() => {
  const names = [primaryName.value, stableRuntimeId.value, ...ordinaryAliases.value]
  for (const preset of requestPresets.value) names.push(preset.name)
  return [...new Set(names.filter(Boolean))]
})
</script>

<style scoped>
.names-summary {
  display: grid;
  margin: 0;
}

.names-summary__row {
  display: grid;
  grid-template-columns: minmax(9rem, 0.3fr) minmax(0, 1fr);
  gap: 0.35rem 1.25rem;
  padding: 0.85rem 0;
  border-top: 1px solid var(--border-primary);
}

.names-summary__row:first-child {
  padding-top: 0;
  border-top: 0;
}

.names-summary__row:last-child {
  padding-bottom: 0;
}

.names-summary__term {
  min-width: 0;
  color: var(--text-secondary);
  font-size: var(--text-sm);
  font-weight: 600;
  overflow-wrap: anywhere;
}

.names-summary__detail {
  display: grid;
  gap: 0.35rem;
  min-width: 0;
  margin: 0;
}

.names-summary__value {
  color: var(--text-primary);
  font-weight: 600;
  overflow-wrap: anywhere;
}

.names-summary__value--muted {
  color: var(--text-secondary);
  font-weight: 500;
}

.names-summary__code,
.names-summary__chip {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  overflow-wrap: anywhere;
}

.names-summary__code {
  width: fit-content;
  max-width: 100%;
  padding: 0.1rem 0.4rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  background: var(--bg-surface);
  color: var(--text-primary);
  font-size: var(--text-xs);
}

.names-summary__hint {
  margin: 0;
  max-width: 72ch;
  color: var(--text-secondary);
  font-size: var(--text-xs);
  line-height: 1.5;
}

.names-summary__chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
}

.names-summary__chips--inline {
  display: inline-flex;
  margin-left: 0.15rem;
  vertical-align: middle;
}

.names-summary__chip {
  display: inline-flex;
  align-items: baseline;
  gap: 0.35rem;
  padding: 0.1rem 0.4rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  background: var(--bg-surface);
  color: var(--text-primary);
  font-size: var(--text-xs);
}

.names-summary__chip small {
  color: var(--text-secondary);
  font-family: inherit;
  font-size: var(--text-xs);
}

.names-summary__actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.25rem 0.75rem;
}

.names-summary__actions .p-button {
  padding-left: 0;
  padding-right: 0;
}

.names-summary__link {
  color: var(--accent-primary);
  font-size: var(--text-sm);
  font-weight: 600;
  text-decoration: none;
}

.names-summary__link:hover {
  text-decoration: underline;
}

@media (max-width: 640px), (pointer: coarse) {
  .names-summary__link {
    display: inline-flex;
    align-items: center;
    min-height: 2.75rem;
  }
}

@media (max-width: 640px) {
  .names-summary__row {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>
