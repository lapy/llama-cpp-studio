<template>
  <div
    class="install-variant"
    :class="{
      'install-variant--recommended': recommended,
      'install-variant--downloaded': !!state.downloaded,
    }"
  >
    <div>
      <div class="install-variant__title">
        <strong>{{ variant.label || variant.id }}</strong>
        <Tag v-if="recommended" value="Recommended" severity="success" />
        <Tag v-if="state.downloaded" value="Downloaded" severity="success" />
        <Tag
          v-if="catalogInstallMethodLabel(result, variant)"
          :value="catalogInstallMethodLabel(result, variant)"
          :severity="catalogInstallMethodSeverity(variant)"
        />
        <Tag
          v-if="catalogManagerBackendLabel(variant)"
          :value="catalogManagerBackendLabel(variant)"
          severity="secondary"
        />
        <Tag v-if="result.gated || variant.gated" value="Gated HF" severity="warn" />
      </div>
      <span class="install-variant__meta">
        <template v-if="variant.format || variant.precision">
          {{ [variant.format, variant.precision].filter(Boolean).join(' · ') }}
        </template>
        <template v-if="variant.size_bytes">
          <template v-if="variant.format || variant.precision"> · </template>
          <span title="Download size. Runtime memory is not estimated."
            >{{ formatBytes(variant.size_bytes) }} download size</span
          >
        </template>
        <template v-if="variant.files?.length">
          <template v-if="variant.size_bytes || variant.format || variant.precision"> · </template>
          {{ variant.files.length }} file{{ variant.files.length === 1 ? '' : 's' }}
        </template>
      </span>
      <small v-if="catalogInstallMethodHint(result, variant)">{{
        catalogInstallMethodHint(result, variant)
      }}</small>
      <small v-else-if="variant.external_inputs_required"
        >Additional local source input may be required.</small
      >
      <div v-if="hasCompanions" class="install-variant__companions">
        <div v-if="catalogHasProjector(result)" class="install-variant__projector">
          <label :for="`catalog-projector${idSuffix}-${result.id}-${variant.id}`">Projector</label>
          <Select
            :id="`catalog-projector${idSuffix}-${result.id}-${variant.id}`"
            :model-value="state.projector ?? ''"
            :options="catalogProjectorOptions(result)"
            optionLabel="label"
            optionValue="value"
            class="projector-select"
            :disabled="state.busy"
            @update:model-value="emit('set-projector', $event)"
          />
        </div>
        <div v-if="catalogHasMtp(result)" class="install-variant__projector">
          <label :for="`catalog-mtp${idSuffix}-${result.id}-${variant.id}`">MTP draft</label>
          <Select
            :id="`catalog-mtp${idSuffix}-${result.id}-${variant.id}`"
            :model-value="state.mtp ?? ''"
            :options="catalogMtpOptions(result)"
            optionLabel="label"
            optionValue="value"
            class="projector-select"
            :disabled="state.busy"
            @update:model-value="emit('set-mtp', $event)"
          />
        </div>
        <div v-if="catalogHasDflash(result)" class="install-variant__projector">
          <label :for="`catalog-dflash${idSuffix}-${result.id}-${variant.id}`">DFlash draft</label>
          <Select
            :id="`catalog-dflash${idSuffix}-${result.id}-${variant.id}`"
            :model-value="state.dflash ?? ''"
            :options="catalogDflashOptions(result)"
            optionLabel="label"
            optionValue="value"
            class="projector-select"
            :disabled="state.busy"
            @update:model-value="emit('set-dflash', $event)"
          />
        </div>
        <div v-if="showDspark && catalogHasDspark(result)" class="install-variant__projector">
          <label :for="`catalog-dspark${idSuffix}-${result.id}-${variant.id}`">DSpark draft</label>
          <Select
            :id="`catalog-dspark${idSuffix}-${result.id}-${variant.id}`"
            :model-value="state.dspark ?? ''"
            :options="catalogDsparkOptions(result)"
            optionLabel="label"
            optionValue="value"
            class="projector-select"
            :disabled="state.busy"
            @update:model-value="emit('set-dspark', $event)"
          />
        </div>
      </div>
    </div>
    <div class="install-variant__actions">
      <template v-if="state.downloaded">
        <Button
          label="Configure"
          icon="pi pi-cog"
          size="small"
          severity="secondary"
          outlined
          @click="emit('configure')"
        />
        <Button
          label="Refresh"
          icon="pi pi-refresh"
          size="small"
          severity="secondary"
          outlined
          :loading="state.busy"
          :disabled="state.busy"
          @click="emit('refresh')"
        />
        <Button
          v-if="selectionChanged('projector')"
          label="Apply projector"
          icon="pi pi-save"
          size="small"
          severity="success"
          outlined
          :loading="state.busy"
          :disabled="state.busy"
          @click="emit('apply-projector')"
        />
        <Button
          v-if="selectionChanged('mtp')"
          label="Apply MTP"
          icon="pi pi-save"
          size="small"
          severity="success"
          outlined
          :loading="state.busy"
          :disabled="state.busy"
          @click="emit('apply-mtp')"
        />
        <Button
          v-if="selectionChanged('dflash')"
          label="Apply DFlash"
          icon="pi pi-save"
          size="small"
          severity="success"
          outlined
          :loading="state.busy"
          :disabled="state.busy"
          @click="emit('apply-dflash')"
        />
        <Button
          v-if="showDspark && selectionChanged('dspark')"
          label="Apply DSpark"
          icon="pi pi-save"
          size="small"
          severity="success"
          outlined
          :loading="state.busy"
          :disabled="state.busy"
          @click="emit('apply-dspark')"
        />
      </template>
      <Button
        v-else
        :label="catalogVariantActionLabel(result)"
        icon="pi pi-download"
        size="small"
        severity="success"
        outlined
        :disabled="!variant.installable || state.busy"
        :loading="state.busy"
        @click="emit('install')"
      />
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import Button from 'primevue/button'
import Select from 'primevue/select'
import Tag from 'primevue/tag'
import {
  catalogDflashOptions,
  catalogDsparkOptions,
  catalogHasDflash,
  catalogHasDspark,
  catalogHasMtp,
  catalogHasProjector,
  catalogInstallMethodHint,
  catalogInstallMethodLabel,
  catalogInstallMethodSeverity,
  catalogManagerBackendLabel,
  catalogMtpOptions,
  catalogProjectorOptions,
  catalogVariantActionLabel,
  formatBytes,
  isRecommendedCatalogVariant,
} from '@/utils/modelSearchPresentation'

const props = defineProps({
  result: { type: Object, required: true },
  variant: { type: Object, required: true },
  /** Resolved per-variant state owned by the view: downloaded, busy and companion selections. */
  state: { type: Object, default: () => ({}) },
  /** Picker rows keep their historical `catalog-<x>-modal-…` element ids. */
  idSuffix: { type: String, default: '' },
  showDspark: { type: Boolean, default: true },
})

const emit = defineEmits([
  'install',
  'configure',
  'refresh',
  'apply-projector',
  'apply-mtp',
  'apply-dflash',
  'apply-dspark',
  'set-projector',
  'set-mtp',
  'set-dflash',
  'set-dspark',
])

const recommended = computed(() => isRecommendedCatalogVariant(props.variant))

const hasCompanions = computed(
  () =>
    catalogHasProjector(props.result) ||
    catalogHasMtp(props.result) ||
    catalogHasDflash(props.result) ||
    (props.showDspark && catalogHasDspark(props.result)),
)

const COMPANION_AVAILABLE = {
  projector: catalogHasProjector,
  mtp: catalogHasMtp,
  dflash: catalogHasDflash,
  dspark: catalogHasDspark,
}

const INSTALLED_FIELD = {
  projector: 'mmproj_filename',
  mtp: 'mtp_filename',
  dflash: 'dflash_filename',
  dspark: 'dspark_filename',
}

function selectionChanged(kind) {
  if (!props.state.downloaded || !COMPANION_AVAILABLE[kind](props.result)) return false
  return (props.state[kind] || '') !== (props.state.downloaded[INSTALLED_FIELD[kind]] || '')
}
</script>

<style scoped>
.install-variant {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  padding: 0.65rem 0.75rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  background: var(--bg-surface);
}

.install-variant > div {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  min-width: 0;
}

.install-variant--recommended {
  border-color: rgba(16, 185, 129, 0.45);
}

.install-variant--downloaded {
  border-color: rgba(59, 130, 246, 0.35);
}

.install-variant__title {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  flex-wrap: wrap;
}

.install-variant__meta,
.install-variant small {
  color: var(--text-secondary);
  font-size: var(--text-xs);
}

.install-variant__companions {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  margin-top: 0.35rem;
}

.install-variant__actions {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 0.35rem;
  flex-shrink: 0;
}

.install-variant__projector {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 0;
}

.install-variant__projector label {
  font-size: var(--text-xs);
  color: var(--text-secondary);
  white-space: nowrap;
}

.projector-select {
  min-width: 8rem;
}

@media (max-width: 768px) {
  .install-variant {
    flex-direction: column;
    align-items: stretch;
    gap: 0.65rem;
  }
}

@container (max-width: 28rem) {
  .install-variant {
    flex-direction: column;
    align-items: stretch;
    gap: 0.65rem;
  }
}
</style>
