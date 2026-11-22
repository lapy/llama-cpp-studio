<template>
  <Dialog
    :visible="visible"
    modal
    :header="header"
    class="dialog-width-md catalog-variant-picker"
    @update:visible="$emit('update:visible', $event)"
  >
    <div class="dialog-stack">
      <PageToolbar
        v-if="(result?.install_variants || []).length > 8"
        plain
        label="Variant filters"
        class="variant-picker-toolbar"
      >
        <template #search>
          <InputText
            v-model="filter"
            placeholder="Filter quantizations…"
            aria-label="Filter quantizations"
            class="w-full"
          />
        </template>
        <label class="variant-picker-popular">
          <input v-model="popularOnly" type="checkbox" />
          Popular only
        </label>
      </PageToolbar>

      <div class="install-variants install-variants--picker">
        <ModelSearchVariant
          v-for="variant in filteredVariants"
          :key="variant.id"
          :result="result"
          :variant="variant"
          :state="variantState[variant.id] || {}"
          id-suffix="-modal"
          :show-dspark="false"
          @install="$emit('install', variant)"
          @configure="$emit('configure', variant)"
          @refresh="$emit('refresh', variant)"
          @apply-projector="$emit('apply-projector', variant)"
          @apply-mtp="$emit('apply-mtp', variant)"
          @apply-dflash="$emit('apply-dflash', variant)"
          @set-projector="$emit('set-projector', { variant, value: $event })"
          @set-mtp="$emit('set-mtp', { variant, value: $event })"
          @set-dflash="$emit('set-dflash', { variant, value: $event })"
        />
        <p v-if="!filteredVariants.length" class="variant-picker-empty">
          No quantizations match this filter.
        </p>
      </div>
    </div>
    <template #footer>
      <div class="dialog-actions">
        <Button label="Close" severity="secondary" text @click="$emit('update:visible', false)" />
      </div>
    </template>
  </Dialog>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import Dialog from 'primevue/dialog'
import Button from 'primevue/button'
import InputText from 'primevue/inputtext'
import PageToolbar from '@/components/common/PageToolbar.vue'
import ModelSearchVariant from '@/components/models/ModelSearchVariant.vue'
import {
  isRecommendedCatalogVariant,
  recommendedCatalogVariantScore,
} from '@/utils/modelSearchPresentation'

const props = defineProps({
  visible: { type: Boolean, default: false },
  result: { type: Object, default: null },
  /** Per-variant `{ downloaded, busy, projector, mtp, dflash, dspark }`. */
  variantState: { type: Object, default: () => ({}) },
})

defineEmits([
  'update:visible',
  'install',
  'configure',
  'refresh',
  'apply-projector',
  'apply-mtp',
  'apply-dflash',
  'set-projector',
  'set-mtp',
  'set-dflash',
])

const filter = ref('')
const popularOnly = ref(false)

watch(
  () => props.visible,
  (open) => {
    if (!open) {
      filter.value = ''
      popularOnly.value = false
    }
  },
)

const header = computed(() => {
  const result = props.result
  if (!result) return 'Select variant'
  const name = result.display_name || result.provider_item_id || 'model'
  return result.artifact_format === 'gguf' ? `Quantizations — ${name}` : `Install options — ${name}`
})

const filteredVariants = computed(() => {
  let list = [...(props.result?.install_variants || [])]
  if (popularOnly.value) {
    const popular = list.filter(isRecommendedCatalogVariant)
    if (popular.length) list = popular
  }
  const needle = filter.value.trim().toLowerCase()
  if (needle) {
    list = list.filter((variant) =>
      String(variant.label || variant.id || '')
        .toLowerCase()
        .includes(needle),
    )
  }
  list.sort((a, b) => {
    const scoreDelta = recommendedCatalogVariantScore(a) - recommendedCatalogVariantScore(b)
    if (scoreDelta !== 0) return scoreDelta
    return (a.size_bytes || 0) - (b.size_bytes || 0)
  })
  return list
})
</script>

<style scoped>
.variant-picker-toolbar {
  align-items: flex-end;
}

.variant-picker-popular {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  color: var(--text-secondary);
  font-size: var(--text-sm);
  white-space: nowrap;
}

.install-variants {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.install-variants--picker {
  max-height: min(60vh, 28rem);
  overflow-y: auto;
  padding-right: 0.15rem;
}

.variant-picker-empty {
  margin: 0.5rem 0 0;
  color: var(--text-secondary);
  font-size: var(--text-sm);
}
</style>
