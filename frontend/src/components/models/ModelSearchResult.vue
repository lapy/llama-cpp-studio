<template>
  <article class="catalog-card">
    <div class="catalog-card__head">
      <div>
        <a v-if="sourceUrl" :href="sourceUrl" target="_blank" class="model-link">
          {{ result.display_name }}
        </a>
        <span v-else class="model-link">{{ result.display_name }}</span>
        <div class="catalog-card__id">{{ catalogCardSubtitle(result) }}</div>
        <div v-if="meta.length" class="catalog-card__meta">
          <span v-for="item in meta" :key="item.key" class="meta-item">
            <i :class="item.icon" /> {{ item.label }}
          </span>
        </div>
      </div>
      <div class="result-name__tags">
        <Tag
          :value="
            isAudioEngineId(result.provider) ? engineLabel(AUDIO_ENGINE_IDS[0]) : 'Hugging Face'
          "
          severity="info"
        />
        <Tag
          v-if="catalogFormatLabel(result)"
          :value="catalogFormatLabel(result)"
          severity="secondary"
        />
        <Tag v-if="result.gated" value="Gated" severity="warn" />
        <Tag
          v-if="result.metadata?.access_status === 'unknown'"
          value="Access requirements unknown"
          severity="secondary"
        />
        <Tag
          v-if="(result.features || []).includes('multimodal')"
          value="Vision"
          severity="success"
        />
        <Tag
          v-if="
            (result.features || []).includes('mtp') ||
            (result.metadata?.raw?.mtp_files || []).length
          "
          value="MTP"
          severity="info"
        />
        <Tag
          v-if="
            (result.features || []).includes('dflash') ||
            (result.metadata?.raw?.dflash_files || []).length
          "
          value="DFlash"
          severity="info"
        />
        <Tag
          v-if="
            (result.features || []).includes('dspark') ||
            (result.metadata?.raw?.dspark_files || []).length
          "
          value="DSpark"
          severity="info"
        />
        <Tag
          v-if="
            (result.features || []).includes('engram') ||
            (result.metadata?.raw?.engram_files || []).length
          "
          value="Engram"
          severity="info"
        />
      </div>
    </div>

    <p v-if="result.description" class="catalog-card__description">
      {{ result.description }}
    </p>

    <div v-if="badges.length" class="catalog-badges">
      <Tag
        v-for="badge in badges"
        :key="badge.key"
        :value="badge.label"
        :severity="badge.severity"
        v-tooltip.bottom="badge.tooltip || ''"
      />
    </div>

    <div v-if="result.unavailable_reason" class="catalog-unavailable">
      <i class="pi pi-exclamation-triangle" />
      {{ result.unavailable_reason }}
    </div>
    <div v-else-if="compatibleEngines.length" class="compatibility-evidence">
      <i class="pi pi-verified" />
      Compatible with
      {{ compatibleEngines.map(engineLabel).join(' · ') }}
    </div>

    <div v-if="result.gated && !hasToken" class="catalog-gated-cta">
      <span>This model is gated and needs a Hugging Face token.</span>
      <div class="catalog-gated-cta__actions">
        <Button
          label="Set token"
          icon="pi pi-key"
          size="small"
          severity="warning"
          outlined
          @click="emit('set-token')"
        />
        <Button
          v-if="sourceUrl"
          label="Open on HF"
          icon="pi pi-external-link"
          size="small"
          severity="secondary"
          text
          @click="emit('open-external', sourceUrl)"
        />
      </div>
    </div>

    <p v-if="compatibleEngines.length && !ready" class="catalog-fit-note">
      No compatible engine is ready yet.
      <RouterLink to="/engines">Set up an engine</RouterLink> before starting this model.
    </p>
    <p v-if="hasGgufVariant" class="catalog-fit-note">
      Choosing a quantization: higher precision usually preserves more quality and uses more memory.
      Smaller quantizations use less memory.
    </p>
    <div class="install-variants">
      <template v-if="variants.length <= 1">
        <ModelSearchVariant
          v-for="variant in variants"
          :key="variant.id"
          :result="result"
          :variant="variant"
          :state="variantState[variant.id] || {}"
          @install="emit('install', variant)"
          @configure="emit('configure', variant)"
          @refresh="emit('refresh', variant)"
          @apply-projector="emit('apply-projector', variant)"
          @apply-mtp="emit('apply-mtp', variant)"
          @apply-dflash="emit('apply-dflash', variant)"
          @apply-dspark="emit('apply-dspark', variant)"
          @set-projector="emit('set-projector', { variant, value: $event })"
          @set-mtp="emit('set-mtp', { variant, value: $event })"
          @set-dflash="emit('set-dflash', { variant, value: $event })"
          @set-dspark="emit('set-dspark', { variant, value: $event })"
        />
      </template>
      <div v-else class="install-variant install-variant--summary">
        <div>
          <strong>{{ summaryTitle }}</strong>
          <span class="install-variant__meta">{{ summaryMeta }}</span>
          <span v-if="downloadedCount" class="install-variant__owned">
            {{ downloadedCount }} already in library
          </span>
        </div>
        <Button
          :label="catalogVariantsBrowseLabel(result)"
          icon="pi pi-list"
          size="small"
          severity="success"
          @click="emit('browse')"
        />
      </div>
    </div>
  </article>
</template>

<script setup>
import { computed } from 'vue'
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import ModelSearchVariant from '@/components/models/ModelSearchVariant.vue'
import { AUDIO_ENGINE_IDS, isAudioEngineId } from '@/engines/registry'
import {
  catalogCardMeta,
  catalogCardSubtitle,
  catalogFormatLabel,
  catalogPrimaryBadges,
  catalogSourceUrl,
  catalogVariantsBrowseLabel,
  catalogVariantsSummaryMeta,
  catalogVariantsSummaryTitle,
} from '@/utils/modelSearchPresentation'

const props = defineProps({
  result: { type: Object, required: true },
  /** The view resolved at least one runnable engine for this result. */
  ready: Boolean,
  /** The view holds a Hugging Face token. */
  hasToken: Boolean,
  /** Engine-id → display label, owned by the view (server descriptors first). */
  engineLabel: { type: Function, required: true },
  /** Per-variant `{ downloaded, busy, projector, mtp, dflash, dspark }`, owned by the view. */
  variantState: { type: Object, default: () => ({}) },
})

const emit = defineEmits([
  'install',
  'configure',
  'refresh',
  'browse',
  'set-token',
  'open-external',
  'apply-projector',
  'apply-mtp',
  'apply-dflash',
  'apply-dspark',
  'set-projector',
  'set-mtp',
  'set-dflash',
  'set-dspark',
])

const variants = computed(() => props.result.install_variants || [])
const sourceUrl = computed(() => catalogSourceUrl(props.result))
const meta = computed(() => catalogCardMeta(props.result))
const badges = computed(() => catalogPrimaryBadges(props.result))
const compatibleEngines = computed(() => props.result.compatible_engines || [])
const hasGgufVariant = computed(() => variants.value.some((variant) => variant.format === 'gguf'))
const summaryTitle = computed(() => catalogVariantsSummaryTitle(props.result))
const summaryMeta = computed(() => catalogVariantsSummaryMeta(props.result))
const downloadedCount = computed(
  () => Object.values(props.variantState).filter((state) => state?.downloaded).length,
)
</script>

<style scoped>
.catalog-card {
  padding: 1rem;
  background: var(--bg-card, #161b2e);
  border: 1px solid var(--border-primary, #2a2f45);
  border-radius: var(--radius-lg, 0.75rem);
  container-type: inline-size;
  min-width: 0;
}

.catalog-card__head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
}

.catalog-card__id {
  margin-top: 0.2rem;
  color: var(--text-secondary);
  font-size: var(--text-xs);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

.catalog-card__meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin-top: 0.4rem;
  color: var(--text-secondary);
  font-size: var(--text-xs);
}

.catalog-card__meta .meta-item {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
}

.meta-item {
  font-size: var(--text-xs);
  color: var(--text-secondary, #9ca3af);
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.meta-item .pi {
  font-size: var(--text-xs);
}

.result-name__tags {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  flex-shrink: 0;
  margin-left: auto;
}

.model-link {
  font-weight: 600;
  font-size: 0.9rem;
  color: var(--text-primary, #f1f5f9);
  text-decoration: none;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  min-width: 0;
  align-self: flex-start;
}

.model-link:hover {
  color: var(--accent-cyan, #22d3ee);
  text-decoration: underline;
}

.catalog-card__description {
  margin: 0.75rem 0;
  color: var(--text-secondary);
  font-size: 0.875rem;
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.catalog-badges {
  display: flex;
  gap: 0.35rem;
  flex-wrap: wrap;
}

.catalog-gated-cta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  margin-top: 0.75rem;
  padding: 0.65rem 0.75rem;
  border-radius: var(--radius-md);
  border: 1px solid rgba(245, 158, 11, 0.3);
  background: rgba(245, 158, 11, 0.06);
  color: var(--text-secondary);
  font-size: 0.8rem;
}

.catalog-gated-cta__actions {
  display: flex;
  gap: 0.35rem;
  flex-wrap: wrap;
}

.catalog-unavailable,
.compatibility-evidence {
  display: flex;
  gap: 0.45rem;
  align-items: center;
  margin-top: 0.75rem;
  font-size: 0.8rem;
}

.catalog-unavailable {
  color: var(--accent-yellow, #f59e0b);
}

.compatibility-evidence {
  color: var(--accent-green, #10b981);
}

.catalog-fit-note {
  margin: 0;
  padding: 0.75rem;
  border-radius: var(--radius-sm);
  background: var(--bg-surface);
  font-size: var(--text-xs);
  line-height: 1.6;
}

.install-variants {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin-top: 0.85rem;
}

/* The summary row reuses the variant shell; keep the selector on the modifier so
   these declarations cannot override the child rows' recommended/downloaded
   borders via the parent scope id on their root element. */
.install-variant--summary {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  padding: 0.65rem 0.75rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  background: var(--bg-elevated, var(--bg-surface));
}

.install-variant--summary > div {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  min-width: 0;
}

.install-variant__meta {
  color: var(--text-secondary);
  font-size: var(--text-xs);
}

.install-variant__owned {
  display: block;
  margin-top: 0.15rem;
  color: var(--accent-green, #10b981);
  font-size: var(--text-xs);
}

@media (max-width: 768px) {
  .catalog-card__head {
    flex-direction: column;
    align-items: stretch;
    gap: 0.65rem;
  }

  .install-variant--summary {
    flex-direction: column;
    align-items: stretch;
    gap: 0.65rem;
  }
}

@container (max-width: 28rem) {
  .catalog-card__head {
    flex-direction: column;
    align-items: stretch;
    gap: 0.65rem;
  }

  .install-variant--summary {
    flex-direction: column;
    align-items: stretch;
    gap: 0.65rem;
  }

  .model-link {
    word-break: break-word;
    overflow-wrap: anywhere;
  }
}
</style>
