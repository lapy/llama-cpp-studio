<template>
  <PageSection
    id="proxy-network"
    title="Network"
    description="Read-only view of how the inference port is bound. Listen host comes from STUDIO_PROXY_LISTEN; terminate TLS at a reverse proxy."
  >
    <dl class="network">
      <div>
        <dt>Listen host</dt>
        <dd>
          <code>{{ network.listen_host || '—' }}</code>
          <small>{{ network.listen_source }}</small>
        </dd>
      </div>
      <div>
        <dt>Port</dt>
        <dd>{{ network.port ?? '—' }}</dd>
      </div>
      <div>
        <dt>Public URL</dt>
        <dd>{{ network.public_url || 'Same origin (default)' }}</dd>
      </div>
      <div>
        <dt>TLS</dt>
        <dd>{{ network.tls_note }}</dd>
      </div>
    </dl>
  </PageSection>
</template>

<script setup>
import PageSection from '@/components/common/PageSection.vue'

defineProps({
  network: { type: Object, default: () => ({}) },
})
</script>

<style scoped>
.network {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 16rem), 1fr));
  gap: 1rem;
  margin: 0;
  font-size: var(--text-sm);
}

.network div {
  display: grid;
  gap: 0.2rem;
}

.network dt {
  color: var(--text-secondary);
}

.network dd {
  margin: 0;
}
</style>
