<template>
  <section class="proxy-card network-panel" aria-labelledby="proxy-network-heading">
    <div class="section-head">
      <div>
        <h2 id="proxy-network-heading">Network</h2>
        <p>
          Read-only view of how the inference port is bound. Listen host comes from
          STUDIO_PROXY_LISTEN; terminate TLS at a reverse proxy.
        </p>
      </div>
    </div>
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
  </section>
</template>

<script setup>
defineProps({
  network: { type: Object, default: () => ({}) },
})
</script>

<style scoped>
.network-panel {
  display: grid;
  gap: 1rem;
  min-width: 0;
  border: 1px solid var(--border-primary);
  background: var(--bg-card);
}

.section-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
}

.section-head h2 {
  margin: 0;
  font-size: 1.05rem;
}

.section-head p {
  color: var(--text-secondary);
  margin: 0.3rem 0 0;
}

.network {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 16rem), 1fr));
  gap: 1rem;
  margin: 0;
  font-size: 0.85rem;
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
