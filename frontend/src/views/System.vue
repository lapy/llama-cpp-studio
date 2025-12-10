<template>
  <div class="system-page">
    <TabView>
      <TabPanel header="System">
        <SystemTab />
      </TabPanel>
      <TabPanel header="llama.cpp">
        <LlamaCppTab />
      </TabPanel>
      <TabPanel header="LMDeploy">
        <LMDeployTab />
      </TabPanel>
    </TabView>
  </div>
</template>

<script setup>
import TabView from 'primevue/tabview'
import TabPanel from 'primevue/tabpanel'
import SystemTab from '@/components/system/SystemTab.vue'
import LlamaCppTab from '@/components/system/LlamaCppTab.vue'
import LMDeployTab from '@/components/system/LMDeployTab.vue'
</script>

<style scoped>
.system-page {
  max-width: 1400px;
  margin: 0 auto;
  padding: var(--spacing-lg);
}

:deep(.p-tabview-nav) {
  background: var(--bg-surface);
  border-radius: var(--radius-lg) var(--radius-lg) 0 0;
  padding: var(--spacing-sm) var(--spacing-sm) 0;
}

:deep(.p-tabview-nav-link) {
  padding: var(--spacing-md) var(--spacing-lg);
  font-weight: 600;
  transition: all var(--transition-fast);
}

:deep(.p-tabview-nav-link:hover) {
  background: var(--bg-card);
}

:deep(.p-tabview-nav-link.p-highlight) {
  background: var(--gradient-primary);
  color: white;
  border-radius: var(--radius-md) var(--radius-md) 0 0;
}

:deep(.p-tabview-panels) {
  background: var(--bg-card);
  border: 1px solid var(--border-primary);
  border-top: none;
  border-radius: 0 0 var(--radius-lg) var(--radius-lg);
  padding: var(--spacing-xl);
}
</style>




