<template>
  <div class="page-shell">
    <PageHeader
      title="Page not found"
      description="Studio could not match this address to a workspace page. Check the URL, or return to the model library."
    />
    <EmptyState
      icon="pi pi-compass"
      title="This address does not match a Studio workspace."
      description="Check the address for typos, or start again from the model library."
    >
      <RouterLink to="/models" class="page-action-link">
        <i class="pi pi-arrow-left" aria-hidden="true" />
        Return to Models
      </RouterLink>
    </EmptyState>
  </div>
</template>

<script setup>
import { RouterLink } from 'vue-router'
import EmptyState from '@/components/common/EmptyState.vue'
import PageHeader from '@/components/common/PageHeader.vue'
</script>
