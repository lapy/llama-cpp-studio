<template>
  <PageSection
    class="first-model"
    title="Add your first model"
    description="Models come from Hugging Face as GGUF files. Pick a starter below, choose a quantization in Search, then start it from this library to get an endpoint."
  >
    <template #actions>
      <RouterLink class="page-action-link first-model__primary" to="/search">
        <i class="pi pi-search" aria-hidden="true" />
        Search models
      </RouterLink>
      <RouterLink class="page-action-link" to="/engines">Set up an engine</RouterLink>
    </template>

    <div class="first-model__starters">
      <h3 id="first-model-starters-title" class="page-subheading">Starter models</h3>
      <ul class="page-card-grid first-model__list" aria-labelledby="first-model-starters-title">
        <li v-for="model in STARTER_MODELS" :key="model.repo">
          <RouterLink class="first-model__card" :to="starterModelSearchPath(model.repo)">
            <span class="first-model__name">{{ model.name }}</span>
            <code class="first-model__repo">{{ model.repo }}</code>
            <span class="first-model__why">{{ model.why }}</span>
            <span class="first-model__foot">
              <span class="first-model__size">{{ model.size }}</span>
              <span class="first-model__cta">Find in Search</span>
            </span>
          </RouterLink>
        </li>
      </ul>
    </div>
  </PageSection>
</template>

<script setup>
import { RouterLink } from 'vue-router'
import PageSection from '@/components/common/PageSection.vue'
import { STARTER_MODELS, starterModelSearchPath } from '@/utils/starterModels'
</script>

<style scoped>
.first-model__primary {
  border-color: var(--accent-primary);
  color: var(--accent-primary);
  font-weight: 650;
}

.first-model__primary:hover {
  color: var(--accent-primary);
  background: var(--accent-cyan-soft);
  box-shadow: none;
}

.first-model__list {
  margin: 0;
  padding: 0;
  list-style: none;
}

.first-model__card {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  height: 100%;
  padding: 0.75rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  background: var(--bg-secondary);
  color: var(--text-primary);
  text-decoration: none;
}

.first-model__card:hover {
  border-color: var(--accent-primary);
  background: var(--bg-card-hover);
  text-decoration: none;
}

.first-model__card:focus-visible {
  outline: 2px solid var(--accent-primary);
  outline-offset: 3px;
}

.first-model__name {
  font-size: var(--text-sm);
  font-weight: 650;
}

.first-model__repo {
  overflow: hidden;
  color: var(--text-secondary);
  font-size: var(--text-xs);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.first-model__why {
  flex: 1 1 auto;
  color: var(--text-secondary);
  font-size: var(--text-xs);
  line-height: 1.5;
}

.first-model__foot {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  justify-content: space-between;
  gap: 0.25rem 0.5rem;
  margin-top: 0.25rem;
}

.first-model__size {
  color: var(--text-secondary);
  font-size: var(--text-xs);
}

.first-model__cta {
  color: var(--accent-primary);
  font-size: var(--text-xs);
  font-weight: 650;
}

@media (max-width: 480px) {
  .first-model__repo {
    white-space: normal;
    overflow-wrap: anywhere;
  }
}
</style>
