<template>
  <section class="first-model" aria-labelledby="first-model-title">
    <div class="first-model__intro">
      <p class="first-model__eyebrow">New here?</p>
      <h2 id="first-model-title">Add your first model</h2>
      <p class="first-model__lede">
        Models come from Hugging Face as GGUF files. Pick a starter below, choose a quantization in
        Search, then start it from this library to get an endpoint.
      </p>
      <RouterLink class="first-model__primary" to="/search">
        <i class="pi pi-search" aria-hidden="true" />
        Search models
      </RouterLink>
    </div>

    <div class="first-model__starters">
      <h3 id="first-model-starters-title">Starter models</h3>
      <ul class="first-model__list" aria-labelledby="first-model-starters-title">
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
  </section>
</template>

<script setup>
import { RouterLink } from 'vue-router'
import { STARTER_MODELS, starterModelSearchPath } from '@/utils/starterModels'
</script>

<style scoped>
.first-model {
  display: grid;
  gap: 1.25rem;
  padding: 1.25rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  background: var(--bg-card);
}

.first-model__intro {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 0.5rem;
}

.first-model__eyebrow {
  margin: 0;
  color: var(--accent-primary);
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.first-model h2 {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 650;
}

.first-model__lede {
  max-width: 52ch;
  margin: 0;
  color: var(--text-secondary);
  font-size: 0.875rem;
  line-height: 1.6;
}

.first-model__primary {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  min-height: 2.75rem;
  margin-top: 0.25rem;
  padding: 0.5rem 1rem;
  border-radius: var(--radius-md);
  background: var(--gradient-primary);
  color: var(--bg-primary);
  font-size: 0.875rem;
  font-weight: 650;
  text-decoration: none;
}

.first-model__primary:hover {
  box-shadow: var(--glow-primary);
}

[data-theme='light'] .first-model__primary {
  color: #fff;
}

.first-model__starters h3 {
  margin: 0 0 0.5rem;
  font-size: 0.875rem;
  font-weight: 650;
}

.first-model__list {
  display: grid;
  grid-template-columns: 1fr;
  gap: 0.5rem;
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
}

.first-model__card:focus-visible {
  outline: 2px solid var(--accent-primary);
  outline-offset: 3px;
}

.first-model__name {
  font-size: 0.875rem;
  font-weight: 650;
}

.first-model__repo {
  overflow: hidden;
  color: var(--text-secondary);
  font-size: 0.75rem;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.first-model__why {
  flex: 1 1 auto;
  color: var(--text-secondary);
  font-size: 0.8125rem;
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
  font-size: 0.75rem;
}

.first-model__cta {
  color: var(--accent-primary);
  font-size: 0.75rem;
  font-weight: 650;
}

@media (min-width: 40rem) {
  .first-model__list {
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 16rem), 1fr));
  }
}

@media (min-width: 60rem) {
  .first-model {
    grid-template-columns: minmax(0, 1fr) minmax(0, 2fr);
    gap: 1.5rem;
    align-items: start;
  }

  .first-model__intro {
    position: sticky;
    top: calc(var(--app-header-height) + 1rem);
  }
}

@media (max-width: 30rem) {
  .first-model {
    padding: 1rem;
  }
}
</style>
