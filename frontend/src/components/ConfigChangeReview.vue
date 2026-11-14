<template>
  <Dialog
    :visible="visible"
    header="Review changes"
    modal
    class="dialog-width-lg config-change-review"
    @update:visible="$emit('update:visible', $event)"
  >
    <div class="change-review__intro">
      <div>
        <p class="change-review__lead">
          These values will be written to this model’s saved configuration.
        </p>
        <p class="change-review__hint">
          {{ changeCount }} {{ changeCount === 1 ? 'change' : 'changes' }} · nothing is applied to a
          running engine until you use Apply afterwards.
        </p>
      </div>
      <Tag
        :value="`${changeCount} ${changeCount === 1 ? 'change' : 'changes'}`"
        :severity="changeCount ? 'info' : 'secondary'"
      />
    </div>

    <Message v-if="invalid" severity="warn" :closable="false" class="change-review__invalid">
      Fix the alias validation errors before saving.
    </Message>

    <div v-if="!groups.length && !invalid" class="change-review__empty">
      <i class="pi pi-check-circle" aria-hidden="true" />
      <div>
        <strong>No changes to save</strong>
        <p>The form already matches the saved configuration.</p>
      </div>
    </div>

    <section v-for="group in groups" :key="group.id" class="change-review__group">
      <h3>{{ group.label }}</h3>
      <ul class="change-review__list">
        <li v-for="row in group.rows" :key="`${group.id}:${row.key}`" class="change-review__row">
          <div class="change-review__row-head">
            <span class="change-review__name">{{ row.label }}</span>
            <Tag :value="kindLabel(row.kind)" :severity="kindSeverity(row.kind)" />
          </div>
          <div class="change-review__values">
            <span class="change-review__before" :title="row.before">{{ row.before }}</span>
            <i class="pi pi-arrow-right change-review__arrow" aria-hidden="true" />
            <strong class="change-review__after" :title="row.after">{{ row.after }}</strong>
          </div>
        </li>
      </ul>
    </section>

    <template #footer>
      <Button
        label="Keep editing"
        severity="secondary"
        outlined
        @click="$emit('update:visible', false)"
      />
      <Button
        v-if="groups.length"
        label="Discard changes"
        severity="danger"
        text
        :disabled="saving"
        @click="$emit('discard')"
      />
      <Button
        label="Save changes"
        icon="pi pi-save"
        severity="success"
        :loading="saving"
        :disabled="saving || invalid || !groups.length"
        @click="$emit('save')"
      />
    </template>
  </Dialog>
</template>

<script setup>
import { computed } from 'vue'
import Dialog from 'primevue/dialog'
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import Message from 'primevue/message'

const props = defineProps({
  visible: { type: Boolean, default: false },
  groups: { type: Array, default: () => [] },
  saving: { type: Boolean, default: false },
  invalid: { type: Boolean, default: false },
})
defineEmits(['update:visible', 'save', 'discard'])

const changeCount = computed(() =>
  props.groups.reduce((total, group) => total + (group.rows?.length || 0), 0),
)

function kindLabel(kind) {
  if (kind === 'added') return 'Added'
  if (kind === 'removed') return 'Removed'
  return 'Changed'
}

function kindSeverity(kind) {
  if (kind === 'added') return 'success'
  if (kind === 'removed') return 'danger'
  return 'info'
}
</script>

<style scoped>
.change-review__intro {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1rem;
}
.change-review__lead {
  margin: 0;
  font-weight: 600;
}
.change-review__hint {
  margin: 0.3rem 0 0;
  color: var(--text-secondary);
  font-size: var(--text-xs);
}
.change-review__invalid {
  margin-bottom: 1rem;
}
.change-review__empty {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  padding: 1rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  color: var(--text-secondary);
}
.change-review__empty i {
  color: var(--status-success);
  font-size: 1.15rem;
  margin-top: 0.1rem;
}
.change-review__empty p {
  margin: 0.25rem 0 0;
  font-size: var(--text-xs);
}
.change-review__group + .change-review__group {
  margin-top: 1.1rem;
}
.change-review__group h3 {
  margin: 0 0 0.45rem;
  font-size: var(--text-sm);
}
.change-review__list {
  display: grid;
  gap: 0.4rem;
  margin: 0;
  padding: 0;
  list-style: none;
}
.change-review__row {
  display: grid;
  gap: 0.35rem;
  padding: 0.6rem 0.7rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  background: var(--bg-card);
}
.change-review__row-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
}
.change-review__name {
  font-weight: 600;
  overflow-wrap: anywhere;
}
.change-review__values {
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 0.45rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 0.78rem;
}
.change-review__before {
  max-width: 100%;
  overflow-wrap: anywhere;
  color: var(--text-secondary);
  text-decoration: line-through;
  text-decoration-color: color-mix(in srgb, var(--text-secondary) 60%, transparent);
}
.change-review__arrow {
  color: var(--text-secondary);
  font-size: 0.7rem;
}
.change-review__after {
  overflow-wrap: anywhere;
  color: var(--text-primary);
}
</style>
