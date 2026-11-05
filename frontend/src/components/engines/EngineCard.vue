<template>
  <button type="button" class="engine-card" @click="$emit('open')">
    <div class="engine-card-head">
      <div class="engine-card-title">
        <slot name="mark" />
        <div>
          <div class="engine-card-name">{{ title }}</div>
          <div class="engine-card-meta">{{ count }} version{{ count === 1 ? '' : 's' }}</div>
        </div>
        <slot name="badge" />
      </div>
    </div>
    <div class="engine-card-body"><slot /></div>
    <span class="engine-card-cta">{{ cta }}</span>
  </button>
</template>
<script setup>
defineProps({
  title: { type: String, required: true },
  count: { type: Number, default: 0 },
  cta: String,
})
defineEmits(['open'])
</script>
