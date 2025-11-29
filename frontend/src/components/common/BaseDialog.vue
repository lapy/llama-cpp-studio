<template>
  <Dialog
    :visible="visible"
    :header="header"
    :modal="modal"
    :style="dialogStyle"
    :draggable="draggable"
    :resizable="resizable"
    :closable="closable"
    :class="dialogClass"
    @update:visible="$emit('update:visible', $event)"
    @hide="$emit('hide')"
  >
    <template v-if="$slots.header" #header>
      <slot name="header"></slot>
    </template>
    
    <slot></slot>
    
    <template v-if="$slots.footer" #footer>
      <slot name="footer"></slot>
    </template>
  </Dialog>
</template>

<script setup>
import Dialog from 'primevue/dialog'

defineProps({
  visible: {
    type: Boolean,
    default: false
  },
  header: {
    type: String,
    default: ''
  },
  modal: {
    type: Boolean,
    default: true
  },
  dialogStyle: {
    type: Object,
    default: () => ({ width: '50vw', maxWidth: '600px' })
  },
  draggable: {
    type: Boolean,
    default: false
  },
  resizable: {
    type: Boolean,
    default: false
  },
  closable: {
    type: Boolean,
    default: true
  },
  dialogClass: {
    type: String,
    default: ''
  }
})

defineEmits(['update:visible', 'hide'])
</script>

<style scoped>
/* Dialog styling handled by global PrimeVue overrides */
</style>

