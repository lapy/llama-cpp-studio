<template>
  <Dialog
    :visible="visible"
    header="Unsaved configuration"
    modal
    :closable="false"
    class="dialog-width-sm"
  >
    <p>You have edits that are not saved. Save them before leaving, or discard them.</p>
    <template #footer>
      <Button label="Stay" severity="secondary" outlined @click="$emit('stay')" />
      <Button label="Discard" severity="danger" outlined @click="$emit('discard')" />
      <Button label="Save" icon="pi pi-save" :loading="saving" @click="$emit('save')" />
    </template>
  </Dialog>
</template>
<script setup>
import Dialog from 'primevue/dialog'
import Button from 'primevue/button'
defineProps({ visible: Boolean, saving: Boolean })
defineEmits(['stay', 'discard', 'save'])
</script>
