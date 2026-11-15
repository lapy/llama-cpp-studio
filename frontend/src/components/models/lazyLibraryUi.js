import { defineAsyncComponent } from 'vue'

// Cold /models graph budget: first-run guidance and the click-only connect
// dialog do not need to ship before the library itself is usable.
export const SetupChecklist = defineAsyncComponent(
  () => import('@/components/common/SetupChecklist.vue'),
)
export const FirstModelPanel = defineAsyncComponent(
  () => import('@/components/models/FirstModelPanel.vue'),
)
export const ConnectDialog = defineAsyncComponent(
  () => import('@/components/common/ConnectDialog.vue'),
)
