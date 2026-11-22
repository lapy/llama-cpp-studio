import { defineAsyncComponent, defineComponent, h } from 'vue'

// Same-height placeholder while the cold-path filter toolbar chunk loads.
const LibraryToolbarSkeleton = defineComponent({
  name: 'LibraryToolbarSkeleton',
  render() {
    return h('div', { class: 'page-toolbar page-toolbar--plain', 'aria-hidden': 'true' }, [
      h('span', { class: 'page-toolbar__summary' }, 'Loading library filters…'),
    ])
  },
})

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
export const HuggingFaceTokenDialog = defineAsyncComponent(
  () => import('@/components/models/HuggingFaceTokenDialog.vue'),
)
// The gated-model notice, the filter toolbar and the list layout are cold-path
// UI. Keep them out of the initial library graph; each is a small chunk loaded
// on demand (the toolbar has a same-size fallback in the page).
export const HuggingFaceTokenNotice = defineAsyncComponent(
  () => import('@/components/models/HuggingFaceTokenNotice.vue'),
)
export const LibraryToolbar = defineAsyncComponent({
  loader: () => import('@/components/models/LibraryToolbar.vue'),
  loadingComponent: LibraryToolbarSkeleton,
  delay: 0,
})
export const LibraryListView = defineAsyncComponent(
  () => import('@/components/models/LibraryListView.vue'),
)
