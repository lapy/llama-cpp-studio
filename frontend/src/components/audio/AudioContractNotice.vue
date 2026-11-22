<template>
  <StateBanner
    v-if="failureKind === 'schema-mismatch'"
    severity="error"
    class="config-scan-message"
  >
    <strong>audio.cpp contract schema mismatch.</strong>
    {{ audioContractError }} Studio will not guess at routing it does not understand; forms stay
    disabled until Studio and the engine agree on the contract schema.
  </StateBanner>

  <StateBanner
    v-else-if="failureKind === 'no-engine'"
    severity="warning"
    class="config-scan-message"
  >
    <strong>No active modern audio.cpp engine.</strong>
    Studio needs an active v0.9.1 or newer audio.cpp version with JSON loader listing. Activate one
    on the Engines page and let its parameter scan finish; forms appear once it does. ({{
      audioContractError
    }})
  </StateBanner>

  <StateBanner
    v-else-if="failureKind === 'unreachable'"
    severity="error"
    class="config-scan-message"
  >
    <strong>Could not load the audio.cpp integration contract.</strong>
    {{ audioContractError }} This is a connection or server error, not a build-version signal.
    <template #actions>
      <Button
        label="Retry contract"
        icon="pi pi-refresh"
        size="small"
        severity="secondary"
        outlined
        :loading="contractRetrying"
        @click="retryContract"
      />
    </template>
  </StateBanner>
</template>

<script setup>
import { computed, ref } from 'vue'
import Button from 'primevue/button'
import StateBanner from '@/components/common/StateBanner.vue'
import {
  audioContractError,
  audioContractLoaded,
  audioContractSchemaMismatch,
  audioContractStatus,
  fetchAudioContract,
} from '@/composables/useAudioContract'

const props = defineProps({
  scanError: { type: [String, Object], default: null },
  scanPending: { type: Boolean, default: false },
})

const contractRetrying = ref(false)

// No composed contract means either no modern engine is active (409) or the
// contract request itself failed. Only a schema mismatch is a version signal;
// a network/500 error must never be reported as "build too old".
const failureKind = computed(() => {
  if (props.scanError || props.scanPending) return ''
  if (audioContractLoaded.value || !audioContractError.value) return ''
  if (audioContractSchemaMismatch.value) return 'schema-mismatch'
  if (audioContractStatus.value === 409) return 'no-engine'
  return 'unreachable'
})

async function retryContract() {
  contractRetrying.value = true
  try {
    await fetchAudioContract({ force: true })
  } finally {
    contractRetrying.value = false
  }
}
</script>
