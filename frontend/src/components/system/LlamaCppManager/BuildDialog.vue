<template>
  <BaseDialog
    :visible="visible"
    header="Build from Source"
    :modal="true"
    :dialog-style="{ width: '70vw', maxWidth: '900px' }"
    :draggable="false"
    :resizable="false"
    dialog-class="build-dialog"
    @update:visible="$emit('update:visible', $event)"
  >
    <div class="build-form">
      <div class="dialog-section">
        <h4 class="section-title">Source Information</h4>
        <div class="form-field full-width">
          <label>Commit SHA or Branch *</label>
          <InputText 
            v-model="buildForm.commitSha"
            placeholder="master or commit hash"
          />
          <small>Default: master (latest stable)</small>
        </div>
        
        <div class="form-field full-width">
          <label>Patches (Optional)</label>
          <Textarea 
            v-model="buildForm.patches"
            rows="3"
            placeholder="One patch URL per line"
          />
          <small>GitHub PR URLs or raw patch file URLs</small>
        </div>
      </div>
      
      <div class="dialog-section">
        <h4 class="section-title">Build Configuration</h4>
        <div class="form-row">
          <div class="form-field">
            <label>Build Type</label>
            <Dropdown 
              v-model="buildForm.buildType"
              :options="['Release', 'Debug', 'RelWithDebInfo']"
              placeholder="Select build type"
            />
            <small>Release recommended for production</small>
          </div>
        </div>
      </div>
      
      <div class="dialog-section">
        <h4 class="section-title">GPU Backends</h4>
        <div class="checkbox-group">
          <div class="checkbox-item">
            <Checkbox 
              v-model="buildForm.enableCuda" 
              :binary="true"
            />
            <div class="checkbox-label">
              <span>CUDA</span>
              <small class="capability-info" :class="getCapabilityClass(buildCapabilities?.cuda)">
                Enables NVIDIA GPU acceleration (requires driver + CUDA runtime). {{ buildCapabilities?.cuda?.reason || 'Not detected' }}
              </small>
              <div v-if="buildForm.enableCuda && !cudaStatus?.installed" class="cuda-install-prompt">
                <i class="pi pi-exclamation-triangle" style="color: var(--status-warning);"></i>
                <span>CUDA Toolkit not detected. </span>
                <Button 
                  label="Install CUDA" 
                  icon="pi pi-download"
                  size="small"
                  severity="warning"
                  text
                  @click="$emit('show-cuda-install')"
                />
              </div>
            </div>
          </div>
          
          <div class="checkbox-item">
            <Checkbox 
              v-model="buildForm.enableVulkan" 
              :binary="true"
            />
            <div class="checkbox-label">
              <span>Vulkan</span>
              <small class="capability-info" :class="getCapabilityClass(buildCapabilities?.vulkan)">
                Cross-vendor GPU backend for AMD/Intel/NVIDIA. {{ buildCapabilities?.vulkan?.reason || 'Not detected' }}
              </small>
            </div>
          </div>
          
          <div class="checkbox-item">
            <Checkbox 
              v-model="buildForm.enableMetal" 
              :binary="true"
            />
            <div class="checkbox-label">
              <span>Metal</span>
              <small class="capability-info" :class="getCapabilityClass(buildCapabilities?.metal)">
                Apple Silicon/AMD Metal backend for macOS builds. {{ buildCapabilities?.metal?.reason || 'Not detected' }}
              </small>
            </div>
          </div>
          
          <div class="checkbox-item">
            <Checkbox 
              v-model="buildForm.enableOpenBLAS" 
              :binary="true"
            />
            <div class="checkbox-label">
              <span>OpenBLAS</span>
              <small class="capability-info" :class="getCapabilityClass(buildCapabilities?.openblas)">
                Uses CPU BLAS kernels (OpenBLAS backend). {{ buildCapabilities?.openblas?.reason || 'Not detected' }}
              </small>
            </div>
          </div>
          
          <div class="checkbox-item">
            <Checkbox 
              v-model="buildForm.enableFlashAttention" 
              :binary="true"
            />
            <div class="checkbox-label">
              <span>Flash Attention (FA)</span>
              <small class="capability-info" :class="buildForm.enableCuda ? 'text-blue-500' : 'text-gray-500'">
                CUDA-only: compiles FlashAttention kernels needed for KV-cache quantization
              </small>
            </div>
          </div>
        </div>
      </div>
      
      <div class="dialog-section">
        <h4 class="section-title">Build Artifacts</h4>
        <div class="checkbox-group">
          <div class="checkbox-item">
            <Checkbox 
              v-model="buildForm.buildExamples" 
              :binary="true"
            />
            <div class="checkbox-label">
              <span>Examples</span>
              <small class="capability-info text-gray-500">
                Compiles example apps (benchmarking, embedding demos, playground)
              </small>
            </div>
          </div>

          <div class="checkbox-item">
            <Checkbox 
              v-model="buildForm.buildTests" 
              :binary="true"
            />
            <div class="checkbox-label">
              <span>Test Suite</span>
              <small class="capability-info text-gray-500">
                Adds CTest targets; useful for CI or verifying new toolchains
              </small>
            </div>
          </div>
        </div>
        <small class="option-note">
          Core binaries (`llama-server`, CLI tooling, shared libraries) are always built to keep the Studio API fully functional.
        </small>
      </div>

      <div class="dialog-section">
        <h4 class="section-title">CPU &amp; Link Options</h4>
        <div class="checkbox-group">
          <div class="checkbox-item">
            <Checkbox 
              v-model="buildForm.enableCpuAllVariants" 
              :binary="true"
            />
            <div class="checkbox-label">
              <span>CPU All Variants</span>
              <small class="capability-info" :class="buildForm.enableBackendDl ? 'text-blue-500' : 'text-gray-500'">
                Compiles every CPU ISA variant (AVX, AVX2, AVX512, etc.); requires backend loader
              </small>
            </div>
          </div>

          <div class="checkbox-item">
            <Checkbox 
              v-model="buildForm.enableNative" 
              :binary="true"
            />
            <div class="checkbox-label">
              <span>Native Optimizations</span>
              <small class="capability-info text-gray-500">
                Enables `-march=native` style tuning; disable to produce broadly portable binaries
              </small>
            </div>
          </div>

          <div class="checkbox-item">
            <Checkbox 
              v-model="buildForm.enableLto" 
              :binary="true"
            />
            <div class="checkbox-label">
              <span>Link Time Optimization (LTO)</span>
              <small class="capability-info text-gray-500">
                Turns on LTO / thin-LTO, shrinking binaries and improving throughput (longer link step)
              </small>
            </div>
          </div>
        </div>
      </div>
      
      <Accordion class="advanced-accordion">
        <AccordionTab header="Advanced Options">
          <div class="form-field">
            <label>Custom CMake Arguments</label>
            <InputText 
              v-model="buildForm.customCmakeArgs"
              placeholder='-DLLAMA_BUILD_TESTS=OFF -DGGML_LTO=ON'
            />
            <small>Additional CMake flags</small>
          </div>
          <div class="form-row">
            <div class="form-field">
              <label>CFLAGS</label>
              <InputText v-model="buildForm.cflags" placeholder="-O3 -march=native"/>
            </div>
            <div class="form-field">
              <label>CXXFLAGS</label>
              <InputText v-model="buildForm.cxxflags" placeholder="-O3 -march=native"/>
            </div>
          </div>
        </AccordionTab>
      </Accordion>
    </div>
    
    <template #footer>
      <Button 
        label="Cancel" 
        icon="pi pi-times" 
        @click="$emit('update:visible', false)"
        severity="secondary"
        text
      />
      <Button 
        label="Build" 
        icon="pi pi-code" 
        @click="handleBuild"
        :loading="building"
      />
    </template>
  </BaseDialog>
</template>

<script setup>
import { ref, watch, onMounted } from 'vue'
import { useSystemStore } from '@/stores/system'
import { toast } from 'vue3-toastify'
import Button from 'primevue/button'
import InputText from 'primevue/inputtext'
import Textarea from 'primevue/textarea'
import Dropdown from 'primevue/dropdown'
import Checkbox from 'primevue/checkbox'
import Accordion from 'primevue/accordion'
import AccordionTab from 'primevue/accordiontab'
import BaseDialog from '@/components/common/BaseDialog.vue'

const props = defineProps({
  visible: {
    type: Boolean,
    default: false
  },
  buildCapabilities: {
    type: Object,
    default: null
  },
  cudaStatus: {
    type: Object,
    default: null
  }
})

const emit = defineEmits(['update:visible', 'build', 'show-cuda-install'])

const systemStore = useSystemStore()

const building = ref(false)

const buildForm = ref({
  commitSha: 'master',
  patches: '',
  buildType: 'Release',
  enableCuda: false,
  enableVulkan: false,
  enableMetal: false,
  enableOpenBLAS: false,
  enableFlashAttention: false,
  buildCommon: true,
  buildTests: false,
  buildTools: true,
  buildExamples: false,
  buildServer: true,
  installTools: true,
  enableBackendDl: false,
  enableCpuAllVariants: false,
  enableLto: false,
  enableNative: true,
  customCmakeArgs: '',
  cflags: '',
  cxxflags: ''
})

const getCapabilityClass = (capability) => {
  if (!capability) return 'text-gray-500'
  return capability.available ? 'text-green-500' : 'text-gray-500'
}

const handleBuild = async () => {
  building.value = true
  
  try {
    const patches = buildForm.value.patches
      .split('\n')
      .map(line => line.trim())
      .filter(line => line)
    
    const buildConfig = {
      build_type: buildForm.value.buildType || 'Release',
      enable_cuda: buildForm.value.enableCuda || false,
      enable_vulkan: buildForm.value.enableVulkan || false,
      enable_metal: buildForm.value.enableMetal || false,
      enable_openblas: buildForm.value.enableOpenBLAS || false,
      enable_flash_attention: buildForm.value.enableFlashAttention || false,
      build_common: buildForm.value.buildCommon,
      build_tests: buildForm.value.buildTests,
      build_tools: buildForm.value.buildTools,
      build_examples: buildForm.value.buildExamples,
      build_server: buildForm.value.buildServer,
      install_tools: buildForm.value.installTools,
      enable_backend_dl: buildForm.value.enableBackendDl,
      enable_cpu_all_variants: buildForm.value.enableCpuAllVariants,
      enable_lto: buildForm.value.enableLto,
      enable_native: buildForm.value.enableNative,
      custom_cmake_args: buildForm.value.customCmakeArgs || '',
      cflags: buildForm.value.cflags || '',
      cxxflags: buildForm.value.cxxflags || ''
    }
    
    await systemStore.buildSource(buildForm.value.commitSha, patches, buildConfig)
    emit('build', { commitSha: buildForm.value.commitSha, patches, buildConfig })
    emit('update:visible', false)
    toast.success('Build started successfully')
  } catch (error) {
    toast.error('Failed to start build from source')
  } finally {
    building.value = false
  }
}

watch(() => props.buildCapabilities, (capabilities) => {
  if (capabilities && props.visible) {
    buildForm.value.enableCuda = capabilities.cuda?.recommended || false
    buildForm.value.enableVulkan = capabilities.vulkan?.recommended || false
    buildForm.value.enableMetal = capabilities.metal?.recommended || false
    buildForm.value.enableOpenBLAS = capabilities.openblas?.recommended || false
  }
}, { immediate: true })

watch(() => props.visible, (newVisible) => {
  if (newVisible && props.buildCapabilities) {
    buildForm.value.enableCuda = props.buildCapabilities.cuda?.recommended || false
    buildForm.value.enableVulkan = props.buildCapabilities.vulkan?.recommended || false
    buildForm.value.enableMetal = props.buildCapabilities.metal?.recommended || false
    buildForm.value.enableOpenBLAS = props.buildCapabilities.openblas?.recommended || false
  }
})

watch(
  () => buildForm.value.enableCpuAllVariants,
  (value) => {
    if (value) {
      buildForm.value.enableBackendDl = true
    }
  }
)

watch(
  () => buildForm.value.enableBackendDl,
  (value) => {
    if (!value && buildForm.value.enableCpuAllVariants) {
      buildForm.value.enableCpuAllVariants = false
    }
  }
)
</script>

<style scoped>
.build-dialog :deep(.p-dialog-content) {
  padding: 2rem 1.5rem !important;
  overflow-y: auto !important;
  max-height: calc(80vh - 120px) !important;
  background: transparent !important;
}

.build-form {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.dialog-section {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  padding: 1.5rem;
  background: var(--gradient-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  transition: all var(--transition-normal);
  position: relative;
  overflow: hidden;
  backdrop-filter: blur(10px);
  animation: fadeIn 0.6s ease-out;
}

.dialog-section::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--gradient-primary);
  opacity: 0;
  transition: opacity var(--transition-normal);
}

.dialog-section:hover {
  border-color: var(--accent-cyan);
  box-shadow: var(--shadow-lg), var(--glow-primary);
  transform: translateY(-2px);
}

.dialog-section:hover::before {
  opacity: 1;
}

.section-title {
  margin: 0 0 0.75rem 0;
  color: var(--text-primary);
  font-size: 1.2rem;
  font-weight: 700;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.section-title::before {
  content: '';
  width: 4px;
  height: 1.5rem;
  background: var(--gradient-primary);
  border-radius: 2px;
}

.form-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
}

.form-field {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.form-field.full-width {
  grid-column: 1 / -1;
}

.form-field label {
  font-weight: 600;
  color: var(--text-primary);
  font-size: 0.9rem;
}

.form-field small {
  color: var(--text-secondary);
  font-size: 0.8rem;
  margin-top: 0.25rem;
}

.advanced-accordion {
  margin-top: 0.5rem;
}

.checkbox-group {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.checkbox-item {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  padding: 1rem;
  background: var(--gradient-surface);
  border-radius: var(--radius-lg);
  border: 1px solid var(--border-primary);
  transition: all var(--transition-normal);
  position: relative;
  overflow: hidden;
}

.checkbox-item::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: var(--gradient-primary);
  opacity: 0;
  transition: opacity var(--transition-normal);
}

.checkbox-item:hover {
  background: var(--gradient-card);
  border-color: var(--accent-cyan);
  transform: translateX(4px);
  box-shadow: var(--shadow-md);
}

.checkbox-item:hover::before {
  opacity: 1;
}

.checkbox-label {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  flex: 1;
}

.checkbox-label span {
  font-weight: 600;
  color: var(--text-primary);
  font-size: 0.9rem;
}

.cuda-install-prompt {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 0.5rem;
  padding: 0.5rem;
  background: var(--status-warning-soft);
  border-radius: var(--radius-md);
  font-size: 0.875rem;
}

.capability-info {
  color: var(--text-secondary);
  font-size: 0.8rem;
  font-weight: 400;
  line-height: 1.4;
}

.capability-info.text-green-500 {
  color: var(--status-success);
}

.capability-info.text-gray-500 {
  color: var(--text-secondary);
}

.capability-info.text-blue-500 {
  color: var(--accent-blue);
}

.option-note {
  display: block;
  margin-top: 0.5rem;
  color: var(--text-secondary);
  font-size: 0.8rem;
}
</style>

