<template>
  <div class="model-config">
    <div class="config-layout">
      <!-- Main Configuration Panel -->
      <div class="config-main">
        <div class="card">
          <div class="card-header">
            <div class="config-header">
              <div class="model-info">
                <h2 class="card-title">{{ model?.huggingface_id || model?.name || 'Model Configuration' }}</h2>
                <div class="model-tags">
                  <span class="model-tag tag-size">{{ formatFileSize(model?.file_size) }}</span>
                  <span class="model-tag tag-quantization">{{ model?.quantization }}</span>
                  <span class="model-tag tag-type">{{ model?.model_type }}</span>
                  <span v-if="modelLayerInfo?.architecture && modelLayerInfo.architecture !== model?.model_type"
                    class="model-tag tag-architecture">
                    {{ modelLayerInfo.architecture }}
                  </span>
                  <span v-if="modelLayerInfo?.layer_count" class="model-tag tag-layers">
                    {{ modelLayerInfo.layer_count }} layers
                  </span>
                </div>
                <div v-if="model?.huggingface_id && model?.name !== model?.huggingface_id"
                  class="quantization-indicator">
                  <i class="pi pi-microchip"></i>
                  <span>Configuring: {{ model?.quantization }} quantization</span>
                </div>
              </div>
              <div class="header-actions">
                <div class="preset-buttons">
                  <Button label="Coding" icon="pi pi-code" @click="applyPreset('coding')" severity="secondary"
                    size="small" outlined v-tooltip="'Apply coding-optimized parameters'" />
                  <Button label="Chat" icon="pi pi-comments" @click="applyPreset('conversational')" severity="secondary"
                    size="small" outlined v-tooltip="'Apply conversational parameters'" />
                </div>
                <Button label="Smart Auto" icon="pi pi-bolt" @click="generateAutoConfig" :loading="autoConfigLoading"
                  severity="info" size="small" />
                <Button label="Save Config" icon="pi pi-save" @click="saveConfig" :loading="saveLoading"
                  severity="success" />
              </div>
            </div>
          </div>
        </div>

        <!-- Configuration Grid -->
        <div class="config-grid">
          <!-- Model Loading -->
          <div class="config-section">
            <h3 class="section-title">
              <i class="pi pi-microchip"></i>
              Model Loading
            </h3>
            <div class="section-grid">
              <div v-if="!systemStore.gpuInfo.cpu_only_mode" class="config-field">
                <label>GPU Layers</label>
                <SliderInput v-model="config.n_gpu_layers" :min="0" :max="maxGpuLayers" :disabled="!gpuAvailable" @input="updateVramEstimate" />
                <small>
                  Layers offloaded to GPU (max: {{ maxGpuLayers }})
                  <template v-if="!gpuAvailable"> — no GPU detected</template>
                </small>
              </div>
              <div v-if="!systemStore.gpuInfo.cpu_only_mode" class="config-field">
                <label>Main GPU</label>
                <Dropdown v-model="config.main_gpu" :options="gpuOptions" optionLabel="label" optionValue="value"
                  placeholder="Select GPU" :disabled="!gpuAvailable" />
                <small>Primary GPU</small>
              </div>
              <div v-if="!systemStore.gpuInfo.cpu_only_mode" class="config-field">
                <label>Tensor Split</label>
                <InputText v-model="config.tensor_split" placeholder="0.5,0.5" :disabled="!gpuAvailable" />
                <small>Multi-GPU ratios</small>
              </div>
              <div class="config-field">
                <label>CPU Threads</label>
                <SliderInput v-model="config.threads" :min="1" :max="systemStore.gpuInfo.cpu_threads" />
                <small>CPU threads for computation</small>
              </div>
            </div>
          </div>

          <!-- Context & Memory -->
          <div class="config-section">
            <h3 class="section-title">
              <i class="pi pi-memory"></i>
              Context & Memory
            </h3>
            <div class="section-grid">
              <div class="config-field">
                <label>Context Size</label>
                <SliderInput v-model="config.ctx_size" :min="512" :max="maxContextSize" @input="updateVramEstimate" />
                <small>Max context length (max: {{ maxContextSize.toLocaleString() }})</small>
              </div>
              <div class="config-field">
                <label>Batch Size</label>
                <SliderInput v-model="config.batch_size" :min="1" :max="maxBatchSize" @input="updateVramEstimate" />
                <small>Parallel tokens (max: {{ maxBatchSize }})</small>
              </div>
              <div class="config-field">
                <label>U-Batch Size</label>
                <SliderInput v-model="config.ubatch_size" :min="1" :max="maxBatchSize" />
                <small>Unified batch (max: {{ maxBatchSize }})</small>
              </div>
              <div class="config-field">
                <label>No Memory Map</label>
                <Checkbox v-model="config.no_mmap" binary />
                <small>Disable mmap</small>
              </div>
              <div class="config-field">
                <label>Mlock</label>
                <Checkbox v-model="config.mlock" binary />
                <small>Lock model in RAM (prevent swapping)</small>
              </div>
            </div>
          </div>

          <!-- Generation -->
          <div class="config-section">
            <h3 class="section-title">
              <i class="pi pi-cog"></i>
              Generation
            </h3>
            <div class="section-grid">
              <div class="config-field">
                <label>Max Predict</label>
                <InputNumber v-model="config.n_predict" :min="-1" :max="2048" />
                <small>Max tokens (-1=unlimited)</small>
              </div>
              <div class="config-field">
                <label>Temperature</label>
                <SliderInput v-model="config.temp" :min="0.1" :max="2.0" :step="0.1" :maxFractionDigits="1" />
                <small>{{ getTemperatureTooltip() }}</small>
              </div>
              <div class="config-field">
                <label>Top-K</label>
                <SliderInput v-model="config.top_k" :min="1" :max="maxTopK" />
                <small>{{ getTopKTooltip() }}</small>
              </div>
              <div class="config-field">
                <label>Top-P</label>
                <SliderInput v-model="config.top_p" :min="0.1" :max="1.0" :step="0.1" :maxFractionDigits="1" />
                <small>{{ getTopPTooltip() }}</small>
              </div>
              <div class="config-field">
                <label>Repeat Penalty</label>
                <SliderInput v-model="config.repeat_penalty" :min="0.5" :max="2.0" :step="0.05" :maxFractionDigits="2" />
                <small>Penalty for repeating tokens</small>
              </div>
              <details class="advanced-section">
                <summary>Advanced generation options</summary>
                <div class="advanced-grid">
                  <div v-if="isMinPSupported" class="config-field">
                    <label>Min-P</label>
                    <SliderInput v-model="config.min_p" :min="0.0" :max="1.0" :step="0.05" :maxFractionDigits="2" />
                  </div>
                  <div v-if="isTypicalPSupported" class="config-field">
                    <label>Typical-P</label>
                    <SliderInput v-model="config.typical_p" :min="0.0" :max="1.0" :step="0.05" :maxFractionDigits="2" />
                  </div>
                  <div v-if="isTfsZSupported" class="config-field">
                    <label>TFS-Z</label>
                    <SliderInput v-model="config.tfs_z" :min="0.0" :max="1.0" :step="0.05" :maxFractionDigits="2" />
                  </div>
                  <div v-if="isPresencePenaltySupported" class="config-field">
                    <label>Presence Penalty</label>
                    <SliderInput v-model="config.presence_penalty" :min="0.0" :max="2.0" :step="0.1"
                      :maxFractionDigits="1" />
                  </div>
                  <div v-if="isFrequencyPenaltySupported" class="config-field">
                    <label>Frequency Penalty</label>
                    <SliderInput v-model="config.frequency_penalty" :min="0.0" :max="2.0" :step="0.1"
                      :maxFractionDigits="1" />
                  </div>
                  <div class="config-field">
                    <label>Mirostat Mode</label>
                    <Dropdown v-model="config.mirostat"
                      :options="[{ label: 'Off (0)', value: 0 }, { label: 'Mirostat (1)', value: 1 }, { label: 'Mirostat 2.0 (2)', value: 2 }]"
                      optionLabel="label" optionValue="value" />
                  </div>
                  <div class="config-field">
                    <label>Mirostat Tau</label>
                    <SliderInput v-model="config.mirostat_tau" :min="0.1" :max="20.0" :step="0.1"
                      :maxFractionDigits="2" />
                  </div>
                  <div class="config-field">
                    <label>Mirostat Eta</label>
                    <SliderInput v-model="config.mirostat_eta" :min="0.01" :max="2.0" :step="0.01"
                      :maxFractionDigits="2" />
                  </div>
                  <div class="config-field">
                    <label>Seed</label>
                    <InputNumber v-model="config.seed" :min="-1" :max="2147483647" />
                  </div>
                  <div class="config-field">
                    <label>Stop Words (comma-separated)</label>
                    <InputText v-model="stopWordsInput" @blur="applyStopWords"
                      placeholder="e.g. \\n, \\n\\n, &lt;/s&gt;" />
                  </div>
                  <div class="config-field">
                    <label>Grammar</label>
                    <InputText v-model="config.grammar" placeholder="optional grammar" />
                  </div>
                  <div v-if="isJsonSchemaSupported" class="config-field">
                    <label>JSON Schema</label>
                    <InputText v-model="config.json_schema" placeholder="optional JSON schema" />
                  </div>
                  <div class="config-field">
                    <label>Use Jinja Template</label>
                    <Checkbox v-model="config.jinja" binary />
                  </div>
                </div>
              </details>
            </div>
          </div>

          <!-- Performance -->
          <div class="config-section">
            <h3 class="section-title">
              <i class="pi pi-tachometer"></i>
              Performance
            </h3>
            <div class="section-grid">
              <div class="config-field">
                <label>Batch Threads</label>
                <SliderInput v-model="config.threads_batch" :min="1" :max="systemStore.gpuInfo.cpu_threads" />
                <small>Threads for batch processing</small>
              </div>
              <div class="config-field">
                <label>Parallel</label>
                <SliderInput v-model="config.parallel" :min="1" :max="maxParallel" />
                <small>Parallel processing (max: {{ maxParallel }})</small>
              </div>
              <div v-if="!systemStore.gpuInfo.cpu_only_mode" class="config-field">
                <label>Flash Attention</label>
                <Checkbox v-model="config.flash_attn" binary :disabled="!gpuAvailable" />
                <small>Enable flash attn (enables V cache quantization)</small>
              </div>
              <div v-if="!systemStore.gpuInfo.cpu_only_mode" class="config-field">
                <label>Low VRAM</label>
                <Checkbox v-model="config.low_vram" binary :disabled="!gpuAvailable" />
                <small>Optimize for low VRAM usage</small>
              </div>
              <div class="config-field">
                <label>Continuous Batching</label>
                <Checkbox v-model="config.cont_batching" binary />
                <small>Enable continuous/dynamic batching</small>
              </div>
              <div class="config-field">
                <label>No KV Offload</label>
                <Checkbox v-model="config.no_kv_offload" binary />
                <small>Disable KV cache offloading</small>
              </div>
              <div class="config-field">
                <label>Logits All</label>
                <Checkbox v-model="config.logits_all" binary />
                <small>Return logits for all tokens</small>
              </div>
              <div class="config-field">
                <label>Embedding Mode</label>
                <Checkbox v-model="config.embedding" binary />
                <small>Enable embedding generation mode</small>
              </div>
            </div>
          </div>

          <!-- KV Cache Optimization -->
          <div class="config-section">
            <h3 class="section-title">
              <i class="pi pi-database"></i>
              KV Cache Optimization
            </h3>
            <div v-if="!config.flash_attn && (config.cache_type_v && config.cache_type_v !== 'f16')"
              class="flash-attention-warning">
              <i class="pi pi-exclamation-triangle"></i>
              <div class="warning-content">
                <strong>Flash Attention Required</strong>
                <p>V cache quantization requires llama.cpp compiled with Flash Attention support (flag:
                  -DGGML_CUDA_FA_ALL_QUANTS=ON). Recompile your llama.cpp version or disable V cache quantization.</p>
              </div>
            </div>
            <div class="section-grid">
              <div class="config-field">
                <label>K Cache Type</label>
                <Dropdown v-model="config.cache_type_k" :options="kvCacheOptions" optionLabel="label"
                  optionValue="value" placeholder="Select K cache type" />
                <small>Key cache quantization (reduces memory usage)</small>
              </div>
              <div v-if="config.flash_attn && !systemStore.gpuInfo.cpu_only_mode && isCacheTypeVSupported" class="config-field">
                <label>V Cache Type</label>
                <Dropdown v-model="config.cache_type_v" :options="kvCacheOptions" optionLabel="label"
                  optionValue="value" placeholder="Select V cache type" />
                <small>Value cache quantization (requires Flash Attention)</small>
              </div>
            </div>
          </div>

          <!-- MoE Expert Offloading -->
          <div v-if="modelLayerInfo?.is_moe" class="config-section">
            <h3 class="section-title">
              <i class="pi pi-sitemap"></i>
              MoE Expert Offloading
            </h3>
            <div class="section-grid">
              <div class="config-field">
                <label>Offload Pattern</label>
                <Dropdown v-model="config.moe_offload_pattern" :options="moeOffloadPatterns" optionLabel="label"
                  optionValue="value" @change="handleMoEPatternChange" />
                <small>Control which MoE layers go to CPU/GPU</small>
              </div>
              <div class="config-field full-width">
                <label>Custom Offload Pattern</label>
                <InputText v-model="config.moe_offload_custom" placeholder="e.g., .ffn_.*_exps.=CPU" />
                <small>Advanced regex pattern for -ot parameter (leave empty to use pattern above)</small>
              </div>
              <div class="config-field">
                <label>Expert Info</label>
                <div class="expert-info">
                  <span>{{ modelLayerInfo.expert_count }} experts</span>
                  <span>·</span>
                  <span>{{ modelLayerInfo.experts_used_count }} active per token</span>
                </div>
              </div>
            </div>
          </div>

          <!-- Advanced -->
          <div class="config-section">
            <h3 class="section-title">
              <i class="pi pi-wrench"></i>
              Advanced
            </h3>
            <div class="section-grid">
              <div class="config-field">
                <label>RoPE Freq Base</label>
                <InputNumber v-model="config.rope_freq_base" :min="0" :max="100000" />
                <small>RoPE frequency base</small>
              </div>
              <div class="config-field">
                <label>RoPE Freq Scale</label>
                <InputNumber v-model="config.rope_freq_scale" :min="0" :max="100" :step="0.1" :maxFractionDigits="1" />
                <small>RoPE frequency scale</small>
              </div>
              <div class="config-field">
                <label>YARN Ext Factor</label>
                <InputNumber v-model="config.yarn_ext_factor" :min="0" :max="100" :step="0.1" :maxFractionDigits="1" />
                <small>YARN extension factor</small>
              </div>
              <div class="config-field">
                <label>YARN Attn Factor</label>
                <InputNumber v-model="config.yarn_attn_factor" :min="0" :max="100" :step="0.1" :maxFractionDigits="1" />
                <small>YARN attention factor</small>
              </div>
              <div class="config-field">
                <label>RoPE Scaling</label>
                <InputText v-model="config.rope_scaling" placeholder="linear, yarn" />
                <small>RoPE scaling type</small>
              </div>
              <div class="config-field">
                <label>YAML Config</label>
                <Textarea v-model="config.yaml" rows="3" placeholder="Additional YAML configuration" />
                <small>Extra YAML config</small>
              </div>
            </div>
          </div>

          <!-- Custom Arguments -->
          <div class="config-section">
            <h3 class="section-title">
              <i class="pi pi-code"></i>
              Custom Arguments
            </h3>
            <div class="section-grid">
              <div class="config-field full-width">
                <label>Custom Arguments</label>
                <Textarea v-model="config.customArgs" rows="4" placeholder="Enter custom llama.cpp arguments..." />
                <small>Additional command-line arguments</small>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Configuration Warnings -->
      <div v-if="configWarnings.length > 0" class="config-warnings">
        <h3 class="section-title">
          <i class="pi pi-exclamation-triangle"></i>
          Configuration Warnings
        </h3>
        <div class="warnings-list">
          <div v-for="warning in configWarnings" :key="warning.field" class="warning-item" :class="warning.type">
            <i :class="warning.type === 'error' ? 'pi pi-times-circle' : 'pi pi-exclamation-triangle'"></i>
            <span>{{ warning.message }}</span>
          </div>
        </div>
      </div>
      <!-- RAM Monitor Sidebar -->
      <div class="config-sidebar">
        <div class="ram-monitor">
          <div class="monitor-header">
            <h3>
              <i class="pi pi-calculator"></i>
              Memory Estimation
            </h3>
            <div class="monitor-meta" v-if="ramEstimate || realtimeRamData">
              <span v-if="ramEstimate?.system_ram_total" class="ram-snapshot">
                System RAM: {{ formatFileSize(ramEstimate.system_ram_used) }} / {{
                  formatFileSize(ramEstimate.system_ram_total) }}
              </span>
              <span v-else-if="realtimeRamData" class="ram-snapshot">
                System RAM: {{ formatFileSize(realtimeRamData.used) }} / {{ formatFileSize(realtimeRamData.total) }}
              </span>
            </div>
          </div>

          <div class="monitor-content" v-if="ramEstimate || realtimeRamData">
            <template v-if="ramEstimate">
              <div class="ram-summary">
                <div class="ram-total">
                  <span class="total-label">Estimated Usage</span>
                  <span class="total-value" :class="ramEstimate.fits_in_ram ? 'success' : 'warning'">
                    {{ formatFileSize(ramEstimate.estimated_ram) }}
                  </span>
                </div>
                <div class="ram-progress">
                  <div class="stacked-bar" :class="ramEstimate.fits_in_ram ? 'success' : 'warning'">
                    <div class="bar-current" :style="{ width: currentRamPercent + '%' }"></div>
                    <div class="bar-additional" :style="{ width: additionalRamPercent + '%', left: currentRamPercent + '%' }"></div>
                  </div>
                  <span class="progress-text">
                    {{ currentRamPercent }}% used + {{ additionalRamPercent }}% est • {{ formatFileSize(totalEstimatedRamBytes)
                    }}
                    total est
                  </span>
                </div>
              </div>

              <div class="ram-breakdown">
                <div class="breakdown-item">
                  <span class="item-label">Model</span>
                  <span class="item-value">{{ formatFileSize(ramEstimate.model_ram) }}</span>
                </div>
                <div class="breakdown-item">
                  <span class="item-label">KV Cache</span>
                  <span class="item-value">{{ formatFileSize(ramEstimate.kv_cache_ram) }}</span>
                </div>
                <div class="breakdown-item">
                  <span class="item-label">Batch</span>
                  <span class="item-value">{{ formatFileSize(ramEstimate.batch_ram) }}</span>
                </div>
                <div class="breakdown-item">
                  <span class="item-label">Overhead</span>
                  <span class="item-value">{{ formatFileSize(ramEstimate.overhead_ram) }}</span>
                </div>
              </div>

              <div v-if="showRamWarning" class="ram-warning">
                <i class="pi pi-exclamation-triangle"></i>
                <span>Estimated RAM usage exceeds available system memory</span>
              </div>
            </template>
            <template v-else>
              <div class="ram-summary">
                <div class="ram-total">
                  <span class="total-label">Current System Usage</span>
                  <span class="total-value" :class="realtimeRamData.percent > 80 ? 'warning' : 'success'">
                    {{ formatFileSize(realtimeRamData.used) }}
                  </span>
                </div>
                <div class="ram-progress">
                  <ProgressBar :value="realtimeRamData.percent" :showValue="false"
                    :class="realtimeRamData.percent > 80 ? 'warning' : 'success'" />
                  <span class="progress-text">{{ (realtimeRamData.percent || 0).toFixed(1) }}% of {{
                    formatFileSize(realtimeRamData.total) }}</span>
                </div>
              </div>

              <div class="ram-breakdown">
                <div class="breakdown-item">
                  <span class="item-label">Used</span>
                  <span class="item-value">{{ formatFileSize(realtimeRamData.used) }}</span>
                </div>
                <div class="breakdown-item">
                  <span class="item-label">Available</span>
                  <span class="item-value">{{ formatFileSize(realtimeRamData.available) }}</span>
                </div>
                <div class="breakdown-item">
                  <span class="item-label">Cached</span>
                  <span class="item-value">{{ formatFileSize(realtimeRamData.cached || 0) }}</span>
                </div>
                <div v-if="realtimeRamData.swap_total > 0" class="breakdown-item">
                  <span class="item-label">Swap Used</span>
                  <span class="item-value">{{ formatFileSize(realtimeRamData.swap_used) }} ({{
                    (realtimeRamData.swap_percent || 0).toFixed(1) }}%)</span>
                </div>
              </div>

              <div v-if="realtimeRamData.percent > 90" class="ram-warning">
                <i class="pi pi-exclamation-triangle"></i>
                <span>High RAM usage detected</span>
              </div>
            </template>
          </div>
          <div v-else class="monitor-loading">
            <i class="pi pi-spin pi-spinner"></i>
            <span>Connecting to real-time monitoring...</span>
          </div>
        </div>

        <!-- VRAM Monitor (only shown when GPU is available) -->
        <div v-if="!systemStore.gpuInfo.cpu_only_mode && systemStore.gpuInfo.device_count > 0" class="vram-monitor">
          <div class="monitor-header">
            <h3>
              <i class="pi pi-microchip"></i>
              VRAM Monitor
            </h3>
            <div class="monitor-meta" v-if="vramEstimate">
              <span class="mode-badge">Mode: {{ vramEstimate.memory_mode || 'unknown' }}</span>
              <span v-if="vramEstimate.system_ram_total" class="ram-snapshot">
                System RAM: {{ formatFileSize(vramEstimate.system_ram_used) }} / {{
                  formatFileSize(vramEstimate.system_ram_total) }}
              </span>
            </div>
          </div>

          <div class="monitor-content" v-if="realtimeVramData || vramEstimate">
            <template v-if="realtimeVramData">
              <div class="vram-summary">
                <div class="vram-total">
                  <span class="total-label">Real-time Usage</span>
                  <span class="total-value" :class="realtimeVramData.percent > 80 ? 'warning' : 'success'">
                    {{ formatFileSize(realtimeVramData.used_vram) }}
                  </span>
                </div>
                <div class="vram-progress">
                  <div class="stacked-bar"
                    :class="(currentVramPercent + additionalVramPercent) > 80 ? 'warning' : 'success'">
                    <div class="bar-current" :style="{ width: currentVramPercent + '%' }"></div>
                    <div class="bar-additional" :style="{ width: additionalVramPercent + '%', left: currentVramPercent + '%' }"></div>
                  </div>
                  <span class="progress-text">
                    {{ currentVramPercent }}% used + {{ additionalVramPercent }}% est • {{
                      formatFileSize(totalEstimatedVramBytes)
                    }} total est
                  </span>
                </div>
              </div>

              <div class="vram-breakdown">
                <div v-for="gpu in realtimeVramData.gpus" :key="gpu.index" class="gpu-item">
                  <div class="gpu-header">
                    <span class="gpu-name">{{ gpu.name }}</span>
                    <span class="gpu-temp" v-if="gpu.temperature">{{ gpu.temperature }}°C</span>
                  </div>
                  <div class="gpu-memory">
                    <span class="memory-label">VRAM</span>
                    <span class="memory-value">{{ formatFileSize(gpu.memory.used) }} / {{
                      formatFileSize(gpu.memory.total)
                    }}</span>
                    <span class="memory-percent">{{ (gpu.memory.percent || 0).toFixed(1) }}%</span>
                  </div>
                  <div v-if="gpu.utilization" class="gpu-utilization">
                    <span class="util-label">GPU</span>
                    <span class="util-value">{{ gpu.utilization.gpu || 0 }}%</span>
                    <span class="util-label">Mem</span>
                    <span class="util-value">{{ gpu.utilization.memory || 0 }}%</span>
                  </div>
                </div>
              </div>

              <div v-if="realtimeVramData.percent > 90" class="vram-warning">
                <i class="pi pi-exclamation-triangle"></i>
                <span>High VRAM usage detected</span>
              </div>
            </template>
            <template v-else>
              <div class="vram-summary">
                <div class="vram-total">
                  <span class="total-label">Estimated Usage</span>
                  <span class="total-value" :class="vramEstimate.fits_in_gpu ? 'success' : 'warning'">
                    {{ formatFileSize(vramEstimate.estimated_vram) }}
                  </span>
                </div>
                <div class="vram-progress">
                  <ProgressBar :value="vramUsagePercentage" :showValue="false"
                    :class="vramEstimate.fits_in_gpu ? 'success' : 'warning'" />
                  <span class="progress-text">{{ vramUsagePercentage }}% of available VRAM</span>
                </div>
              </div>

              <div class="vram-breakdown">
                <div class="breakdown-item">
                  <span class="item-label">Model</span>
                  <span class="item-value">{{ formatFileSize(vramEstimate.model_vram) }}</span>
                </div>
                <div class="breakdown-item">
                  <span class="item-label">KV Cache</span>
                  <span class="item-value">{{ formatFileSize(vramEstimate.kv_cache_vram) }}</span>
                </div>
                <div class="breakdown-item">
                  <span class="item-label">Batch</span>
                  <span class="item-value">{{ formatFileSize(vramEstimate.batch_vram) }}</span>
                </div>
              </div>

              <div v-if="!vramEstimate.fits_in_gpu" class="vram-warning">
                <i class="pi pi-exclamation-triangle"></i>
                <span>VRAM usage exceeds available GPU memory</span>
              </div>

              <div v-if="systemStore.gpuInfo.nvlink_topology?.has_nvlink" class="nvlink-info">
                <i class="pi pi-link"></i>
                <span>{{ systemStore.gpuInfo.nvlink_topology.recommended_strategy }}</span>
              </div>
            </template>
          </div>

          <div v-else class="monitor-loading">
            <i class="pi pi-spin pi-spinner"></i>
            <span>Connecting to real-time monitoring...</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useModelStore } from '@/stores/models'
import { useSystemStore } from '@/stores/system'
import { useWebSocketStore } from '@/stores/websocket'
import { toast } from 'vue3-toastify'
import Button from 'primevue/button'
import InputNumber from 'primevue/inputnumber'
import InputText from 'primevue/inputtext'
import Textarea from 'primevue/textarea'
import Checkbox from 'primevue/checkbox'
import Dropdown from 'primevue/dropdown'
import ProgressBar from 'primevue/progressbar'
import SliderInput from '@/components/SliderInput.vue'

const route = useRoute()
const router = useRouter()
const modelStore = useModelStore()
const systemStore = useSystemStore()
const wsStore = useWebSocketStore()

// Reactive state
const model = ref(null)
const config = ref({})
const stopWordsInput = ref('')
const applyStopWords = () => {
  const parts = (stopWordsInput.value || '').split(',').map(s => s.trim()).filter(Boolean)
  config.value.stop = parts
}
const vramEstimate = ref(null)
const vramLoading = ref(false)
const ramEstimate = ref(null)
const ramLoading = ref(false)
const autoConfigLoading = ref(false)
const saveLoading = ref(false)
const modelLayerInfo = ref(null)
const layerInfoLoading = ref(false)

// Real-time memory data from WebSocket
const realtimeRamData = ref(null)
const realtimeVramData = ref(null)

// Supported flags from llama-server
const supportedFlags = ref(null)
const supportedConfigKeys = ref({})

// GPU options
const gpuOptions = computed(() => {
  return Array.from({ length: systemStore.gpuInfo.device_count }, (_, i) => ({
    label: `GPU ${i}`,
    value: i
  }))
})

// GPU availability
const gpuAvailable = computed(() => {
  const dc = systemStore.gpuInfo?.device_count || 0
  return dc > 0
})

watch(gpuAvailable, (avail) => {
  if (!avail) {
    if (config.value && config.value.n_gpu_layers !== 0) {
      config.value.n_gpu_layers = 0
      updateVramEstimate()
    }
  }
})

onMounted(() => {
  if (!gpuAvailable.value) {
    if (config.value && config.value.n_gpu_layers !== 0) {
      config.value.n_gpu_layers = 0
      updateVramEstimate()
    }
  }
})

// VRAM usage percentage
const vramUsagePercentage = computed(() => {
  if (!vramEstimate.value || !systemStore.gpuInfo.total_vram) return 0
  return Math.round((vramEstimate.value.estimated_vram / systemStore.gpuInfo.total_vram) * 100)
})

// RAM usage percentage
const ramUsagePercentage = computed(() => {
  if (!ramEstimate.value || !systemStore.systemStatus.system?.memory?.total) return 0
  return Math.round((ramEstimate.value.estimated_ram / systemStore.systemStatus.system.memory.total) * 100)
})

// Only show RAM warning when we have valid totals and estimate exceeds available
const showRamWarning = computed(() => {
  const est = ramEstimate.value
  if (!est) return false
  const total = est.system_ram_total || systemStore.systemStatus.system?.memory?.total || 0
  const estimated = est.estimated_ram || 0
  return total > 0 && estimated > total
})

// Stacked progress computed values for RAM (current + estimated additional)
const totalRamBytes = computed(() => ramEstimate.value?.system_ram_total || systemStore.systemStatus.system?.memory?.total || 0)
const currentRamBytes = computed(() => ramEstimate.value?.system_ram_used || systemStore.systemStatus.system?.memory?.used || 0)
const estimatedRamBytes = computed(() => ramEstimate.value?.estimated_ram || 0)
const totalEstimatedRamBytes = computed(() => {
  // Total estimated RAM = current system usage + estimated additional RAM
  return currentRamBytes.value + estimatedRamBytes.value
})
const currentRamPercent = computed(() => {
  const total = totalRamBytes.value || 1
  return Math.min(100, Math.max(0, Math.round((currentRamBytes.value / total) * 100)))
})
const additionalRamPercent = computed(() => {
  const total = totalRamBytes.value || 1
  const add = ramEstimate.value?.estimated_ram || 0
  // Cap so current + additional does not exceed 100
  const pct = Math.round((add / total) * 100)
  return Math.max(0, Math.min(100 - currentRamPercent.value, pct))
})

// Stacked progress computed values for VRAM (current + estimated additional)
const totalVramBytes = computed(() => realtimeVramData.value?.total_vram || systemStore.gpuInfo.total_vram || 0)
const currentVramBytes = computed(() => realtimeVramData.value?.used_vram || 0)
const estimatedVramBytes = computed(() => vramEstimate.value?.estimated_vram || 0)
const totalEstimatedVramBytes = computed(() => {
  // Total estimated VRAM = current GPU usage + estimated additional VRAM
  return currentVramBytes.value + estimatedVramBytes.value
})
const currentVramPercent = computed(() => {
  const total = totalVramBytes.value || 1
  return Math.min(100, Math.max(0, Math.round(((currentVramBytes.value || 0) / total) * 100)))
})
const additionalVramPercent = computed(() => {
  const total = totalVramBytes.value || 1
  const add = vramEstimate.value?.estimated_vram || 0
  const pct = Math.round((add / total) * 100)
  return Math.max(0, Math.min(100 - currentVramPercent.value, pct))
})

// Maximum GPU layers based on model architecture
const maxGpuLayers = computed(() => {
  if (modelLayerInfo.value?.layer_count) {
    return modelLayerInfo.value.layer_count
  }
  return 32 // Default fallback
})

// Maximum context size based on model's actual context length
const maxContextSize = computed(() => {
  if (modelLayerInfo.value?.context_length) {
    return modelLayerInfo.value.context_length
  }
  return 131072 // Default fallback
})

// Maximum batch size based on model architecture
const maxBatchSize = computed(() => {
  if (modelLayerInfo.value?.attention_head_count && modelLayerInfo.value?.embedding_length) {
    // More sophisticated batch sizing based on model architecture
    const attentionHeads = modelLayerInfo.value.attention_head_count
    const embeddingDim = modelLayerInfo.value.embedding_length

    // Base batch size on attention complexity
    if (attentionHeads > 0) {
      return Math.min(1024, Math.max(512, attentionHeads * 16))
    }
  }
  return 512 // Default fallback
})

// Maximum parallel processing based on attention head count
const maxParallel = computed(() => {
  if (modelLayerInfo.value?.attention_head_count) {
    // Limit parallel processing based on attention heads
    return Math.min(8, Math.max(1, Math.floor(modelLayerInfo.value.attention_head_count / 4)))
  }
  return 8 // Default fallback
})

// Load supported flags from the active llama-server binary
const loadSupportedFlags = async () => {
  try {
    const response = await fetch('/api/models/supported-flags')
    if (response.ok) {
      const data = await response.json()
      supportedFlags.value = data.supported_flags || []
      supportedConfigKeys.value = data.supported_config_keys || {}
    } else {
      console.warn('Failed to load supported flags:', response.statusText)
      supportedFlags.value = []
      supportedConfigKeys.value = {}
    }
  } catch (error) {
    console.error('Error loading supported flags:', error)
    supportedFlags.value = []
    supportedConfigKeys.value = {}
  }
}

// Computed properties to check if specific flags are supported
const isTypicalPSupported = computed(() => supportedConfigKeys.value.typical_p === true)
const isMinPSupported = computed(() => supportedConfigKeys.value.min_p === true)
const isTfsZSupported = computed(() => supportedConfigKeys.value.tfs_z === true)
const isPresencePenaltySupported = computed(() => supportedConfigKeys.value.presence_penalty === true)
const isFrequencyPenaltySupported = computed(() => supportedConfigKeys.value.frequency_penalty === true)
const isJsonSchemaSupported = computed(() => supportedConfigKeys.value.json_schema === true)
const isCacheTypeVSupported = computed(() => supportedConfigKeys.value.cache_type_v === true)

// Configuration validation warnings
const configWarnings = computed(() => {
  const warnings = []

  if (modelLayerInfo.value) {
    // Check context size
    if (config.value.ctx_size > modelLayerInfo.value.context_length) {
      warnings.push({
        type: 'warning',
        message: `Context size (${config.value.ctx_size}) exceeds model's maximum context length (${modelLayerInfo.value.context_length})`,
        field: 'ctx_size'
      })
    }

    // Check batch size
    if (config.value.batch_size > maxBatchSize.value) {
      warnings.push({
        type: 'warning',
        message: `Batch size (${config.value.batch_size}) exceeds recommended maximum (${maxBatchSize.value}) based on model architecture`,
        field: 'batch_size'
      })
    }

    // Check GPU layers
    if (config.value.n_gpu_layers > modelLayerInfo.value.layer_count) {
      warnings.push({
        type: 'error',
        message: `GPU layers (${config.value.n_gpu_layers}) exceeds model's total layers (${modelLayerInfo.value.layer_count})`,
        field: 'n_gpu_layers'
      })
    }

    // Check parallel processing
    if (config.value.parallel > maxParallel.value) {
      warnings.push({
        type: 'warning',
        message: `Parallel processing (${config.value.parallel}) exceeds recommended maximum (${maxParallel.value}) based on attention head count`,
        field: 'parallel'
      })
    }
  }

  return warnings
})

onMounted(async () => {
  await loadModel()
  await systemStore.fetchSystemStatus()
  await loadSupportedFlags()
  initializeConfig()

  // Subscribe to unified monitoring updates for real-time memory data
  wsStore.subscribeToUnifiedMonitoring((data) => {
    // Extract RAM data from unified stream
    if (data.system?.memory) {
      realtimeRamData.value = {
        total: data.system.memory.total,
        available: data.system.memory.available,
        used: data.system.memory.used,
        percent: data.system.memory.percent,
        free: data.system.memory.free,
        cached: data.system.memory.cached,
        buffers: data.system.memory.buffers,
        swap_total: data.system.memory.swap_total,
        swap_used: data.system.memory.swap_used,
        timestamp: Date.now()
      }
    }

    // Extract VRAM data from unified stream
    if (data.gpu?.vram_data) {
      realtimeVramData.value = data.gpu.vram_data
    }
  })

  // Automatically trigger initial estimates if no real-time data yet
  if (!realtimeRamData.value) {
    estimateRam()
  }
  if (!realtimeVramData.value && !systemStore.gpuInfo.cpu_only_mode) {
    estimateVram()
  }
})

onUnmounted(() => {
  // WebSocket subscriptions are automatically cleaned up by the store
})

const loadModel = async () => {
  const modelId = route.params.id
  if (modelId) {
    try {
      await modelStore.fetchModels()
      // Find the specific quantization in the grouped structure
      model.value = modelStore.allQuantizations.find(m => m.id === parseInt(modelId))
      if (!model.value) {
        toast.error('The requested model quantization could not be found')
        router.push('/models')
        return
      }

      // Load model layer information
      await loadModelLayerInfo()

    } catch (error) {
      console.error('Failed to load model:', error)
      toast.error('Failed to load model')
    }
  }
}

const loadModelLayerInfo = async () => {
  if (!model.value) return

  layerInfoLoading.value = true
  try {
    const response = await fetch(`/api/models/${model.value.id}/layer-info`)
    if (response.ok) {
      modelLayerInfo.value = await response.json()
      console.log('Model layer info:', modelLayerInfo.value)
    } else {
      console.warn('Failed to load model layer info, using defaults')
      modelLayerInfo.value = { layer_count: 32 }
    }
  } catch (error) {
    console.error('Error loading model layer info:', error)
    modelLayerInfo.value = { layer_count: 32 }
  } finally {
    layerInfoLoading.value = false
  }
}

const initializeConfig = () => {
  const defaults = getDefaultConfig()
  if (model.value?.config) {
    try {
      const loaded = JSON.parse(model.value.config)
      // Merge defaults to ensure all fields have safe values
      config.value = { ...defaults, ...loaded }
      // Ensure numeric/boolean fields are properly typed
      for (const key in defaults) {
        if (config.value[key] === undefined || config.value[key] === null) {
          config.value[key] = defaults[key]
        }
        // Type coercion for critical fields
        if (typeof defaults[key] === 'boolean' && typeof config.value[key] !== 'boolean') {
          config.value[key] = Boolean(config.value[key])
        }
        if (typeof defaults[key] === 'number' && typeof config.value[key] !== 'number') {
          const num = Number(config.value[key])
          config.value[key] = isNaN(num) ? defaults[key] : num
        }
      }
    } catch (error) {
      console.error('Failed to parse model config:', error)
      config.value = { ...defaults }
    }
  } else {
    config.value = { ...defaults }
  }
}

const getDefaultConfig = () => ({
  n_gpu_layers: 0,
  main_gpu: 0,
  tensor_split: '',
  ctx_size: 4096,
  batch_size: 256,
  ubatch_size: 128,
  no_mmap: false,
  mlock: false,
  low_vram: false,
  logits_all: false,
  embedding: false,
  cont_batching: true,
  no_kv_offload: false,
  n_predict: -1,
  temp: 0.8,
  temperature: 0.8,
  top_k: 40,
  top_p: 0.9,
  repeat_penalty: 1.1,
  threads: 6,
  threads_batch: 6,
  parallel: 1,
  flash_attn: false,
  cache_type_k: 'f16',
  cache_type_v: null,
  moe_offload_pattern: 'none',
  moe_offload_custom: '',
  rope_freq_base: 10000,
  rope_freq_scale: 1.0,
  yarn_ext_factor: 1.0,
  yarn_attn_factor: 1.0,
  rope_scaling: '',
  yaml: '',
  customArgs: '',
  min_p: 0.0,
  typical_p: 1.0,
  tfs_z: 1.0,
  presence_penalty: 0.0,
  frequency_penalty: 0.0,
  mirostat: 0,
  mirostat_tau: 5.0,
  mirostat_eta: 0.1,
  seed: -1,
  stop: [],
  grammar: '',
  json_schema: '',
  jinja: false,
  host: '0.0.0.0',
  port: 0,
  timeout: 300
})

// KV cache options
const kvCacheOptions = [
  { label: 'FP32 (full precision)', value: 'f32' },
  { label: 'FP16 (half precision)', value: 'f16' },
  { label: 'BF16 (bfloat16)', value: 'bf16' },
  { label: 'Q8_0 (8-bit)', value: 'q8_0' },
  { label: 'Q5_1 (5-bit high quality)', value: 'q5_1' },
  { label: 'Q5_0 (5-bit)', value: 'q5_0' },
  { label: 'Q4_1 (4-bit high quality)', value: 'q4_1' },
  { label: 'Q4_0 (4-bit)', value: 'q4_0' },
  { label: 'IQ4_NL (4-bit non-linear)', value: 'iq4_nl' }
]

// MoE offload patterns
const moeOffloadPatterns = [
  { label: 'None', value: 'none' },
  { label: 'All MoE layers to CPU', value: 'all' },
  { label: 'Up/Down projections to CPU', value: 'up_down' },
  { label: 'Up projection only to CPU', value: 'up' },
  { label: 'Custom pattern', value: 'custom' }
]

// Apply architecture preset
const selectedPreset = ref(null)
const applyPreset = async (presetName) => {
  if (!model.value) return

  try {
    const response = await fetch(`/api/models/${model.value.id}/architecture-presets`)
    if (!response.ok) throw new Error('Failed to fetch presets')

    const data = await response.json()
    const preset = data.presets[presetName]

    if (preset) {
      selectedPreset.value = presetName
      // Merge all preset keys into config, even if not previously present
      Object.assign(config.value, preset)

      // Re-estimate memory to reflect preset changes
      await estimateVram()
      await estimateRam()

      toast.success(`${presetName.charAt(0).toUpperCase() + presetName.slice(1)} preset applied`)
    }
  } catch (error) {
    toast.error('Failed to apply preset')
  }
}

// Handle MoE pattern change
const handleMoEPatternChange = () => {
  // Automatically set the custom pattern based on selection
  switch (config.value.moe_offload_pattern) {
    case 'all':
      config.value.moe_offload_custom = '.ffn_.*_exps.=CPU'
      break
    case 'up_down':
      config.value.moe_offload_custom = '.ffn_(up|down)_exps.=CPU'
      break
    case 'up':
      config.value.moe_offload_custom = '.ffn_(up)_exps.=CPU'
      break
    case 'none':
      config.value.moe_offload_custom = ''
      break
    case 'custom':
      // User will input custom pattern
      break
  }
}

// Watch for flash attention changes to warn about V cache
watch(() => config.value.flash_attn, (newVal) => {
  // Ensure cache_type_v has a safe value based on flash_attn state
  if (!newVal) {
    // When Flash Attention is disabled, V cache should be null or f16
    if (!config.value.cache_type_v || typeof config.value.cache_type_v !== 'string') {
      config.value.cache_type_v = null
    }
  } else {
    // When Flash Attention is enabled, ensure V cache has a valid string value
    if (!config.value.cache_type_v || typeof config.value.cache_type_v !== 'string') {
      config.value.cache_type_v = config.value.cache_type_k || 'f16'
    }
  }
})

const generateAutoConfig = async () => {
  autoConfigLoading.value = true
  try {
    if (!model.value) {
      toast.error('No model selected')
      return
    }

    // Call backend smart auto API (send preset if selected)
    const presetParam = selectedPreset.value ? `?preset=${encodeURIComponent(selectedPreset.value)}` : ''
    const response = await fetch(`/api/models/${model.value.id}/smart-auto${presetParam}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    })

    if (!response.ok) {
      throw new Error(`Smart auto failed: ${response.statusText}`)
    }

    const smartConfig = await response.json()

    // Apply the smart configuration with defaults fallback
    const defaults = getDefaultConfig()
    config.value = { ...defaults, ...smartConfig }
    
    // Ensure all fields have safe values (handle nulls/undefined)
    for (const key in defaults) {
      if (config.value[key] === undefined || config.value[key] === null) {
        config.value[key] = defaults[key]
      }
      // Type coercion for critical fields
      if (typeof defaults[key] === 'boolean' && typeof config.value[key] !== 'boolean') {
        config.value[key] = Boolean(config.value[key])
      }
      if (typeof defaults[key] === 'number' && typeof config.value[key] !== 'number') {
        const num = Number(config.value[key])
        config.value[key] = isNaN(num) ? defaults[key] : num
      }
    }

    // Show success message with optimization details
    const isCpuOnlyMode = systemStore.gpuInfo.cpu_only_mode
    const optimizationType = isCpuOnlyMode ? 'CPU-optimized' : 'GPU-optimized'

    toast.success(`${optimizationType} configuration generated successfully`)

    // Update estimates after applying smart config
    await updateVramEstimate()
    await updateRamEstimate()

  } catch (error) {
    toast.error('Failed to generate automatic configuration')
  } finally {
    autoConfigLoading.value = false
  }
}

const estimateVram = async () => {
  if (!model.value) return

  vramLoading.value = true
  try {
    const response = await fetch('/api/models/vram-estimate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model_id: model.value.id,
        config: config.value
      })
    })

    if (response.ok) {
      const data = await response.json()
      vramEstimate.value = data
    } else {
      throw new Error('VRAM estimation failed')
    }
  } catch (error) {
    console.error('VRAM estimation error:', error)
    toast.error('Could not estimate VRAM usage')
  } finally {
    vramLoading.value = false
  }
}

const estimateRam = async () => {
  if (!model.value) return

  ramLoading.value = true
  try {
    const response = await fetch('/api/models/ram-estimate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model_id: model.value.id,
        config: config.value
      })
    })

    if (response.ok) {
      ramEstimate.value = await response.json()
    } else {
      throw new Error('RAM estimation failed')
    }
  } catch (error) {
    console.error('RAM estimation error:', error)
    toast.error('Could not estimate RAM usage')
  } finally {
    ramLoading.value = false
  }
}

const updateVramEstimate = () => {
  if (vramEstimate.value) {
    estimateVram()
  }
}

const updateRamEstimate = () => {
  if (ramEstimate.value) {
    estimateRam()
  }
}

const saveConfig = async () => {
  if (!model.value) return

  saveLoading.value = true
  try {
    await modelStore.updateModelConfig(model.value.id, config.value)

    toast.success('Model configuration has been updated')
  } catch (error) {
    console.error('Save config error:', error)
    toast.error('Failed to save configuration')
  } finally {
    saveLoading.value = false
  }
}

const formatFileSize = (bytes) => {
  if (!bytes) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

// Tooltip generators with architecture-specific recommendations
const getTemperatureTooltip = () => {
  const architecture = modelLayerInfo.value?.architecture?.toLowerCase() || ''
  let baseMsg = 'Controls randomness (0.1=deterministic, 2.0=creative)'

  if (architecture.includes('glm')) {
    baseMsg += ' | GLM: Recommended 1.0'
  } else if (architecture.includes('deepseek')) {
    baseMsg += ' | DeepSeek: Recommended 1.0'
  } else if (architecture.includes('qwen')) {
    baseMsg += ' | Qwen: Recommended 0.7'
  } else if (architecture.includes('codellama') || model.value?.name.toLowerCase().includes('code')) {
    baseMsg += ' | Coding: Recommended 0.1-0.7'
  }

  return baseMsg
}

const getTopKTooltip = () => {
  return `Top-K sampling (limit to top K tokens) | Recommended: 40 for GLM/DeepSeek, 50 for others`
}

const getTopPTooltip = () => {
  const architecture = modelLayerInfo.value?.architecture?.toLowerCase() || ''
  let baseMsg = 'Top-P (nucleus) sampling'

  if (architecture.includes('glm') || architecture.includes('deepseek')) {
    baseMsg += ' | Recommended: 0.95'
  } else if (architecture.includes('qwen')) {
    baseMsg += ' | Recommended: 0.9-0.95'
  } else {
    baseMsg += ' | Recommended: 0.95'
  }

  return baseMsg
}

const getRepeatPenaltyTooltip = () => {
  let baseMsg = 'Penalty for repeating tokens (1.0=no penalty)'

  const ctxLength = modelLayerInfo.value?.context_length || 0
  if (ctxLength > 32768) {
    baseMsg += ' | Long context: Use 1.0-1.05'
  } else if (ctxLength < 2048) {
    baseMsg += ' | Short context: Use 1.1-1.2'
  } else {
    baseMsg += ' | Standard: Use 1.1'
  }

  return baseMsg
}

// Watch for config changes to update estimates
watch(config, () => {
  updateVramEstimate()
  updateRamEstimate()
}, { deep: true })
</script>

<style scoped>
.model-config {
  min-height: 100vh;
  background: var(--bg-primary);
}

.config-layout {
  display: grid;
  grid-template-columns: 1fr 320px;
  gap: var(--spacing-xl);
  max-width: 1400px;
  margin: 0 auto;
  padding: var(--spacing-xl);
}

.config-main {
  min-width: 0;
}

.config-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: var(--spacing-lg);
}

.model-info {
  flex: 1;
}

.quantization-indicator {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  margin-top: var(--spacing-sm);
  padding: var(--spacing-sm);
  background: rgba(34, 211, 238, 0.1);
  border: 1px solid rgba(34, 211, 238, 0.2);
  border-radius: var(--radius-sm);
  color: var(--accent-cyan);
  font-size: 0.875rem;
  font-weight: 500;
}

.model-tag.tag-architecture {
  background: rgba(59, 130, 246, 0.1);
  color: var(--accent-blue);
  border: 1px solid rgba(59, 130, 246, 0.2);
}

.model-tag.tag-layers {
  background: rgba(34, 211, 238, 0.1);
  color: var(--accent-cyan);
  border: 1px solid rgba(34, 211, 238, 0.2);
}

/* Configuration Warnings */
.config-warnings {
  margin-top: var(--spacing-xl);
  padding: var(--spacing-lg);
  background: var(--gradient-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  backdrop-filter: blur(10px);
  box-shadow: var(--shadow-md);
  position: relative;
  overflow: hidden;
  animation: fadeIn 0.6s ease-out;
}

.config-warnings::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: var(--gradient-warning);
  z-index: 1;
}

.warnings-list {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  margin-top: var(--spacing-md);
}

.warning-item {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm) var(--spacing-md);
  border-radius: var(--radius-md);
  font-size: 0.9rem;
  transition: all var(--transition-normal);
}

.warning-item.warning {
  background: rgba(255, 193, 7, 0.1);
  border: 1px solid rgba(255, 193, 7, 0.3);
  color: var(--warning-color);
}

.warning-item.error {
  background: rgba(220, 53, 69, 0.1);
  border: 1px solid rgba(220, 53, 69, 0.3);
  color: var(--error-color);
}

.warning-item i {
  font-size: 1rem;
  flex-shrink: 0;
}

.warning-item.warning i {
  color: var(--warning-color);
}

.warning-item.error i {
  color: var(--error-color);
}

.meta-item {
  padding: var(--spacing-xs) var(--spacing-sm);
  background: var(--bg-surface);
  border-radius: var(--radius-sm);
  font-size: 0.875rem;
  color: var(--text-secondary);
}

.header-actions {
  display: flex;
  gap: var(--spacing-sm);
  align-items: center;
}

.preset-buttons {
  display: flex;
  gap: var(--spacing-xs);
}

.flash-attention-warning {
  display: flex;
  align-items: flex-start;
  gap: var(--spacing-sm);
  padding: var(--spacing-md);
  margin-bottom: var(--spacing-md);
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: var(--radius-md);
  color: var(--error-color);
}

.flash-attention-warning i {
  font-size: 1.5rem;
  flex-shrink: 0;
  margin-top: 2px;
}

.warning-content strong {
  display: block;
  margin-bottom: var(--spacing-xs);
  font-weight: 600;
}

.warning-content p {
  margin: 0;
  font-size: 0.875rem;
  line-height: 1.5;
}

.expert-info {
  display: flex;
  gap: var(--spacing-sm);
  align-items: center;
  font-size: 0.875rem;
  color: var(--text-secondary);
}

/* Configuration Grid */
.config-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: var(--spacing-xl);
  margin-top: var(--spacing-lg);
}

.config-section {
  background: var(--gradient-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  padding: var(--spacing-xl);
  box-shadow: var(--shadow-md);
  transition: all var(--transition-normal);
  position: relative;
  overflow: visible;
  backdrop-filter: blur(10px);
  animation: fadeIn 0.6s ease-out;
}

.config-section::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--gradient-primary);
  opacity: 0;
  transition: opacity var(--transition-normal);
  z-index: 1;
}

.config-section:hover {
  box-shadow: var(--shadow-lg), var(--glow-primary);
  transform: translateY(-3px);
  border-color: var(--accent-cyan);
}

.config-section:hover::before {
  opacity: 1;
}

.section-title {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  margin: 0 0 var(--spacing-lg) 0;
  color: var(--text-primary);
  font-size: 1.1rem;
  font-weight: 600;
  padding-bottom: var(--spacing-sm);
  border-bottom: 1px solid var(--border-primary);
}

.section-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: var(--spacing-lg);
  width: 100%;
  min-width: 0;
}

.config-field {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  min-width: 0;
  width: 100%;
}

.config-field.full-width {
  grid-column: 1 / -1;
}

.config-field label {
  font-weight: 500;
  color: var(--text-primary);
  font-size: 0.9rem;
}

.config-field small {
  color: var(--text-secondary);
  font-size: 0.75rem;
  line-height: 1.3;
}

/* Sidebar */
.config-sidebar {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-lg);
}

.vram-monitor {
  background: var(--gradient-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  padding: var(--spacing-xl);
  box-shadow: var(--shadow-md);
  transition: all var(--transition-normal);
  position: relative;
  overflow: visible;
  backdrop-filter: blur(10px);
  animation: fadeIn 0.6s ease-out;
}

.vram-monitor::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--gradient-primary);
  opacity: 0;
  transition: opacity var(--transition-normal);
  z-index: 1;
}

.vram-monitor:hover {
  box-shadow: var(--shadow-lg), var(--glow-primary);
  transform: translateY(-3px);
  border-color: var(--accent-cyan);
}

.vram-monitor:hover::before {
  opacity: 1;
}

.monitor-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-lg);
}

.monitor-header h3 {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  margin: 0;
  color: var(--text-primary);
  font-size: 1.1rem;
}

.monitor-meta {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  font-size: 0.875rem;
  color: var(--text-secondary);
}

.mode-badge {
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  padding: 2px 6px;
  font-weight: 500;
  color: var(--text-primary);
}

.ram-snapshot {
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  padding: 2px 6px;
  font-weight: 500;
  color: var(--text-primary);
}

.monitor-content {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.vram-summary {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.vram-total {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.total-label {
  font-size: 0.875rem;
  color: var(--text-secondary);
}

.total-value {
  font-weight: 700;
  font-size: 1.1rem;
}

.total-value.success {
  color: var(--status-success);
}

.total-value.warning {
  color: var(--status-warning);
}

.vram-progress {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.progress-text {
  font-size: 0.75rem;
  color: var(--text-secondary);
  text-align: center;
}

.vram-breakdown {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.breakdown-item {
  display: flex;
  justify-content: space-between;
  font-size: 0.875rem;
}

.item-label {
  color: var(--text-secondary);
}

.item-value {
  color: var(--text-primary);
  font-weight: 500;
}

.vram-warning {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm);
  background: rgba(245, 158, 11, 0.1);
  border: 1px solid rgba(245, 158, 11, 0.3);
  border-radius: var(--radius-md);
  color: var(--status-warning);
  font-size: 0.875rem;
}

.nvlink-info {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm);
  background: rgba(34, 211, 238, 0.1);
  border: 1px solid rgba(34, 211, 238, 0.3);
  border-radius: var(--radius-md);
  color: var(--accent-cyan);
  font-size: 0.875rem;
}

.monitor-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-xl);
  color: var(--text-muted);
  text-align: center;
}

.monitor-empty i {
  font-size: 2rem;
  color: var(--text-muted);
}

.monitor-loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-xl);
  color: var(--text-secondary);
  text-align: center;
}

.monitor-loading i {
  font-size: 2rem;
  color: var(--accent-primary);
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from {
    transform: rotate(0deg);
  }

  to {
    transform: rotate(360deg);
  }
}

.cpu-mode-info {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.cpu-mode-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  font-weight: 600;
  color: var(--text-primary);
  font-size: 1rem;
}

.cpu-threads-info {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.threads-item {
  display: flex;
  justify-content: space-between;
  font-size: 0.875rem;
}

.threads-label {
  color: var(--text-secondary);
}

.threads-value {
  color: var(--text-primary);
  font-weight: 500;
}

.cpu-performance-tip {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm);
  background: rgba(34, 211, 238, 0.1);
  border: 1px solid rgba(34, 211, 238, 0.3);
  border-radius: var(--radius-md);
  color: var(--accent-cyan);
  font-size: 0.875rem;
}

/* RAM Monitor Styles */
.ram-monitor {
  background: var(--gradient-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  padding: var(--spacing-xl);
  box-shadow: var(--shadow-md);
  transition: all var(--transition-normal);
  position: relative;
  overflow: visible;
  backdrop-filter: blur(10px);
  animation: fadeIn 0.6s ease-out;
  margin-bottom: var(--spacing-lg);
}

.ram-monitor::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--gradient-primary);
  opacity: 0;
  transition: opacity var(--transition-normal);
  z-index: 1;
}

.ram-monitor:hover {
  box-shadow: var(--shadow-lg), var(--glow-primary);
  transform: translateY(-3px);
  border-color: var(--accent-cyan);
}

.ram-monitor:hover::before {
  opacity: 1;
}

.ram-summary {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  margin-bottom: var(--spacing-lg);
}

.ram-total {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.ram-progress {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.ram-breakdown {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  margin-bottom: var(--spacing-lg);
}

.ram-warning {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm);
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: var(--radius-md);
  color: var(--accent-red);
  font-size: 0.875rem;
}

/* GPU-specific styles for real-time monitoring */
.gpu-item {
  padding: var(--spacing-sm);
  background: var(--bg-tertiary);
  border-radius: var(--radius-md);
  margin-bottom: var(--spacing-sm);
}

.gpu-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-xs);
}

.gpu-name {
  font-weight: 500;
  color: var(--text-primary);
  font-size: 0.875rem;
}

.gpu-temp {
  color: var(--text-secondary);
  font-size: 0.75rem;
  background: var(--bg-surface);
  padding: 2px 6px;
  border-radius: var(--radius-sm);
}

.gpu-memory {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-xs);
}

.memory-label {
  color: var(--text-secondary);
  font-size: 0.75rem;
}

.memory-value {
  color: var(--text-primary);
  font-size: 0.875rem;
  font-weight: 500;
}

.memory-percent {
  color: var(--text-secondary);
  font-size: 0.75rem;
}

.gpu-utilization {
  display: flex;
  gap: var(--spacing-sm);
  font-size: 0.75rem;
}

.util-label {
  color: var(--text-secondary);
}

.util-value {
  color: var(--text-primary);
  font-weight: 500;
}

/* Responsive */
@media (max-width: 1200px) {
  .config-grid {
    grid-template-columns: 1fr;
  }

  .section-grid {
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  }
}

@media (max-width: 768px) {
  .config-layout {
    grid-template-columns: 1fr;
    gap: var(--spacing-lg);
  }

  .config-sidebar {
    order: -1;
  }
}

@media (max-width: 600px) {
  .config-layout {
    padding: var(--spacing-md);
  }

  .config-header {
    flex-direction: column;
    align-items: stretch;
    gap: var(--spacing-md);
  }

  .header-actions {
    justify-content: flex-end;
  }

  .config-grid {
    grid-template-columns: 1fr;
    gap: var(--spacing-lg);
  }

  .section-grid {
    grid-template-columns: 1fr;
  }

  .model-meta {
    flex-wrap: wrap;
  }
}

.stacked-bar {
  position: relative;
  width: 100%;
  height: 10px;
  background: var(--bg-secondary);
  border-radius: 5px;
  overflow: hidden;
}

.stacked-bar .bar-current {
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  background: var(--accent-red);
}

.stacked-bar .bar-additional {
  position: absolute;
  top: 0;
  bottom: 0;
  background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
  opacity: 0.8;
  transform-origin: left;
  /* left position is set via inline style dynamically */
}

/* Use inline styles for widths; classes control colors */
.stacked-bar.success {
  box-shadow: inset 0 0 0 1px rgba(16, 185, 129, 0.2);
}

.stacked-bar.warning {
  box-shadow: inset 0 0 0 1px rgba(234, 179, 8, 0.3);
}

.advanced-section {
  grid-column: 1 / -1;
  padding: var(--spacing-sm) 0;
}

.advanced-section summary {
  cursor: pointer;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: var(--spacing-sm);
}

.advanced-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: var(--spacing-md);
}
</style>