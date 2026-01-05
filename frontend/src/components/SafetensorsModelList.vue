<template>
  <div class="safetensors-card">
    <div class="card-header">
      <div>
        <h2>Safetensors Models</h2>
        <p class="subtitle">Run Hugging Face safetensors via LMDeploy TurboMind</p>
      </div>
      <div class="actions">
        <Button 
          icon="pi pi-refresh" 
          @click="$emit('refresh')" 
          :loading="loading"
          severity="secondary"
          text
        />
        <Button 
          icon="pi pi-sync"
          @click="refreshStatus"
          :loading="statusLoading"
          severity="secondary"
          text
          v-tooltip.bottom="'Refresh LMDeploy status'"
        />
        <Button 
          icon="pi pi-database"
          @click="reloadFromDisk"
          :loading="reloadingFromDisk"
          severity="secondary"
          text
          v-tooltip.bottom="'Reset database entries and reload all safetensors models from disk'"
        />
      </div>
    </div>

    <div v-if="loading" class="loading-state">
      <i class="pi pi-spin pi-spinner"></i>
      <span>Loading safetensors models...</span>
    </div>

    <div v-if="!lmdeployReady" class="lmdeploy-alert">
      <div class="alert-icon">
        <i class="pi pi-exclamation-triangle"></i>
      </div>
      <div class="alert-content">
        <h3>LMDeploy is not installed</h3>
        <p>
          Install LMDeploy from the new LMDeploy page before starting safetensors runtimes.
          The install happens at runtime, keeping the Docker image slim.
        </p>
        <Button label="Open LMDeploy Page" icon="pi pi-box" size="small" @click="openLmdeployPage" />
      </div>
    </div>

    <div v-if="lmdeployOperation" class="lmdeploy-alert info">
      <div class="alert-icon">
        <i class="pi pi-spin pi-spinner"></i>
      </div>
      <div class="alert-content">
        <h3>Installer running</h3>
        <p>LMDeploy installer is currently {{ lmdeployOperation }}. Runtime controls are disabled until it finishes.</p>
      </div>
    </div>

    <div v-else-if="groupedModels.length > 0" class="model-grid">
      <div 
        v-for="group in groupedModels" 
        :key="group.huggingface_id"
        class="model-card grouped-card"
      >
        <div class="model-card-header group-header">
          <div class="group-header-main">
            <div class="model-name">{{ group.huggingface_id }}</div>
            <div class="model-path" v-if="group.metadata?.base_model">{{ group.metadata.base_model }}</div>
          </div>
          <div class="group-summary">
            <span>{{ group.file_count }} {{ group.file_count === 1 ? 'file' : 'files' }}</span>
            <span class="dot">•</span>
            <span>{{ formatFileSize(group.total_size) }}</span>
          </div>
        </div>

        <div class="group-status-row">
          <span
            :class="[
              'status-indicator',
              isGroupRunning(group) ? 'status-running' : 'status-stopped'
            ]"
          >
            <i :class="isGroupRunning(group) ? 'pi pi-play' : 'pi pi-pause'"></i>
            <span>{{ isGroupRunning(group) ? 'Running in LMDeploy' : 'Stopped' }}</span>
          </span>
        </div>

        <div class="model-body grouped-body">
          <div class="file-list plain-file-list">
            <span 
              v-for="file in group.files" 
              :key="file.model_id || file.filename"
              class="file-name-plain"
            >
              {{ file.filename }}
            </span>
          </div>

          <div class="group-actions">
            <div class="action-group">
              <Button 
                label="Configure & Run" 
                icon="pi pi-sliders-h"
                severity="secondary"
                size="small"
                :disabled="!group.files?.length"
                @click="openGroupConfig(group)"
                :loading="isGroupConfigLoading(group)"
              />
              <Button 
                v-if="group.files?.length && isGroupRunning(group)"
                label="Stop" 
                icon="pi pi-stop"
                severity="danger"
                size="small"
                outlined
                :loading="isGroupStopping(group)"
                @click="stopGroupRuntime(group)"
              />
            </div>
            <Button 
              icon="pi pi-trash"
              severity="danger"
              size="small"
              outlined
              text
              :disabled="!group.files?.length"
              @click="$emit('delete', group)"
            />
          </div>
        </div>
      </div>
    </div>

    <div v-else class="empty-state">
      <i class="pi pi-shield"></i>
      <h3>No Safetensors Models</h3>
      <p>Download safetensors files from the Search tab to prepare them for LMDeploy.</p>
    </div>

    <Dialog 
      v-model:visible="dialogVisible" 
      modal 
      :style="{ width: '1000px', maxWidth: '95vw' }"
      :breakpoints="{ '1200px': '90vw', '960px': '95vw', '640px': '95vw' }"
    >
      <template #header>
        <div class="dialog-header">
          <div>
            <h3>{{ selectedModel?.huggingface_id || 'Configure LMDeploy' }}</h3>
            <p v-if="selectedModel?.files?.length">
              {{ selectedModel.files.length }} file{{ selectedModel.files.length !== 1 ? 's' : '' }}
            </p>
            <p v-else-if="selectedModel?.filename">{{ selectedModel.filename }}</p>
          </div>
          <div class="dialog-header-actions">
            <Button
              label="Regenerate Metadata"
              icon="pi pi-refresh"
              severity="info"
              outlined
              size="small"
              :loading="metadataRefreshing"
              :disabled="metadataRefreshing || !selectedModelId"
              @click="regenerateMetadata"
              v-tooltip.top="'Refresh model metadata from Hugging Face'"
            />
            <Tag 
              :severity="selectedModelRunning ? 'success' : 'warning'"
              :value="selectedModelRunning ? 'Running' : 'Stopped'"
            />
          </div>
        </div>
      </template>

      <div v-if="selectedRuntime">
        <div class="config-section">
          <h4>Sequence & Parallelism</h4>
          <div class="config-grid">
            <div class="config-field span-2">
              <label>Session Length (--session-len)</label>
              <div class="slider-row">
                <Slider v-model="formState.session_len" :min="1024" :max="sessionLimit" :step="256" />
                <InputNumber v-model="formState.session_len" :min="1024" :max="sessionLimit" :step="256" inputId="sessionInput" />
              </div>
              <small class="field-help">
                Maximum sequence length for a conversation. Controls the context window size.
                Base context from metadata:
                <span v-if="baseContextLength">{{ baseContextLength.toLocaleString() }} tokens</span>
                <span v-else>unknown</span>.
                <span v-if="isQwen3 && baseContextLength === 32768" class="qwen3-note">
                  <strong>Note:</strong> For Qwen3 models, max_position_embeddings (40,960) includes 32,768 tokens for outputs and 8,192 reserved for prompts.
                  The UI shows the usable output context (32,768) for reference, but you can configure up to the full capacity.
                  If average context ≤ 32,768, YaRN scaling is not recommended as it may degrade performance.
                </span>
                Enable RoPE / YaRN scaling below to multiply the base context (up to {{ MAX_SCALING_FACTOR }}×) when supported.
                Use the "Regenerate Metadata" button in the dialog header to refresh model metadata.
              </small>
            </div>
            <div class="config-field span-2">
              <label>RoPE / YaRN Scaling</label>
              <div class="rope-scaling-controls">
                <Dropdown
                  v-model="formState.rope_scaling_mode"
                  :options="ropeScalingOptions"
                  optionLabel="label"
                  optionValue="value"
                  :disabled="!canUseScaling"
                  class="rope-mode-dropdown"
                />
                <div class="slider-row rope-factor-row">
                  <Slider
                    v-model="formState.rope_scaling_factor"
                    :min="1"
                    :max="MAX_SCALING_FACTOR"
                    :step="0.05"
                    :disabled="!scalingEnabled"
                  />
                  <InputNumber
                    v-model="formState.rope_scaling_factor"
                    :min="1"
                    :max="MAX_SCALING_FACTOR"
                    :step="0.05"
                    :disabled="!scalingEnabled"
                    inputId="ropeScalingInput"
                    mode="decimal"
                  />
                </div>
              </div>
              <small class="field-help">
                <template v-if="canUseScaling">
                  Effective context: {{ effectiveContextLength.toLocaleString() }} tokens
                  <span v-if="scalingEnabled">
                    <span v-if="adaptedBaseContext">
                      ({{ formState.rope_scaling_factor.toFixed(2) }}× {{ adaptedBaseContext.toLocaleString() }} adapted base)
                    </span>
                    <span v-else>
                      ({{ formState.rope_scaling_factor.toFixed(2) }}× {{ sessionLimit.toLocaleString() }} base)
                    </span>
                    <span v-if="maxPositionEmbeddings && effectiveContextLength >= maxPositionEmbeddings" class="max-length-warning">
                      (clamped to max_position_embeddings: {{ maxPositionEmbeddings.toLocaleString() }})
                    </span>
                  </span>
                  <span v-else>(scaling disabled)</span>.
                  <div v-if="scalingWarning" class="scaling-warning" style="color: orange; margin-top: 0.5rem;">
                    <i class="pi pi-exclamation-triangle"></i>
                    <span>{{ scalingWarning }}</span>
                  </div>
                  <div v-if="modelMaxLength" class="max-length-info">
                    Model max length: {{ modelMaxLength.toLocaleString() }} tokens.
                  </div>
                  <div v-if="maxPositionEmbeddings" class="max-length-info">
                    Max position embeddings: {{ maxPositionEmbeddings.toLocaleString() }} tokens.
                  </div>
                </template>
                <template v-else>
                  <span v-if="maxPositionEmbeddings">
                    Max position embeddings: {{ maxPositionEmbeddings.toLocaleString() }} tokens.
                  </span>
                  <span v-else>
                    RoPE scaling requires a known base context length. Regenerate metadata if you expect one.
                  </span>
                </template>
              </small>
            </div>
            <div class="config-field span-2">
              <label>HF Rope Scaling Overrides (--hf-overrides.⋯)</label>
              <div class="hf-overrides-grid">
                <div class="hf-override-field">
                  <span class="field-label">rope_scaling.rope_type</span>
                  <InputText
                    v-model="formState.hf_override_rope_type"
                    placeholder="e.g. yarn"
                    :disabled="!scalingEnabled"
                  />
                </div>
                <div class="hf-override-field">
                  <span class="field-label">rope_scaling.factor</span>
                  <InputNumber
                    v-model="formState.hf_override_rope_factor"
                    :min="1"
                    :max="MAX_SCALING_FACTOR"
                    :step="0.05"
                    mode="decimal"
                    :disabled="!scalingEnabled"
                  />
                </div>
                <div class="hf-override-field">
                  <span class="field-label">rope_scaling.original_max_position_embeddings</span>
                  <InputNumber
                    v-model="formState.hf_override_rope_original_max"
                    :min="0"
                    :max="SESSION_FALLBACK_LIMIT"
                    :step="512"
                    :disabled="!scalingEnabled"
                  />
                </div>
              </div>
              <small class="field-help">
                These map directly to individual <code>--hf-overrides.rope_scaling.*</code> flags.
                Fill them when LMDeploy requires explicit Hugging Face rope overrides for scaling.
              </small>
            </div>
            <div class="config-field">
              <label>Max Prefill Tokens (--max-prefill-token-num)</label>
              <InputNumber v-model="formState.max_prefill_token_num" :step="256" />
              <small class="field-help">Maximum tokens processed per iteration during prefill phase. Higher values increase throughput but use more memory. Default: 8192</small>
            </div>
            <div class="config-field">
              <label>Tensor Parallel (--tp)</label>
              <InputNumber v-model="formState.tensor_parallel" :min="1" :max="8" :step="1" />
              <small class="field-help">Number of GPUs for tensor parallelism. Must be a power of 2 (1, 2, 4, 8). Splits model layers across GPUs.</small>
            </div>
            <div class="config-field">
              <label>Max Batch Size (--max-batch-size)</label>
              <InputNumber v-model="formState.max_batch_size" :min="1" :max="128" :step="1" />
              <small class="field-help">Maximum number of concurrent requests processed in a single batch. Higher values improve throughput but increase latency.</small>
            </div>
          </div>
        </div>

        <div class="config-section">
          <h4>Precision & Backend</h4>
          <div class="config-grid">
            <div class="config-field">
              <label>DType (--dtype)</label>
              <Dropdown v-model="formState.dtype" :options="dtypeOptions" optionLabel="label" optionValue="value" />
              <small class="field-help">Data type for model weights and activations. Auto selects FP16 for FP32/FP16 models, BF16 for BF16 models. Ignored for quantized models.</small>
            </div>
            <div class="config-field">
              <label>Model Format (--model-format)</label>
              <Dropdown v-model="formState.model_format" :options="modelFormatOptions" optionLabel="label" optionValue="value" placeholder="Auto-detect" />
              <small class="field-help">Model quantization format. Leave empty for auto-detection. Required for AWQ, GPTQ, FP8, or MXFP4 quantized models.</small>
            </div>
            <div class="config-field">
              <label>Quant Policy (--quant-policy)</label>
              <Dropdown v-model="formState.quant_policy" :options="quantPolicyOptions" optionLabel="label" optionValue="value" />
              <small class="field-help">KV cache quantization: 0 = no quantization, 4 = 4-bit KV cache, 8 = 8-bit KV cache. Reduces memory usage at cost of slight accuracy.</small>
            </div>
            <div class="config-field">
              <label>Communicator (--communicator)</label>
              <Dropdown v-model="formState.communicator" :options="communicatorOptions" optionLabel="label" optionValue="value" />
              <small class="field-help">Multi-GPU communication backend. NCCL (recommended) for most setups. CUDA-IPC can be faster for same-node NVLink-connected GPUs.</small>
            </div>
          </div>
        </div>

        <div class="config-section">
          <h4>Cache & Performance</h4>
          <div class="config-grid">
            <div class="config-field">
              <label>Cache Max Entry (--cache-max-entry-count)</label>
              <InputNumber v-model="formState.cache_max_entry_count" :min="0.1" :max="1" :step="0.05" mode="decimal" />
              <small class="field-help">Percentage of free GPU memory used for KV cache (excluding model weights). Higher values allow longer contexts but reduce available memory. Default: 0.8 (80%)</small>
            </div>
            <div class="config-field">
              <label>Cache Block Seq Len (--cache-block-seq-len)</label>
              <InputNumber v-model="formState.cache_block_seq_len" :min="32" :max="2048" :step="32" />
              <small class="field-help">Token sequence length per KV cache block. Must be multiple of 32 for compute capability ≥8.0, or 64 otherwise. Default: 64</small>
            </div>
            <div class="config-field switch-field">
              <div class="switch-label-group">
                <label>Prefix Caching (--enable-prefix-caching)</label>
                <InputSwitch v-model="formState.enable_prefix_caching" />
              </div>
              <small class="field-help">Enable prefix matching and caching. Reuses cached KV for common prompt prefixes, improving performance for repeated prompts.</small>
            </div>
            <div class="config-field">
              <label>Tokens Per Iteration (--num-tokens-per-iter)</label>
              <InputNumber v-model="formState.num_tokens_per_iter" :min="0" :max="262144" :step="64" />
              <small class="field-help">Number of tokens processed in a single forward pass. 0 = auto-detect. Higher values increase throughput but use more memory.</small>
            </div>
            <div class="config-field">
              <label>Max Prefill Iterations (--max-prefill-iters)</label>
              <InputNumber v-model="formState.max_prefill_iters" :min="1" :max="16" :step="1" />
              <small class="field-help">Maximum number of forward passes during prefill stage. Higher values allow processing longer prompts in fewer iterations. Default: 1</small>
            </div>
            <div class="config-field switch-field">
              <div class="switch-label-group">
                <label>Enable Metrics (--enable-metrics)</label>
                <InputSwitch v-model="formState.enable_metrics" />
              </div>
              <small class="field-help">Enable performance metrics collection. Provides detailed timing and throughput statistics for monitoring and optimization.</small>
            </div>
          </div>
        </div>

        <div class="config-section">
          <h4>Server Configuration</h4>
          <div class="config-grid">
            <div class="config-field">
              <label>Model Name (--model-name)</label>
              <InputText v-model="formState.model_name" placeholder="Optional model identifier" />
              <small class="field-help">Model name for OpenAI-style /v1/models listing. Leave empty for auto-generated name.</small>
            </div>
            <div class="config-field">
              <label>Log Level (--log-level)</label>
              <Dropdown v-model="formState.log_level" :options="logLevelOptions" optionLabel="label" optionValue="value" placeholder="None (Default)" />
              <small class="field-help">Logging verbosity level. Select None to use LMDeploy's default.</small>
            </div>
            <div class="config-field">
              <label>Max Concurrent Requests (--max-concurrent-requests)</label>
              <InputNumber v-model="formState.max_concurrent_requests" :min="1" :step="1" :allowEmpty="true" />
              <small class="field-help">Maximum number of concurrent API requests. Leave empty to use default.</small>
            </div>
            <div class="config-field">
              <label>API Keys (--api-keys)</label>
              <InputText v-model="formState.api_keys" placeholder="key1,key2,key3" />
              <small class="field-help">Comma-separated list of API keys for authentication. Leave empty to disable.</small>
            </div>
            <div class="config-field">
              <label>Proxy URL (--proxy-url)</label>
              <InputText v-model="formState.proxy_url" placeholder="http://proxy.example.com" />
              <small class="field-help">Proxy URL for requests. Leave empty for no proxy.</small>
            </div>
            <div class="config-field">
              <label>Allow Origins (--allow-origins)</label>
              <InputText v-model="formState.allow_origins" placeholder="origin1,origin2" />
              <small class="field-help">Comma-separated list of allowed CORS origins. Leave empty for default.</small>
            </div>
            <div class="config-field">
              <label>Allow Methods (--allow-methods)</label>
              <InputText v-model="formState.allow_methods" placeholder="GET,POST,OPTIONS" />
              <small class="field-help">Comma-separated list of allowed HTTP methods. Leave empty for default.</small>
            </div>
            <div class="config-field">
              <label>Allow Headers (--allow-headers)</label>
              <InputText v-model="formState.allow_headers" placeholder="header1,header2" />
              <small class="field-help">Comma-separated list of allowed CORS headers. Leave empty for default.</small>
            </div>
            <div class="config-field">
              <label>Max Log Length (--max-log-len)</label>
              <InputNumber v-model="formState.max_log_len" :min="1" :step="1" :allowEmpty="true" />
              <small class="field-help">Maximum log message length. Leave empty to use default.</small>
            </div>
            <div class="config-field switch-field">
              <div class="switch-label-group">
                <label>Allow Credentials (--allow-credentials)</label>
                <InputSwitch v-model="formState.allow_credentials" />
              </div>
              <small class="field-help">Allow credentials in CORS requests.</small>
            </div>
            <div class="config-field switch-field">
              <div class="switch-label-group">
                <label>SSL (--ssl)</label>
                <InputSwitch v-model="formState.ssl" />
              </div>
              <small class="field-help">Enable SSL/TLS for the API server.</small>
            </div>
            <div class="config-field switch-field">
              <div class="switch-label-group">
                <label>Disable FastAPI Docs (--disable-fastapi-docs)</label>
                <InputSwitch v-model="formState.disable_fastapi_docs" />
              </div>
              <small class="field-help">Disable FastAPI automatic documentation endpoints.</small>
            </div>
            <div class="config-field switch-field">
              <div class="switch-label-group">
                <label>Allow Terminate by Client (--allow-terminate-by-client)</label>
                <InputSwitch v-model="formState.allow_terminate_by_client" />
              </div>
              <small class="field-help">Allow clients to terminate requests.</small>
            </div>
            <div class="config-field switch-field">
              <div class="switch-label-group">
                <label>Enable Abort Handling (--enable-abort-handling)</label>
                <InputSwitch v-model="formState.enable_abort_handling" />
              </div>
              <small class="field-help">Enable handling of aborted requests.</small>
            </div>
          </div>
        </div>

        <div class="config-section">
          <h4>Model Configuration</h4>
          <div class="config-grid">
            <div class="config-field">
              <label>Device (--device)</label>
              <Dropdown v-model="formState.device" :options="deviceOptions" optionLabel="label" optionValue="value" placeholder="None (Default: cuda)" />
              <small class="field-help">Target device for model execution. Select None to use default (cuda).</small>
            </div>
            <div class="config-field">
              <label>Chat Template (--chat-template)</label>
              <InputText v-model="formState.chat_template" placeholder="Path to chat template file" />
              <small class="field-help">Path to custom chat template file. Leave empty for default.</small>
            </div>
            <div class="config-field">
              <label>Tool Call Parser (--tool-call-parser)</label>
              <InputText v-model="formState.tool_call_parser" placeholder="Parser name or path" />
              <small class="field-help">Tool call parser configuration. Leave empty for default.</small>
            </div>
            <div class="config-field">
              <label>Reasoning Parser (--reasoning-parser)</label>
              <InputText v-model="formState.reasoning_parser" placeholder="Parser name or path" />
              <small class="field-help">Reasoning parser configuration. Leave empty for default.</small>
            </div>
            <div class="config-field">
              <label>Revision (--revision)</label>
              <InputText v-model="formState.revision" placeholder="git revision or branch" />
              <small class="field-help">Model revision/branch to use. Leave empty for default.</small>
            </div>
            <div class="config-field">
              <label>Download Dir (--download-dir)</label>
              <InputText v-model="formState.download_dir" placeholder="/path/to/downloads" />
              <small class="field-help">Directory for model downloads. Leave empty for default.</small>
            </div>
            <div class="config-field">
              <label>Adapters (--adapters)</label>
              <InputText v-model="formState.adapters" placeholder="adapter1,adapter2" />
              <small class="field-help">Comma-separated list of adapter paths (LoRA, etc.). Leave empty for none.</small>
            </div>
            <div class="config-field">
              <label>Logprobs Mode (--logprobs-mode)</label>
              <Dropdown v-model="formState.logprobs_mode" :options="logprobsModeOptions" optionLabel="label" optionValue="value" placeholder="None" />
              <small class="field-help">Log probabilities output mode. Leave empty for None.</small>
            </div>
            <div class="config-field switch-field">
              <div class="switch-label-group">
                <label>Eager Mode (--eager-mode)</label>
                <InputSwitch v-model="formState.eager_mode" />
              </div>
              <small class="field-help">Enable eager execution mode.</small>
            </div>
            <div class="config-field switch-field">
              <div class="switch-label-group">
                <label>Disable Vision Encoder (--disable-vision-encoder)</label>
                <InputSwitch v-model="formState.disable_vision_encoder" />
              </div>
              <small class="field-help">Disable vision encoder for vision-language models.</small>
            </div>
          </div>
        </div>

        <div class="config-section">
          <h4>Vision</h4>
          <div class="config-grid">
            <div class="config-field">
              <label>Vision Max Batch Size (--vision-max-batch-size)</label>
              <InputNumber v-model="formState.vision_max_batch_size" :min="1" :step="1" :allowEmpty="true" />
              <small class="field-help">Maximum batch size for vision-related tasks. Default: 1. Leave empty to use default.</small>
            </div>
          </div>
        </div>

        <div class="config-section">
          <h4>Speculative Decoding</h4>
          <div class="config-grid">
            <div class="config-field">
              <label>Speculative Algorithm (--speculative-algorithm)</label>
              <Dropdown v-model="formState.speculative_algorithm" :options="speculativeAlgorithmOptions" optionLabel="label" optionValue="value" placeholder="None" />
              <small class="field-help">Speculative decoding algorithm. Leave empty to disable.</small>
            </div>
            <div class="config-field">
              <label>Speculative Draft Model (--speculative-draft-model)</label>
              <InputText v-model="formState.speculative_draft_model" placeholder="Path to draft model" />
              <small class="field-help">Path to draft model for speculative decoding. Required if algorithm is set.</small>
            </div>
            <div class="config-field">
              <label>Speculative Num Draft Tokens (--speculative-num-draft-tokens)</label>
              <InputNumber v-model="formState.speculative_num_draft_tokens" :min="1" :step="1" :allowEmpty="true" />
              <small class="field-help">Number of draft tokens for speculative decoding. Leave empty to use default.</small>
            </div>
          </div>
        </div>

        <div class="config-section">
          <h4>Distributed / Multi-node</h4>
          <div class="config-grid">
            <div class="config-field">
              <label>Data Parallelism (--dp)</label>
              <InputNumber v-model="formState.dp" :min="1" :step="1" :allowEmpty="true" />
              <small class="field-help">Data parallelism degree. Leave empty to use default (1).</small>
            </div>
            <div class="config-field">
              <label>Expert Parallelism (--ep)</label>
              <InputNumber v-model="formState.ep" :min="1" :step="1" :allowEmpty="true" />
              <small class="field-help">Expert parallelism degree. Leave empty to use default (1).</small>
            </div>
            <div class="config-field">
              <label>Role (--role)</label>
              <Dropdown v-model="formState.role" :options="roleOptions" optionLabel="label" optionValue="value" placeholder="None (Default: Hybrid)" />
              <small class="field-help">Node role in distributed setup. Select None to use default (Hybrid).</small>
            </div>
            <div class="config-field">
              <label>Node Rank (--node-rank)</label>
              <InputNumber v-model="formState.node_rank" :min="0" :step="1" :allowEmpty="true" />
              <small class="field-help">Rank of this node in distributed setup. Leave empty to use default (0).</small>
            </div>
            <div class="config-field">
              <label>Number of Nodes (--nnodes)</label>
              <InputNumber v-model="formState.nnodes" :min="1" :step="1" :allowEmpty="true" />
              <small class="field-help">Total number of nodes. Leave empty to use default (1).</small>
            </div>
            <div class="config-field">
              <label>CP (--cp)</label>
              <InputNumber v-model="formState.cp" :min="1" :step="1" :allowEmpty="true" />
              <small class="field-help">Checkpoint parallelism. Leave empty to use default (1).</small>
            </div>
            <div class="config-field">
              <label>Distributed Executor Backend (--distributed-executor-backend)</label>
              <Dropdown v-model="formState.distributed_executor_backend" :options="distributedExecutorBackendOptions" optionLabel="label" optionValue="value" placeholder="None" />
              <small class="field-help">Backend for distributed execution. Leave empty for default.</small>
            </div>
            <div class="config-field">
              <label>Migration Backend (--migration-backend)</label>
              <Dropdown v-model="formState.migration_backend" :options="migrationBackendOptions" optionLabel="label" optionValue="value" placeholder="None" />
              <small class="field-help">Migration backend for distributed setup. Leave empty for default.</small>
            </div>
            <div class="config-field switch-field">
              <div class="switch-label-group">
                <label>Enable Microbatch (--enable-microbatch)</label>
                <InputSwitch v-model="formState.enable_microbatch" />
              </div>
              <small class="field-help">Enable microbatch processing.</small>
            </div>
            <div class="config-field switch-field">
              <div class="switch-label-group">
                <label>Enable EPLB (--enable-eplb)</label>
                <InputSwitch v-model="formState.enable_eplb" />
              </div>
              <small class="field-help">Enable expert parallelism load balancing.</small>
            </div>
            <div class="config-field switch-field">
              <div class="switch-label-group">
                <label>Enable Return Routed Experts (--enable-return-routed-experts)</label>
                <InputSwitch v-model="formState.enable_return_routed_experts" />
              </div>
              <small class="field-help">Enable return routed experts for MoE models.</small>
            </div>
          </div>
        </div>

        <div class="config-section">
          <h4>DLLM (Diffusion LLM) - Advanced</h4>
          <div class="config-grid">
            <div class="config-field">
              <label>DLLM Block Length (--dllm-block-length)</label>
              <InputNumber v-model="formState.dllm_block_length" :min="1" :step="1" :allowEmpty="true" />
              <small class="field-help">Block length for DLLM. Leave empty to use default.</small>
            </div>
            <div class="config-field">
              <label>DLLM Unmasking Strategy (--dllm-unmasking-strategy)</label>
              <Dropdown v-model="formState.dllm_unmasking_strategy" :options="dllmUnmaskingStrategyOptions" optionLabel="label" optionValue="value" placeholder="None" />
              <small class="field-help">Unmasking strategy for DLLM. Leave empty for default.</small>
            </div>
            <div class="config-field">
              <label>DLLM Denoising Steps (--dllm-denoising-steps)</label>
              <InputNumber v-model="formState.dllm_denoising_steps" :min="1" :step="1" :allowEmpty="true" />
              <small class="field-help">Number of denoising steps for DLLM. Leave empty to use default.</small>
            </div>
            <div class="config-field">
              <label>DLLM Confidence Threshold (--dllm-confidence-threshold)</label>
              <InputNumber v-model="formState.dllm_confidence_threshold" :min="0" :max="1" :step="0.01" mode="decimal" :allowEmpty="true" />
              <small class="field-help">Confidence threshold for DLLM (0.0-1.0). Leave empty to use default.</small>
            </div>
          </div>
        </div>

        <div class="config-section">
          <h4>Advanced</h4>
          <div class="config-grid">
            <div class="config-field span-2">
              <label>Additional CLI Arguments</label>
              <InputText v-model="formState.additional_args" placeholder="--custom-flag value" />
              <small class="field-help">Additional command-line arguments passed directly to LMDeploy. Use for experimental or version-specific flags not exposed in the UI.</small>
            </div>
          </div>
        </div>

        <Divider />

        <div class="metadata-section">
          <div>
            <h4>Model Metadata</h4>
            <ul>
              <li v-if="metadata.architecture"><strong>Architecture:</strong> {{ metadata.architecture }}</li>
              <li v-if="metadata.base_model"><strong>Base Model:</strong> {{ metadata.base_model }}</li>
              <li v-if="metadata.pipeline_tag"><strong>Pipeline:</strong> {{ metadata.pipeline_tag }}</li>
              <li v-if="metadata.parameters"><strong>Parameters:</strong> {{ metadata.parameters }}</li>
            </ul>
          </div>
          <div v-if="dtypeEntries.length" class="dtype-panel">
            <h4>Tensor dtypes</h4>
            <div class="dtype-tags">
              <Tag 
                v-for="item in dtypeEntries" 
                :key="item.label" 
                :value="`${item.label}: ${item.value.toLocaleString()}`"
                severity="secondary"
              />
            </div>
          </div>
        </div>
      </div>
      <div v-else class="loading-state">
        <i class="pi pi-spin pi-spinner"></i>
        <span>Loading configuration...</span>
      </div>

      <template #footer>
        <div class="dialog-footer">
          <Button 
            label="Save Config" 
            icon="pi pi-save"
            severity="secondary"
            :loading="savingConfig"
            @click="saveConfig"
          />
          <Button 
            v-if="selectedModelRunning"
            label="Stop"
            icon="pi pi-stop"
            severity="danger"
            :loading="dialogStopping"
            @click="stopRuntime()"
          />
          <Button 
            v-else
            label="Start LMDeploy"
            icon="pi pi-play"
            severity="success"
            :disabled="!lmdeployReady || !!lmdeployOperation"
            :loading="dialogStarting"
            @click="startRuntime"
          />
          <Button label="Close" text severity="secondary" @click="dialogVisible = false" />
        </div>
      </template>
    </Dialog>
  </div>
</template>

<script setup>
import { computed, ref, reactive, watch, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import Button from 'primevue/button'
import Dialog from 'primevue/dialog'
import Slider from 'primevue/slider'
import InputNumber from 'primevue/inputnumber'
import InputText from 'primevue/inputtext'
import InputSwitch from 'primevue/inputswitch'
import Dropdown from 'primevue/dropdown'
import Divider from 'primevue/divider'
import Tag from 'primevue/tag'
import { toast } from 'vue3-toastify'
import { useModelStore } from '@/stores/models'
import axios from 'axios'
import { formatFileSize, formatDate } from '@/utils/formatting'

const router = useRouter()
const props = defineProps({
  models: {
    type: Array,
    default: () => []
  },
  loading: {
    type: Boolean,
    default: false
  }
})

defineEmits(['refresh', 'delete'])

const modelStore = useModelStore()
const dialogVisible = ref(false)
const selectedModel = ref(null)
const savingConfig = ref(false)
const reloadingFromDisk = ref(false)

const dtypeOptions = [
  { label: 'Auto', value: 'auto' },
  { label: 'float16', value: 'float16' },
  { label: 'bfloat16', value: 'bfloat16' }
]
const modelFormatOptions = [
  { label: 'Auto', value: '' },
  { label: 'HF', value: 'hf' },
  { label: 'AWQ', value: 'awq' },
  { label: 'GPTQ', value: 'gptq' },
  { label: 'FP8', value: 'fp8' },
  { label: 'MXFP4', value: 'mxfp4' }
]
const quantPolicyOptions = [
  { label: '0 • No kv quant', value: 0 },
  { label: '4 • 4-bit kv', value: 4 },
  { label: '8 • 8-bit kv', value: 8 }
]
const communicatorOptions = [
  { label: 'NCCL', value: 'nccl' },
  { label: 'Native', value: 'native' },
  { label: 'CUDA IPC', value: 'cuda-ipc' }
]
const logLevelOptions = [
  { label: 'None (Default)', value: null },
  { label: 'CRITICAL', value: 'CRITICAL' },
  { label: 'FATAL', value: 'FATAL' },
  { label: 'ERROR', value: 'ERROR' },
  { label: 'WARN', value: 'WARN' },
  { label: 'WARNING', value: 'WARNING' },
  { label: 'INFO', value: 'INFO' },
  { label: 'DEBUG', value: 'DEBUG' },
  { label: 'NOTSET', value: 'NOTSET' }
]
const deviceOptions = [
  { label: 'None (Default: cuda)', value: null },
  { label: 'CUDA', value: 'cuda' },
  { label: 'Ascend', value: 'ascend' },
  { label: 'MACA', value: 'maca' },
  { label: 'CAMB', value: 'camb' }
]
const logprobsModeOptions = [
  { label: 'None', value: null },
  { label: 'Raw Logits', value: 'raw_logits' },
  { label: 'Raw Logprobs', value: 'raw_logprobs' }
]
const dllmUnmaskingStrategyOptions = [
  { label: 'None (Default: low_confidence_dynamic)', value: null },
  { label: 'Low Confidence Dynamic', value: 'low_confidence_dynamic' },
  { label: 'Low Confidence Static', value: 'low_confidence_static' },
  { label: 'Sequential', value: 'sequential' }
]
const roleOptions = [
  { label: 'None (Default: Hybrid)', value: null },
  { label: 'Hybrid', value: 'Hybrid' },
  { label: 'Prefill', value: 'Prefill' },
  { label: 'Decode', value: 'Decode' }
]
const migrationBackendOptions = [
  { label: 'None (Default: DLSlime)', value: null },
  { label: 'DLSlime', value: 'DLSlime' },
  { label: 'Mooncake', value: 'Mooncake' }
]
const distributedExecutorBackendOptions = [
  { label: 'None (Auto-select)', value: null },
  { label: 'Uni', value: 'uni' },
  { label: 'MP', value: 'mp' },
  { label: 'Ray', value: 'ray' }
]
const speculativeAlgorithmOptions = [
  { label: 'None (Disabled)', value: null },
  { label: 'Eagle', value: 'eagle' },
  { label: 'Eagle3', value: 'eagle3' },
  { label: 'DeepSeek MTP', value: 'deepseek_mtp' }
]

const formState = reactive({
  session_len: 4096,
  max_prefill_token_num: 8192,
  tensor_parallel: 1,
  max_batch_size: 4,
  dtype: 'auto',
  cache_max_entry_count: 0.8,
  cache_block_seq_len: 64,
  enable_prefix_caching: false,
  quant_policy: 0,
  model_format: '',
  enable_metrics: true,
  rope_scaling_mode: 'disabled',
  rope_scaling_factor: 1,
  hf_override_rope_type: '',
  hf_override_rope_factor: null,
  hf_override_rope_original_max: null,
  num_tokens_per_iter: 0,
  max_prefill_iters: 1,
  communicator: 'nccl',
  model_name: '',
  // Server configuration
  allow_origins: '',
  allow_credentials: false,
  allow_methods: '',
  allow_headers: '',
  proxy_url: '',
  max_concurrent_requests: null,
  log_level: null,
  api_keys: '',
  ssl: false,
  max_log_len: null,
  disable_fastapi_docs: false,
  allow_terminate_by_client: false,
  enable_abort_handling: false,
  // Model configuration
  chat_template: '',
  tool_call_parser: '',
  reasoning_parser: '',
  revision: '',
  download_dir: '',
  adapters: '',
  device: null,
  eager_mode: false,
  disable_vision_encoder: false,
  logprobs_mode: null,
  // DLLM parameters
  dllm_block_length: null,
  dllm_unmasking_strategy: null,
  dllm_denoising_steps: null,
  dllm_confidence_threshold: null,
  // Distributed/Multi-node parameters
  dp: null,
  ep: null,
  enable_microbatch: false,
  enable_eplb: false,
  role: null,
  migration_backend: null,
  node_rank: null,
  nnodes: null,
  cp: null,
  enable_return_routed_experts: false,
  distributed_executor_backend: null,
  // Vision parameters
  vision_max_batch_size: null,
  // Speculative decoding parameters
  speculative_algorithm: null,
  speculative_draft_model: null,
  speculative_num_draft_tokens: null,
  additional_args: ''
})

const getEntryModelId = (entry) => entry?.model_id ?? entry?.modelId ?? entry?.id

const groupedModels = computed(() => {
  if (!Array.isArray(props.models)) return []
  return [...props.models].sort((a, b) => {
    const aDate = new Date(a.latest_downloaded_at || 0).getTime()
    const bDate = new Date(b.latest_downloaded_at || 0).getTime()
    return bDate - aDate
  })
})

const statusLoading = computed(() => modelStore.lmdeployStatusLoading)

const currentInstanceId = computed(() => modelStore.lmdeployStatus?.running_instance?.model_id)
const installerStatus = computed(() => modelStore.lmdeployStatus?.installer || null)
const lmdeployReady = computed(() => !!installerStatus.value?.installed)
const lmdeployOperation = computed(() => installerStatus.value?.operation || null)
const selectedModelId = computed(() => getEntryModelId(selectedModel.value))
const selectedRuntime = computed(() => {
  const id = selectedModelId.value
  if (!id) return null
  return modelStore.safetensorsRuntime[id]
})
const metadata = computed(() => selectedRuntime.value?.metadata || {})

// Base context length reported by metadata (typically the trained / sliding-window
// context). This is informative only; users can exceed it via RoPE / YaRN scaling.
const baseContextLength = computed(() => {
  const runtime = selectedRuntime.value
  if (!runtime) return 0
  return (
    runtime.max_context_length ||
    runtime.metadata?.max_context_length ||
    runtime.metadata?.context_length ||
    0
  )
})

const SESSION_FALLBACK_LIMIT = 256000
const MAX_SCALING_FACTOR = 4
const ropeScalingOptions = [
  { label: 'Disabled', value: 'disabled' },
  { label: 'YaRN (recommended)', value: 'yarn' },
  { label: 'Generic scaling', value: 'generic' },
]

const isQwen3 = computed(() => {
  const runtime = selectedRuntime.value
  if (!runtime) return false
  const config = runtime.metadata?.config || {}
  const modelType = (config.model_type || '').toLowerCase()
  const huggingfaceId = (selectedModel.value?.huggingface_id || '').toLowerCase()
  return modelType.includes('qwen3') || huggingfaceId.includes('qwen3')
})

// Model max length from tokenizer_config.json (clamps RoPE scaling)
// Model max length from metadata (required for rope scaling)
const modelMaxLength = computed(() => {
  const runtime = selectedRuntime.value
  if (!runtime) return null
  return runtime.metadata?.model_max_length || null
})

// Max position embeddings from config
const maxPositionEmbeddings = computed(() => {
  const runtime = selectedRuntime.value
  if (!runtime) return null
  const config = runtime.metadata?.config || {}
  return config.max_position_embeddings || null
})

// Check if scaling should be available
const canUseScaling = computed(() => {
  const baseLimit = Number(baseContextLength.value) || 0
  if (baseLimit <= 0) return false
  
  // Allow scaling if we have base context
  return true
})

// Warning if model_max_length is missing
const scalingWarning = computed(() => {
  if (!modelMaxLength.value && canUseScaling.value) {
    return "RoPE scaling is not recommended without model_max_length. Use max_position_embeddings as fallback."
  }
  return null
})

// Adapted base context for scaling (model_max_length / 4 when model_max_length > max_position_embeddings)
const adaptedBaseContext = computed(() => {
  const modelMax = modelMaxLength.value
  const maxPos = maxPositionEmbeddings.value
  if (modelMax && maxPos && modelMax > maxPos) {
    // If model_max_length > max_position_embeddings, it means rope scaling can achieve model_max_length
    // Adapt base context to model_max_length / 4 (allows 4x scaling to reach model_max_length)
    return Math.floor(modelMax / 4)
  }
  return null
})

const sessionLimit = computed(() => {
  const baseLimit = Number(baseContextLength.value) || 0
  if (baseLimit > 0) {
    return baseLimit
  }
  return SESSION_FALLBACK_LIMIT
})
const scalingEnabled = computed(() => {
  const mode = (formState.rope_scaling_mode || '').toLowerCase()
  return canUseScaling.value && mode !== '' && mode !== 'disabled'
})
const effectiveContextLength = computed(() => {
  const base = Number(formState.session_len) || 0
  if (base <= 0) {
    return 0
  }
  if (!scalingEnabled.value) {
    return base
  }
  const rawFactor = Number(formState.rope_scaling_factor) || 1
  const clampedFactor = Math.min(Math.max(rawFactor, 1), MAX_SCALING_FACTOR)
  let effective = Math.round(base * clampedFactor)
  // Clamp to model_max_length if available, otherwise max_position_embeddings
  if (modelMaxLength.value && effective > modelMaxLength.value) {
    effective = modelMaxLength.value
  } else if (maxPositionEmbeddings.value && effective > maxPositionEmbeddings.value) {
    effective = maxPositionEmbeddings.value
  }
  return effective
})
const metadataRefreshing = computed(() => {
  const id = selectedModelId.value
  if (!id) return false
  return !!modelStore.safetensorsMetadataRefreshing[id]
})
const dtypeEntries = computed(() => {
  const summary = selectedRuntime.value?.tensor_summary?.dtype_counts || {}
  return Object.entries(summary).map(([label, value]) => ({ label, value }))
})
const selectedModelRunning = computed(() => {
  if (!selectedModelId.value || !currentInstanceId.value) return false
  return currentInstanceId.value === selectedModelId.value
})

watch(sessionLimit, (limit) => {
  const maxLimit = Number(limit) || 0
  if (!maxLimit) return
  if (formState.session_len > maxLimit) {
    formState.session_len = maxLimit
  }
})

watch(
  () => [scalingEnabled.value, sessionLimit.value, adaptedBaseContext.value],
  ([enabled, limit, adaptedBase]) => {
    if (!enabled) return
    // Use adapted base context if available (model_max_length / 4), otherwise use session limit
    const targetLimit = adaptedBase && adaptedBase >= 1024 ? adaptedBase : Number(limit) || 0
    if (!targetLimit) return
    if (formState.session_len !== targetLimit) {
      formState.session_len = targetLimit
    }
    // Auto-set hf_override_rope_original_max to adapted base when scaling is enabled
    if (adaptedBase && adaptedBase >= 1024) {
      formState.hf_override_rope_original_max = adaptedBase
    }
  }
)

watch(
  () => formState.rope_scaling_mode,
  (mode) => {
    if (!canUseScaling.value) {
      if (mode !== 'disabled') {
        formState.rope_scaling_mode = 'disabled'
      }
      if (formState.rope_scaling_factor !== 1) {
        formState.rope_scaling_factor = 1
      }
      return
    }
    if (mode && mode !== 'disabled') {
      if (formState.rope_scaling_factor <= 1) {
        formState.rope_scaling_factor = Math.min(2, MAX_SCALING_FACTOR)
      }
      const limit = Number(sessionLimit.value) || 0
      if (limit && formState.session_len !== limit) {
        formState.session_len = limit
      }
    } else if (formState.rope_scaling_factor !== 1) {
      formState.rope_scaling_factor = 1
    }
  }
)

watch(
  () => formState.rope_scaling_factor,
  (factor) => {
    if (factor > MAX_SCALING_FACTOR) {
      formState.rope_scaling_factor = MAX_SCALING_FACTOR
    } else if (factor < 1) {
      formState.rope_scaling_factor = 1
    }
  }
)



const isConfigLoading = (entry) => {
  const id = getEntryModelId(entry)
  return !!modelStore.safetensorsRuntimeLoading[id]
}

const isGroupConfigLoading = (group) => {
  if (!group?.files?.length) return false
  return group.files.some(file => isConfigLoading(file))
}

const isStopping = (entry) => {
  const id = getEntryModelId(entry)
  return !!modelStore.lmdeployStopping[id]
}

const isGroupStopping = (group) => {
  if (!group?.files?.length) return false
  return group.files.some(file => isStopping(file))
}

const openGroupConfig = (group) => {
  if (!group) return
  // Pass the unified group directly - it has all the necessary info
  openConfig(group)
}

const stopGroupRuntime = (group) => {
  if (!group?.files?.length) return
  // Use the group's model_id to stop the runtime
  const groupModelId = group?.model_id
  if (groupModelId) {
    stopRuntime({ model_id: groupModelId })
  } else {
    stopRuntime(group.files[0])
  }
}

const dialogStarting = computed(() => {
  if (!selectedModelId.value) return false
  return !!modelStore.lmdeployStarting[selectedModelId.value]
})
const dialogStopping = computed(() => {
  if (!selectedModelId.value) return false
  return !!modelStore.lmdeployStopping[selectedModelId.value]
})

const refreshStatus = async () => {
  try {
    await modelStore.fetchLmdeployStatus()
  } catch (error) {
    console.error(error)
  }
}

const reloadFromDisk = async () => {
  if (reloadingFromDisk.value) return
  
  const confirmed = confirm(
    'This will reset all safetensors database entries and reload them from disk storage.\n\n' +
    'This action cannot be undone. Continue?'
  )
  if (!confirmed) return
  
  reloadingFromDisk.value = true
  try {
    const response = await axios.post('/api/models/safetensors/reload-from-disk')
    const result = response.data
    toast.success(
      `Reloaded ${result.reloaded} safetensors models from disk` +
      (result.error_count ? ` (${result.error_count} errors)` : '')
    )
    if (result.errors && result.errors.length > 0) {
      console.error('Reload errors:', result.errors)
    }
    // Refresh the model list
    await modelStore.fetchSafetensorsModels()
  } catch (error) {
    console.error('Failed to reload safetensors from disk:', error)
    toast.error(error.response?.data?.detail || 'Failed to reload safetensors from disk')
  } finally {
    reloadingFromDisk.value = false
  }
}

const regenerateMetadata = async () => {
  const modelId = selectedModelId.value
  if (!modelId) return
  try {
    await modelStore.regenerateSafetensorsMetadata(modelId)
    toast.success('Metadata regenerated')
  } catch (error) {
    console.error(error)
    toast.error('Failed to regenerate metadata')
  }
}

const openConfig = async (model) => {
  selectedModel.value = model
  dialogVisible.value = true
  const modelId = getEntryModelId(model)
  try {
    await modelStore.fetchSafetensorsRuntimeConfig(modelId)
  } catch (error) {
    toast.error('Failed to load LMDeploy config')
  }
}

const applyRuntimeConfig = (config) => {
  if (!config) return
  const normalized = { ...config }
  if (normalized.context_length && normalized.session_len === undefined) {
    normalized.session_len = normalized.context_length
  }
  if (normalized.max_batch_tokens && normalized.max_prefill_token_num === undefined) {
    normalized.max_prefill_token_num = normalized.max_batch_tokens
  }
  Object.keys(formState).forEach((key) => {
    if (normalized[key] !== undefined) {
      // Handle array/string conversion for list fields (store as comma-separated string in formState)
      if (['allow_origins', 'allow_methods', 'allow_headers', 'api_keys', 'adapters'].includes(key)) {
        if (Array.isArray(normalized[key])) {
          formState[key] = normalized[key].join(', ')
        } else if (typeof normalized[key] === 'string') {
          formState[key] = normalized[key]
        } else {
          formState[key] = ''
        }
      } else if (Array.isArray(normalized[key])) {
        formState[key] = [...normalized[key]]
      } else {
        formState[key] = normalized[key]
      }
    }
  })
  hydrateHfOverrideFields(normalized.hf_overrides)
}

watch(selectedRuntime, (runtime) => {
  if (runtime?.config) {
    applyRuntimeConfig(runtime.config)
  }
}, { immediate: true })

function hydrateHfOverrideFields(overrides) {
  const rope = overrides?.rope_scaling || {}
  formState.hf_override_rope_type = rope.rope_type || ''
  const factorCandidate = rope.factor ?? rope.scale ?? null
  formState.hf_override_rope_factor = factorCandidate !== undefined ? factorCandidate : null
  const originalMax = rope.original_max_position_embeddings ?? rope.original_max_position_embedding ?? null
  formState.hf_override_rope_original_max = originalMax !== undefined ? originalMax : null
}

function buildHfOverrides() {
  const overrides = {}
  const rope = {}
  
  // If scaling is enabled and we have adapted base context, use it
  if (scalingEnabled.value && adaptedBaseContext.value && adaptedBaseContext.value >= 1024) {
    rope.original_max_position_embeddings = adaptedBaseContext.value
    // Set rope_type if scaling mode is yarn
    if (formState.rope_scaling_mode === 'yarn') {
      rope.rope_type = 'yarn'
    }
    // Set factor from scaling factor
    if (formState.rope_scaling_factor && Number(formState.rope_scaling_factor) > 1) {
      rope.factor = Number(formState.rope_scaling_factor)
    }
  } else {
    // Use manual overrides if provided
    if (formState.hf_override_rope_type) {
      rope.rope_type = formState.hf_override_rope_type
    }
    if (formState.hf_override_rope_factor && Number(formState.hf_override_rope_factor) > 0) {
      rope.factor = Number(formState.hf_override_rope_factor)
    }
    if (formState.hf_override_rope_original_max && Number(formState.hf_override_rope_original_max) > 0) {
      rope.original_max_position_embeddings = Number(formState.hf_override_rope_original_max)
    }
  }
  
  if (Object.keys(rope).length) {
    overrides.rope_scaling = rope
  }
  return overrides
}

const buildPayload = () => {
  const payload = {
    session_len: formState.session_len,
    max_prefill_token_num: formState.max_prefill_token_num,
    tensor_parallel: formState.tensor_parallel,
    max_batch_size: formState.max_batch_size,
    dtype: formState.dtype,
    cache_max_entry_count: formState.cache_max_entry_count,
    cache_block_seq_len: formState.cache_block_seq_len,
    enable_prefix_caching: formState.enable_prefix_caching,
    quant_policy: formState.quant_policy,
    model_format: formState.model_format,
    hf_overrides: buildHfOverrides(),
    enable_metrics: formState.enable_metrics,
    rope_scaling_mode: formState.rope_scaling_mode,
    rope_scaling_factor: formState.rope_scaling_factor,
    num_tokens_per_iter: formState.num_tokens_per_iter,
    max_prefill_iters: formState.max_prefill_iters,
    communicator: formState.communicator,
    model_name: formState.model_name || null,
    // Server configuration
    allow_origins: formState.allow_origins && formState.allow_origins.length > 0 ? (typeof formState.allow_origins === 'string' ? formState.allow_origins.split(',').map(s => s.trim()).filter(s => s) : formState.allow_origins) : null,
    allow_credentials: formState.allow_credentials || null,
    allow_methods: formState.allow_methods && formState.allow_methods.length > 0 ? (typeof formState.allow_methods === 'string' ? formState.allow_methods.split(',').map(s => s.trim()).filter(s => s) : formState.allow_methods) : null,
    allow_headers: formState.allow_headers && formState.allow_headers.length > 0 ? (typeof formState.allow_headers === 'string' ? formState.allow_headers.split(',').map(s => s.trim()).filter(s => s) : formState.allow_headers) : null,
    proxy_url: formState.proxy_url || null,
    max_concurrent_requests: formState.max_concurrent_requests || null,
    log_level: formState.log_level || null,
    api_keys: formState.api_keys && formState.api_keys.length > 0 ? (typeof formState.api_keys === 'string' ? formState.api_keys.split(',').map(s => s.trim()).filter(s => s) : formState.api_keys) : null,
    ssl: formState.ssl || null,
    max_log_len: formState.max_log_len || null,
    disable_fastapi_docs: formState.disable_fastapi_docs || null,
    allow_terminate_by_client: formState.allow_terminate_by_client || null,
    enable_abort_handling: formState.enable_abort_handling || null,
    // Model configuration
    chat_template: formState.chat_template || null,
    tool_call_parser: formState.tool_call_parser || null,
    reasoning_parser: formState.reasoning_parser || null,
    revision: formState.revision || null,
    download_dir: formState.download_dir || null,
    adapters: formState.adapters && formState.adapters.length > 0 ? (typeof formState.adapters === 'string' ? formState.adapters.split(',').map(s => s.trim()).filter(s => s) : formState.adapters) : null,
    device: formState.device || null,
    eager_mode: formState.eager_mode || null,
    disable_vision_encoder: formState.disable_vision_encoder || null,
    logprobs_mode: formState.logprobs_mode || null,
    // DLLM parameters
    dllm_block_length: formState.dllm_block_length || null,
    dllm_unmasking_strategy: formState.dllm_unmasking_strategy || null,
    dllm_denoising_steps: formState.dllm_denoising_steps || null,
    dllm_confidence_threshold: formState.dllm_confidence_threshold || null,
    // Distributed/Multi-node parameters
    dp: formState.dp || null,
    ep: formState.ep || null,
    enable_microbatch: formState.enable_microbatch || null,
    enable_eplb: formState.enable_eplb || null,
    role: formState.role || null,
    migration_backend: formState.migration_backend || null,
    node_rank: formState.node_rank || null,
    nnodes: formState.nnodes || null,
    cp: formState.cp || null,
    enable_return_routed_experts: formState.enable_return_routed_experts || null,
    distributed_executor_backend: formState.distributed_executor_backend || null,
    // Vision parameters
    vision_max_batch_size: formState.vision_max_batch_size || null,
    // Speculative decoding parameters
    speculative_algorithm: formState.speculative_algorithm || null,
    speculative_draft_model: formState.speculative_draft_model || null,
    speculative_num_draft_tokens: formState.speculative_num_draft_tokens || null,
    additional_args: formState.additional_args
  }
  
  // Remove null/undefined values to keep payload clean
  Object.keys(payload).forEach(key => {
    if (payload[key] === null || payload[key] === undefined || payload[key] === '') {
      delete payload[key]
    }
  })
  
  return payload
}

const saveConfig = async () => {
  if (!selectedModelId.value) return
  savingConfig.value = true
  try {
    await modelStore.updateSafetensorsRuntimeConfig(selectedModelId.value, buildPayload())
    toast.success('LMDeploy config saved')
  } catch (error) {
    toast.error('Failed to save config')
  } finally {
    savingConfig.value = false
  }
}

const startRuntime = async () => {
  if (!selectedModelId.value) return
  if (!lmdeployReady.value) {
    toast.error('Install LMDeploy before starting a runtime')
    return
  }
  if (lmdeployOperation.value) {
    toast.info('LMDeploy installer is running—wait until it finishes')
    return
  }
  try {
    await modelStore.startSafetensorsRuntime(selectedModelId.value, buildPayload())
    toast.success('LMDeploy starting…')
  } catch (error) {
    toast.error('Failed to start LMDeploy')
  }
}

const stopRuntime = async (entry = null) => {
  const targetId = entry ? getEntryModelId(entry) : selectedModelId.value
  if (!targetId) return
  try {
    await modelStore.stopSafetensorsRuntime(targetId)
    toast.success('LMDeploy stopped')
  } catch (error) {
    toast.error('Failed to stop LMDeploy')
  }
}

const isModelRunning = (model) => {
  const modelId = getEntryModelId(model)
  return currentInstanceId.value === modelId
}

const isGroupRunning = (group) => {
  // Check if the group's model_id matches the running instance
  const groupModelId = group?.model_id
  if (!groupModelId || !currentInstanceId.value) return false
  return currentInstanceId.value === groupModelId
}

// Formatting functions are now imported from utils

const openLmdeployPage = () => {
  router.push('/lmdeploy')
}

onMounted(() => {
  refreshStatus()
})
</script>

<style scoped>
.lmdeploy-alert {
  display: flex;
  gap: var(--spacing-md);
  padding: var(--spacing-md);
  background: var(--status-warning-soft);
  border: 1px solid rgba(245, 158, 11, 0.3);
  border-radius: var(--radius-md);
  margin-bottom: var(--spacing-lg);
  align-items: center;
}

.lmdeploy-alert.info {
  background: var(--status-info-soft);
  border-color: rgba(34, 211, 238, 0.3);
}

.alert-icon {
  font-size: 1.5rem;
  color: var(--status-warning);
}

.alert-content h3 {
  margin: 0 0 var(--spacing-xs) 0;
}

.alert-content p {
  margin: 0 0 var(--spacing-sm) 0;
  color: var(--text-secondary);
}

.safetensors-card {
  margin-top: var(--spacing-xl);
  padding: var(--spacing-lg);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  background: var(--bg-card);
  box-shadow: var(--shadow-sm);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: var(--spacing-md);
  gap: var(--spacing-md);
}

.card-header h2 {
  margin: 0;
  font-weight: 600;
  color: var(--text-primary);
}

.subtitle {
  margin: 4px 0 0;
  font-size: 0.9rem;
  color: var(--text-secondary);
}

.actions {
  display: flex;
  gap: var(--spacing-xs);
}

.loading-state,
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-xl);
  color: var(--text-secondary);
  text-align: center;
}

.model-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: var(--spacing-md);
}

.model-card {
  border: 1px solid var(--border-secondary);
  border-radius: var(--radius-xl);
  padding: var(--spacing-md);
  background: var(--bg-surface);
  box-shadow: var(--shadow-sm);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.grouped-card {
  gap: var(--spacing-md);
}

.model-card-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: var(--spacing-sm);
}

.model-name {
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 2px;
}

.model-path {
  font-size: 0.9rem;
  color: var(--text-secondary);
}

.group-header {
  border-bottom: 1px solid var(--border-secondary);
  padding-bottom: var(--spacing-sm);
}

.group-summary {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.group-header-main {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.group-status-row {
  margin-top: var(--spacing-xs);
  margin-bottom: var(--spacing-sm);
}

.group-status-row :deep(.status-indicator) {
  font-size: 0.7rem;
  padding: 2px 6px;
}

.grouped-body {
  padding-top: var(--spacing-sm);
}

.file-list {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.plain-file-list {
  gap: 2px;
}

.file-name-plain {
  font-family: monospace;
  font-size: 0.85rem;
  color: var(--text-secondary);
  word-break: break-all;
}

.file-row {
  border: 1px solid var(--border-secondary);
  border-radius: var(--radius-lg);
  padding: var(--spacing-sm);
  background: var(--bg-surface-2, var(--bg-card));
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  cursor: default;
}

.file-row-header {
  display: flex;
  justify-content: space-between;
  gap: var(--spacing-sm);
}

.file-name {
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 4px;
}

.model-meta {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.meta-row {
  display: flex;
  align-items: center;
  gap: 6px;
}

.endpoint {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--accent-cyan);
  font-size: 0.85rem;
}

.model-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--spacing-sm);
  margin-top: var(--spacing-sm);
}

.group-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: var(--spacing-md);
  gap: var(--spacing-sm);
  flex-wrap: wrap;
}

.config-section {
  border: 1px solid var(--border-secondary);
  border-radius: var(--radius-xl);
  padding: var(--spacing-md);
  background: var(--bg-surface);
  margin-bottom: var(--spacing-lg);
}

.config-section h4 {
  margin: 0 0 var(--spacing-sm);
  font-weight: 600;
  color: var(--text-primary);
}

.action-group {
  display: flex;
  gap: var(--spacing-xs);
}

.dot {
  opacity: 0.6;
}

.dialog-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  gap: var(--spacing-md);
}

.dialog-header h3 {
  margin: 0;
  font-weight: 600;
}

.dialog-header p {
  margin: 2px 0 0;
  color: var(--text-secondary);
  font-size: 0.9rem;
}

.dialog-header-actions {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.config-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: var(--spacing-md);
}

.config-field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.config-field label {
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 2px;
}

.field-help {
  font-size: 0.8rem;
  color: var(--text-secondary);
  line-height: 1.4;
  margin-top: 4px;
  opacity: 0.85;
}

.qwen3-note {
  display: block;
  margin-top: 6px;
  padding: 6px 8px;
  background: var(--bg-secondary);
  border-left: 3px solid var(--accent-cyan);
  border-radius: 4px;
  color: var(--text-primary);
  opacity: 1;
}

.switch-label-group {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  gap: var(--spacing-sm);
}

.switch-field .field-help {
  margin-top: 6px;
}

.slider-row {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  width: 100%;
}

.slider-row :deep(.p-slider) {
  flex: 1 1 auto;
  min-width: 200px;
  max-width: none;
}

.slider-row :deep(.p-slider .p-slider-range) {
  background: var(--accent-cyan);
}

.slider-row :deep(.p-slider .p-slider-handle) {
  border-color: var(--accent-cyan);
}

.slider-row :deep(.p-inputnumber) {
  width: 140px;
  flex-shrink: 0;
  min-width: 140px;
}

.config-field .slider-row ~ * :deep(.p-inputnumber),
.config-field > :not(.slider-row) :deep(.p-inputnumber) {
  width: 100%;
}

.config-field :deep(.p-inputnumber .p-inputnumber-input) {
  padding: 0.5rem;
}

.slider-row :deep(.p-inputnumber .p-inputnumber-input) {
  width: 100%;
}

.config-field :deep(.p-inputtext) {
  width: 100%;
  padding: 0.5rem;
}

.config-field :deep(.p-dropdown) {
  width: 100%;
}

.switch-field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.span-2 {
  grid-column: span 2;
}

@media (max-width: 640px) {
  .span-2 {
    grid-column: span 1;
  }
}

.metadata-section {
  display: flex;
  flex-wrap: wrap;
  gap: var(--spacing-xl);
  margin-top: var(--spacing-lg);
}

.metadata-section ul {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 0.9rem;
}

.rope-scaling-controls {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.rope-mode-dropdown {
  width: 100%;
}

.rope-factor-row {
  align-items: center;
}

.dtype-panel {
  min-width: 220px;
}

.dtype-tags {
  display: flex;
  flex-wrap: wrap;
  gap: var(--spacing-xs);
}

.hf-overrides-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: var(--spacing-sm);
}

.hf-override-field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.hf-override-field .field-label {
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: var(--spacing-sm);
}
</style>
