<template>
  <div class="proxy-page page-shell page-shell--relaxed">
    <PageHeader
      title="Proxy"
      description="Gateway settings, routing, access keys, and inference diagnostics."
    >
      <template #meta>
        <Tag :value="statusLabel" :severity="statusSeverity" />
        <Tag v-if="liveRunning.length" :value="`${liveRunning.length} running`" severity="info" />
        <Tag
          v-if="activeBackend && activeBackend !== configuredBackend"
          value="backend mismatch"
          severity="warn"
        />
      </template>
      <template #actions>
        <Button
          icon="pi pi-refresh"
          text
          severity="secondary"
          :loading="loading"
          aria-label="Refresh proxy status"
          v-tooltip.top="'Refresh'"
          @click="refresh()"
        />
      </template>
    </PageHeader>

    <Message v-if="error" severity="error" :closable="true" @close="error = ''">{{
      error
    }}</Message>
    <Message v-if="!activeBackend && configuredBackend" severity="warn" :closable="false">
      The configured proxy ({{ configuredBackend }}) is not running — inference clients cannot
      connect.
      <Button
        label="Start proxy"
        size="small"
        severity="warn"
        :loading="lifecycleBusy"
        @click="lifecycle('start')"
      />
    </Message>
    <Message
      v-else-if="activeBackend && activeBackend !== configuredBackend"
      severity="warn"
      :closable="false"
    >
      The active proxy is “{{ activeBackend }}”, but configuration names “{{ configuredBackend }}”.
      Switching stops the running proxy, then starts the configured one.
      <Button
        label="Switch backend"
        size="small"
        severity="warn"
        :loading="switching"
        @click="confirmSwitch"
      />
    </Message>
    <Message v-else-if="restartRequired" severity="warn" :closable="false">
      Saved settings changed since the proxy last (re)started. Use “Restart &amp; apply” to activate
      them.
    </Message>

    <LoadingState v-if="loading && !loaded" message="Loading proxy state…" />

    <template v-else>
      <nav class="proxy-tabs" aria-label="Proxy sections" role="tablist">
        <button
          v-for="tab in tabs"
          :id="`proxy-tab-${tab.id}`"
          :key="tab.id"
          role="tab"
          :aria-selected="activeTab === tab.id"
          :aria-controls="`proxy-panel-${tab.id}`"
          :tabindex="activeTab === tab.id ? 0 : -1"
          @click="activeTab = tab.id"
          @keydown="navigateTab($event, tab.id)"
        >
          {{ tab.label }}
        </button>
      </nav>
      <div
        :id="`proxy-panel-${activeTab}`"
        class="proxy-panels"
        role="tabpanel"
        :aria-labelledby="`proxy-tab-${activeTab}`"
      >
        <section
          v-show="activeTab === 'settings'"
          class="proxy-card"
          aria-labelledby="proxy-backend-heading"
        >
          <div class="section-head">
            <div>
              <h2 id="proxy-backend-heading">Backend &amp; serving limits</h2>
              <p>Saved limits and timeouts apply when the native Studio proxy restarts.</p>
            </div>
            <dl class="backend-state">
              <div>
                <dt>Configured</dt>
                <dd>{{ configuredBackend || '—' }}</dd>
              </div>
              <div>
                <dt>Active</dt>
                <dd :class="{ 'state-alert': !activeBackend }">
                  {{ activeBackend || 'not running' }}
                </dd>
              </div>
            </dl>
          </div>
          <div class="settings-grid">
            <label
              >Backend
              <select v-model="draft.backend" class="field">
                <option value="studio">Studio native proxy</option>
                <option value="llama_swap">llama-swap (legacy)</option>
              </select>
            </label>
            <label
              >Inference port
              <input
                v-model.number="draft.proxy_port"
                class="field"
                type="number"
                min="1"
                max="65535"
              />
              <small class="hint">Applies on the next proxy restart.</small>
            </label>
            <label v-for="field in performanceFields" :key="field.name"
              >{{ field.label }}
              <input
                v-model.number="draft[field.name]"
                class="field"
                type="number"
                :min="field.min || 1"
                :step="field.step || 1"
                :placeholder="field.placeholder"
              />
              <small class="hint">{{ field.hint }}</small>
            </label>
            <label
              >Idle unload (seconds)
              <input
                v-model.number="draft.idle_ttl_seconds"
                class="field"
                type="number"
                min="0"
                placeholder="Disabled"
              />
            </label>
            <label
              >Maximum request size (bytes)
              <input
                v-model.number="draft.max_body_bytes"
                class="field"
                type="number"
                min="1"
                placeholder="268435456"
              />
            </label>
            <label
              >Maximum loaded models
              <input
                v-model.number="draft.max_loaded_models"
                class="field"
                type="number"
                min="1"
                placeholder="Unlimited"
              />
            </label>
            <label
              >Global concurrent requests
              <input
                v-model.number="draft.global_concurrency"
                class="field"
                type="number"
                min="1"
                placeholder="Unlimited"
              />
            </label>
            <label
              >Concurrent requests per model
              <input
                v-model.number="draft.per_model_concurrency"
                class="field"
                type="number"
                min="1"
                placeholder="Unlimited"
              />
            </label>
            <label
              >Queue timeout (seconds)
              <input
                v-model.number="draft.queue_timeout_seconds"
                class="field"
                type="number"
                min="0.1"
                step="0.1"
              />
            </label>
            <label
              >Total request deadline (seconds)
              <input
                v-model.number="draft.request_timeout_seconds"
                class="field"
                type="number"
                min="0.1"
                placeholder="Unlimited"
              />
            </label>
            <label
              >Engine readiness timeout (seconds)
              <input
                v-model.number="draft.readiness_timeout_seconds"
                class="field"
                type="number"
                min="1"
              />
            </label>
            <label
              >Graceful drain timeout (seconds)
              <input
                v-model.number="draft.drain_timeout_seconds"
                class="field"
                type="number"
                min="1"
              />
            </label>
          </div>
          <div class="inline-control">
            <ToggleSwitch v-model="draft.allow_anonymous" input-id="allow-anonymous" />
            <label for="allow-anonymous"
              >Allow anonymous inference while no access key is configured
              <small class="hint"
                >Once any key exists, anonymous traffic is rejected regardless of this
                setting.</small
              >
            </label>
          </div>
          <div class="runtime-actions">
            <span class="save-group">
              <Button
                label="Save settings"
                :disabled="!dirty"
                :loading="saving"
                @click="saveSettings"
              />
              <Button
                v-if="dirty"
                label="Discard changes"
                size="small"
                severity="secondary"
                text
                @click="resetDraft"
              />
            </span>
            <span class="lifecycle-group">
              <Button
                label="Start"
                icon="pi pi-play"
                size="small"
                severity="success"
                outlined
                :disabled="Boolean(activeBackend)"
                @click="lifecycle('start')"
              />
              <Button
                label="Restart &amp; apply"
                icon="pi pi-replay"
                size="small"
                severity="secondary"
                outlined
                :loading="lifecycleBusy"
                @click="confirmRestart"
              />
              <Button
                label="Stop"
                icon="pi pi-stop-circle"
                size="small"
                severity="danger"
                outlined
                :disabled="!activeBackend"
                @click="confirmStop"
              />
            </span>
          </div>
        </section>

        <section
          v-show="activeTab === 'settings'"
          class="proxy-card"
          aria-labelledby="proxy-routing-heading"
        >
          <div class="section-head">
            <div>
              <h2 id="proxy-routing-heading">Routing</h2>
              <p>
                Routing changes apply immediately without restarting engines or interrupting
                streams. Default model for requests without a model field, plus aliases. An alias
                with several targets is round-robined.
              </p>
            </div>
          </div>
          <div class="settings-grid">
            <label
              >Default model
              <select v-model="draft.default_model" class="field">
                <option :value="null">Automatic — single running model</option>
                <option v-for="model in catalog" :key="model.id" :value="model.id">
                  {{ model.id }}
                </option>
              </select>
            </label>
          </div>
          <div class="alias-list">
            <div v-for="(row, index) in aliasRows" :key="index" class="alias-row">
              <input
                v-model="row.alias"
                class="field"
                placeholder="Alias, for example fast"
                aria-label="Alias name"
              />
              <input
                v-model="row.targets"
                class="field"
                placeholder="Target model IDs, comma-separated"
                aria-label="Alias targets"
              />
              <Button
                icon="pi pi-times"
                text
                rounded
                severity="danger"
                size="small"
                aria-label="Remove alias"
                v-tooltip.top="'Remove alias'"
                @click="aliasRows.splice(index, 1)"
              />
            </div>
            <div>
              <Button
                label="Add alias"
                icon="pi pi-plus"
                size="small"
                severity="secondary"
                outlined
                @click="aliasRows.push({ alias: '', targets: '' })"
              />
            </div>
          </div>
        </section>

        <section
          v-show="activeTab === 'keys'"
          class="proxy-card"
          aria-labelledby="proxy-keys-heading"
        >
          <div class="section-head">
            <div>
              <h2 id="proxy-keys-heading">Client access keys</h2>
              <p>
                Set model access, request rates, token budgets and concurrency per client. Secrets
                are shown once.
              </p>
            </div>
            <span class="count">{{ keys.length }} active</span>
          </div>
          <Message v-if="configuredBackend !== 'studio'" severity="warn" :closable="false"
            >Key rate limits, token budgets and concurrency policies are enforced by the Studio
            native proxy. Switch to Studio to activate them.</Message
          >
          <div class="settings-grid key-options">
            <label
              >Token accounting<select
                v-model="keyAccounting"
                class="field"
                aria-label="Token accounting"
              >
                <option value="auto">Engine counts when available; text estimate otherwise</option>
                <option value="exact">Require exact engine input counts</option>
              </select></label
            >
            <label
              >Key name
              <input
                ref="keyNameInput"
                v-model="keyName"
                class="field"
                maxlength="120"
                placeholder="for example Production"
                @keyup.enter="createKey"
              />
            </label>
            <label
              >Expires at
              <input v-model="keyExpires" class="field" type="datetime-local" />
            </label>
            <label
              >Requests per minute
              <input
                v-model.number="keyRateLimit"
                class="field"
                type="number"
                min="1"
                placeholder="Unlimited"
              />
            </label>
            <label
              >Allowed model IDs
              <input
                v-model="keyModels"
                class="field"
                placeholder="All models, or comma-separated IDs"
              />
            </label>
            <label v-for="field in policyFields" :key="field.name"
              >{{ field.label }}
              <input
                v-model.number="keyPolicy[field.name]"
                class="field"
                type="number"
                min="1"
                step="1"
                placeholder="Unlimited"
              />
            </label>
          </div>
          <p class="hint">
            Token budgets use the engine’s input-token counter when supported and reserve the
            requested output allowance before generation. Auto mode uses a visible byte estimate for
            text when counting is unavailable; exact mode refuses that fallback. Actual engine usage
            replaces the reservation on completion; missing usage retains the estimate. Daily and
            monthly budgets reset at midnight UTC. Text inference only when token limits are
            enabled.
          </p>
          <div>
            <Button
              label="Create key"
              icon="pi pi-key"
              :disabled="!keyName.trim()"
              :loading="creating"
              @click="createKey"
            />
          </div>
          <div v-if="createdKey" class="secret" role="status">
            <strong>{{ createdKeyLabel }}</strong>
            <code>{{ createdKey }}</code>
            <Button
              label="Copy"
              icon="pi pi-copy"
              size="small"
              severity="secondary"
              outlined
              @click="copyKey"
            />
          </div>
          <div v-if="keys.length" class="key-list">
            <div v-for="key in keys" :key="key.id" class="key-row">
              <div class="key-summary">
                <span class="key-title"
                  ><strong>{{ key.name }}</strong
                  ><Tag :value="keyStatus(key)" :severity="keySeverity(key)"
                /></span>
                <code>{{ key.prefix }}…</code>
                <small
                  >{{ key.models?.length ? key.models.join(', ') : 'All models' }} ·
                  {{
                    key.rate_limit_per_minute
                      ? `${key.rate_limit_per_minute} req/min`
                      : 'no rate limit'
                  }}
                  · {{ key.expires_at ? `expires ${formatDate(key.expires_at)}` : 'no expiry' }} ·
                  {{
                    key.usage?.last_used_at
                      ? `last used ${formatTime(key.usage.last_used_at)}`
                      : key.last_used_at
                        ? `last used ${formatAgo(key.last_used_at)}`
                        : 'never used'
                  }}</small
                >
              </div>
              <dl class="key-usage" v-if="key.usage">
                <div>
                  <dt>Today</dt>
                  <dd>
                    {{ key.usage.daily_tokens.toLocaleString() }} /
                    {{ allowance(key.daily_token_budget) }} tokens
                  </dd>
                  <small
                    >{{
                      key.usage.daily_remaining == null
                        ? 'Unlimited'
                        : `${key.usage.daily_remaining.toLocaleString()} remaining`
                    }}
                    · resets {{ formatTime(key.usage.day_resets_at) }}</small
                  ><ProgressBar
                    v-if="key.daily_token_budget"
                    :value="Math.min(100, (key.usage.daily_tokens / key.daily_token_budget) * 100)"
                    :show-value="false"
                  />
                </div>
                <div>
                  <dt>This month</dt>
                  <dd>
                    {{ key.usage.monthly_tokens.toLocaleString() }} /
                    {{ allowance(key.monthly_token_budget) }} tokens
                  </dd>
                  <small
                    >{{
                      key.usage.monthly_remaining == null
                        ? 'Unlimited'
                        : `${key.usage.monthly_remaining.toLocaleString()} remaining`
                    }}
                    · resets {{ formatTime(key.usage.month_resets_at) }}</small
                  ><ProgressBar
                    v-if="key.monthly_token_budget"
                    :value="
                      Math.min(100, (key.usage.monthly_tokens / key.monthly_token_budget) * 100)
                    "
                    :show-value="false"
                  />
                </div>
                <div>
                  <dt>Last minute</dt>
                  <dd>
                    {{ key.usage.requests_per_minute }} /
                    {{ allowance(key.rate_limit_per_minute) }} requests
                  </dd>
                  <small
                    >{{ key.usage.input_tokens_per_minute }} /
                    {{ allowance(key.input_tokens_per_minute) }} input ·
                    {{ key.usage.output_tokens_per_minute }} /
                    {{ allowance(key.output_tokens_per_minute) }} output tokens</small
                  >
                </div>
                <div>
                  <dt>In flight</dt>
                  <dd>
                    {{ key.usage.concurrent_requests }} /
                    {{ allowance(key.max_concurrent_requests) }}
                  </dd>
                  <small
                    >Output cap: {{ allowance(key.max_output_tokens) }} per request ·
                    {{ key.usage.estimated_requests }} requests charged by estimate</small
                  >
                </div>
              </dl>
              <span class="key-actions">
                <Button
                  label="Edit policy"
                  size="small"
                  severity="secondary"
                  text
                  @click="editKey(key)"
                />
                <Button
                  label="Rotate"
                  size="small"
                  severity="warn"
                  text
                  @click="confirmRotate(key)"
                />
                <Button
                  label="Revoke"
                  size="small"
                  severity="danger"
                  text
                  @click="confirmRevoke(key)"
                />
              </span>
            </div>
          </div>
          <EmptyState
            v-else
            icon="pi pi-key"
            title="No access keys yet"
            description="Without a key the inference port follows the anonymous-access policy above. Create a key to lock it down."
          >
            <Button
              label="Create the first key"
              icon="pi pi-key"
              size="small"
              @click="focusKeyForm"
            />
          </EmptyState>
          <Dialog
            v-model:visible="editingKeyVisible"
            header="Edit key policy"
            modal
            :style="{ width: '45rem', maxWidth: '95vw' }"
          >
            <p>Changes affect new requests immediately. The existing secret remains valid.</p>
            <div class="settings-grid">
              <label
                >Key name<input v-model="editPolicy.name" class="field" maxlength="120"
              /></label>
              <label
                >Expires at<input v-model="editPolicy.expires" class="field" type="datetime-local"
              /></label>
              <label
                >Allowed model IDs<input
                  v-model="editPolicy.models"
                  class="field"
                  placeholder="All models, or comma-separated IDs"
              /></label>
              <label
                >Requests per minute<input
                  v-model.number="editPolicy.rate_limit_per_minute"
                  class="field"
                  type="number"
                  min="1"
                  placeholder="Unlimited"
              /></label>
              <label v-for="field in policyFields" :key="field.name"
                >{{ field.label
                }}<input
                  v-model.number="editPolicy[field.name]"
                  class="field"
                  type="number"
                  min="1"
                  placeholder="Unlimited"
              /></label>
            </div>
            <label class="accounting-mode"
              >Token accounting<select
                v-model="editPolicy.token_accounting"
                class="field"
                aria-label="Token accounting"
              >
                <option value="auto">Engine counts when available; text estimate otherwise</option>
                <option value="exact">Require exact engine input counts</option>
              </select></label
            >
            <fieldset class="scope-options">
              <legend>Permissions</legend>
              <label
                ><input v-model="editPolicy.scopes" type="checkbox" value="inference" /> Run
                inference</label
              ><label
                ><input v-model="editPolicy.scopes" type="checkbox" value="models:read" /> List
                models</label
              >
            </fieldset>
            <Message v-if="policyError" severity="error" :closable="false">{{
              policyError
            }}</Message>
            <div class="runtime-actions">
              <Button
                label="Save policy"
                :loading="savingPolicy"
                :disabled="!editPolicy.name?.trim()"
                @click="saveKeyPolicy"
              /><Button label="Cancel" text severity="secondary" @click="editingKey = null" />
            </div>
          </Dialog>
        </section>

        <section
          v-show="activeTab === 'overview'"
          class="proxy-card"
          aria-labelledby="proxy-running-heading"
        >
          <div class="section-head">
            <div>
              <h2 id="proxy-running-heading">Engine diagnostics</h2>
              <p>
                Process generations supervised by the native proxy. Logs stream live while a panel
                is open.
              </p>
            </div>
            <span class="count">{{ liveRunning.length }} running</span
            ><Button
              label="Reset stats"
              icon="pi pi-refresh"
              size="small"
              severity="secondary"
              outlined
              @click="resetStats"
            />
          </div>
          <dl class="proxy-summary metrics" aria-label="Gateway statistics">
            <div>
              <dt>Requests</dt>
              <dd>{{ summary.requests || 0 }}</dd>
            </div>
            <div>
              <dt>Active / queued</dt>
              <dd>{{ summary.active || 0 }} / {{ summary.queued || 0 }}</dd>
            </div>
            <div>
              <dt>Loading / streaming</dt>
              <dd>{{ summary.loading || 0 }} / {{ summary.streaming || 0 }}</dd>
            </div>
            <div>
              <dt>Latency p50 / p95</dt>
              <dd>{{ metric(summary.p50_latency_ms) }} / {{ metric(summary.p95_latency_ms) }}</dd>
            </div>
            <div>
              <dt>Errors</dt>
              <dd>{{ summary.errors || 0 }}</dd>
            </div>
            <div>
              <dt>Tokens in / out</dt>
              <dd>{{ summary.input_tokens || 0 }} / {{ summary.output_tokens || 0 }}</dd>
            </div>
            <div>
              <dt>Unknown usage</dt>
              <dd>{{ summary.unknown_usage || 0 }}</dd>
            </div>
            <div>
              <dt>Measured since</dt>
              <dd>{{ summary.since ? formatTime(summary.since) : '—' }}</dd>
            </div>
          </dl>
          <Message v-if="summary.dropped_history" severity="warn" :closable="false"
            >{{ summary.dropped_history }} request records could not be retained while history
            storage was unavailable.</Message
          >
          <small class="hint"
            >Reset stats clears the observation window. Key budgets and request history are
            preserved.</small
          >
          <div v-if="running.length" class="runtime-list">
            <article v-for="item in running" :key="`${item.model}-${item.pid}`" class="runtime">
              <div class="runtime-head">
                <div>
                  <strong>{{ item.model }}</strong
                  ><span
                    >{{ item.engine || 'engine' }} · PID {{ item.pid }} · port {{ item.port }}</span
                  >
                </div>
                <Tag
                  :value="item.state"
                  :severity="item.state === 'failed' ? 'danger' : item.ready ? 'success' : 'warn'"
                />
                <Message v-if="item.failure_reason" severity="error" :closable="false"
                  >{{ item.failure_reason }}. Start the model from Models to retry.</Message
                >
              </div>
              <ProgressBar
                v-if="!item.ready && item.loading_progress != null"
                :value="Math.round(item.loading_progress)"
                :show-value="false"
                aria-label="Model loading progress"
              />
              <dl class="metrics">
                <div>
                  <dt>In-flight</dt>
                  <dd>{{ item.inflight }}</dd>
                </div>
                <div>
                  <dt>Requests</dt>
                  <dd>{{ item.request_count || 0 }}</dd>
                </div>
                <div>
                  <dt>Errors</dt>
                  <dd :class="{ 'state-error': item.error_count }">{{ item.error_count || 0 }}</dd>
                </div>
                <div>
                  <dt>Latency p50 / p95</dt>
                  <dd>{{ metric(item.p50_latency_ms) }} · {{ metric(item.p95_latency_ms) }}</dd>
                </div>
                <div>
                  <dt>Tokens out</dt>
                  <dd>
                    {{ item.output_tokens || 0 }} ({{ rate(item.output_tokens_per_second) }} tok/s)
                  </dd>
                </div>
                <div>
                  <dt>Traffic</dt>
                  <dd>
                    {{ formatBytes(item.input_bytes) }} in ·
                    {{ formatBytes(item.output_bytes) }} out
                  </dd>
                </div>
                <div>
                  <dt>Revision</dt>
                  <dd>
                    <code>{{ shortRevision(item.revision) }}</code>
                  </dd>
                </div>
                <div>
                  <dt>Started</dt>
                  <dd>{{ formatTime(item.created_at) }}</dd>
                </div>
              </dl>
              <div class="runtime-actions">
                <Button
                  :label="openLogs === item.model ? 'Hide logs' : 'View logs'"
                  :icon="openLogs === item.model ? 'pi pi-eye-slash' : 'pi pi-list'"
                  size="small"
                  severity="secondary"
                  outlined
                  :aria-expanded="openLogs === item.model"
                  @click="toggleLogs(item.model)"
                />
                <RouterLink :to="modelLocation(item.model)" class="model-link"
                  >Manage model <i class="pi pi-arrow-up-right" aria-hidden="true"
                /></RouterLink>
              </div>
              <template v-if="openLogs === item.model">
                <div class="log-toolbar">
                  <span class="log-search">
                    <i class="pi pi-search" aria-hidden="true" />
                    <input
                      v-model="logSearch"
                      class="field"
                      placeholder="Search log lines…"
                      aria-label="Search logs"
                      @input="debouncedLoadLogs(item.model)"
                    />
                  </span>
                  <select
                    v-model="logLevel"
                    class="field"
                    aria-label="Log severity"
                    @change="loadLogs(item.model)"
                  >
                    <option value="">All severities</option>
                    <option value="warn">Warnings &amp; errors</option>
                    <option value="error">Errors only</option>
                  </select>
                  <span class="log-meta">{{ (logs[item.model] || []).length }} shown · live</span>
                  <Button
                    label="Download"
                    icon="pi pi-download"
                    size="small"
                    severity="secondary"
                    text
                    @click="downloadLogs(item.model)"
                  />
                  <Button
                    label="Clear"
                    icon="pi pi-trash"
                    size="small"
                    severity="danger"
                    text
                    @click="confirmClearLogs(item.model)"
                  />
                </div>
                <LogOutput
                  :lines="logs[item.model] || []"
                  empty-message="No engine output matches the current filter."
                />
              </template>
            </article>
          </div>
          <EmptyState
            v-else
            icon="pi pi-play"
            title="No engine processes running"
            description="Active engines appear here after inference traffic or starting a model from Models."
          />
        </section>

        <section
          v-show="activeTab === 'settings'"
          class="proxy-card"
          aria-labelledby="proxy-network-heading"
        >
          <div class="section-head">
            <div>
              <h2 id="proxy-network-heading">Network</h2>
              <p>
                Read-only view of how the inference port is bound. Listen host comes from
                STUDIO_PROXY_LISTEN; terminate TLS at a reverse proxy.
              </p>
            </div>
          </div>
          <dl class="network">
            <div>
              <dt>Listen host</dt>
              <dd>
                <code>{{ network.listen_host || '—' }}</code>
                <small>{{ network.listen_source }}</small>
              </dd>
            </div>
            <div>
              <dt>Port</dt>
              <dd>{{ network.port ?? '—' }}</dd>
            </div>
            <div>
              <dt>Public URL</dt>
              <dd>{{ network.public_url || 'Same origin (default)' }}</dd>
            </div>
            <div>
              <dt>TLS</dt>
              <dd>{{ network.tls_note }}</dd>
            </div>
          </dl>
        </section>

        <section
          v-show="activeTab === 'events'"
          class="proxy-card"
          aria-labelledby="proxy-activity-heading"
        >
          <div class="section-head">
            <div>
              <h2 id="proxy-activity-heading">Recent activity</h2>
              <p>
                Lifecycle events from the native proxy, newest first. Refreshes every few seconds.
              </p>
            </div>
          </div>
          <ul v-if="activity.length" class="activity-list">
            <li
              v-for="(item, index) in activityDesc"
              :key="`${item.type}-${item.created_at}-${index}`"
            >
              <span class="activity-time">{{ formatTime(item.timestamp) }}</span>
              <Tag :value="item.type" :severity="activitySeverity(item.type)" />
              <span class="activity-text">{{ activityText(item) }}</span>
            </li>
          </ul>
          <p v-else class="empty">No proxy activity recorded in this session.</p>
        </section>
        <section v-show="activeTab === 'requests'" class="proxy-card">
          <div class="section-head">
            <div>
              <h2>Requests</h2>
              <p>
                Live and retained requests, newest first. Up to 100 per page. Rejections and
                cancellations are included.
              </p>
            </div>
            <span class="count">{{ requestTotal }} matching</span>
          </div>
          <form class="request-filters settings-grid" @submit.prevent="loadRequests(true)">
            <label
              >Search<input
                v-model="requestFilters.search"
                class="field"
                placeholder="Request ID, alias, path or key name"
            /></label>
            <label
              >Model<input
                v-model="requestFilters.model"
                class="field"
                placeholder="Resolved model ID"
            /></label>
            <label
              >Client key<select v-model="requestFilters.key_id" class="field">
                <option value="">All clients</option>
                <option v-for="key in keys" :key="key.id" :value="key.id">{{ key.name }}</option>
                <option value="anonymous">Anonymous</option>
                <option value="internal">Studio</option>
              </select></label
            >
            <label
              >State<select v-model="requestFilters.state" class="field">
                <option value="">All states</option>
                <option v-for="state in requestStates" :key="state" :value="state">
                  {{ state }}
                </option>
              </select></label
            >
            <label
              >HTTP status<input
                v-model="requestFilters.status"
                class="field"
                type="number"
                min="100"
                max="599"
                placeholder="Any"
            /></label>
            <Button
              label="Apply filters"
              size="small"
              :loading="requestsLoading"
              @click="loadRequests(true)"
            />
          </form>
          <Message v-if="requestsError" severity="error" :closable="false">{{
            requestsError
          }}</Message>
          <div class="request-table">
            <table>
              <thead>
                <tr>
                  <th>Started</th>
                  <th>Client / model</th>
                  <th>Method / path</th>
                  <th>State / status</th>
                  <th>Queue / load</th>
                  <th>TTFT / total</th>
                  <th>Tokens in / out</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="request in requests" :key="request.id">
                  <td>{{ formatTime(request.started_at || request.completed_at) }}</td>
                  <td>
                    {{ request.key_name || 'Unidentified' }}<br />{{
                      request.model || request.requested_model || '—'
                    }}
                  </td>
                  <td>{{ request.method }} {{ request.path }}</td>
                  <td>
                    <Tag
                      :value="request.state || 'completed'"
                      :severity="requestSeverity(request)"
                    /><br />{{ request.status || 'In progress'
                    }}<small v-if="request.queue_position">
                      · queue #{{ request.queue_position }}</small
                    >
                  </td>
                  <td>{{ metric(request.queue_ms) }} / {{ metric(request.loading_ms) }}</td>
                  <td>{{ metric(request.ttft_ms) }} / {{ metric(request.latency_ms) }}</td>
                  <td>
                    {{ request.input_tokens ?? '—' }} / {{ request.output_tokens ?? '—'
                    }}<small class="usage-source">{{ request.usage_source || 'unknown' }}</small>
                  </td>
                  <td>
                    <Button label="Details" size="small" text @click="selectedRequest = request" />
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <p v-if="!requests.length" class="empty">No requests match these filters.</p>
          <div class="runtime-actions">
            <Button
              label="Previous 100"
              severity="secondary"
              outlined
              :disabled="requestOffset === 0 || requestsLoading"
              @click="changeRequestPage(-100)"
            /><span
              >{{ requestTotal ? requestOffset + 1 : 0 }}–{{
                Math.min(requestOffset + requests.length, requestTotal)
              }}
              of {{ requestTotal }}</span
            ><Button
              label="Next 100"
              severity="secondary"
              outlined
              :disabled="requestOffset + 100 >= requestTotal || requestsLoading"
              @click="changeRequestPage(100)"
            />
          </div>
          <Dialog
            v-model:visible="requestDialog"
            header="Request details"
            modal
            :style="{ width: '45rem', maxWidth: '95vw' }"
          >
            <dl class="request-details">
              <div v-for="field in requestDetailFields" :key="field.name">
                <dt>{{ field.label }}</dt>
                <dd>{{ requestDetail(field.name) }}</dd>
              </div>
            </dl>
            <p class="hint">
              Usage marked “reservation” is a conservative charge retained because the engine did
              not report final usage. Unknown token counts remain blank. Request and response bodies
              are not retained.
            </p>
          </Dialog>
        </section>
      </div>
    </template>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import axios from 'axios'
import { RouterLink } from 'vue-router'
import { useConfirm } from 'primevue/useconfirm'
import { useToast } from 'primevue/usetoast'
import Button from 'primevue/button'
import Dialog from 'primevue/dialog'
import LogOutput from '@/components/common/LogOutput.vue'
import Tag from 'primevue/tag'
import Message from 'primevue/message'
import ProgressBar from 'primevue/progressbar'
import ToggleSwitch from 'primevue/toggleswitch'
import PageHeader from '@/components/common/PageHeader.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import LoadingState from '@/components/common/LoadingState.vue'
import { requireSingleConfirmation } from '@/composables/singleConfirm'

const tabs = [
  { id: 'overview', label: 'Engine diagnostics' },
  { id: 'requests', label: 'Requests' },
  { id: 'events', label: 'Events' },
  { id: 'keys', label: 'Access keys' },
  { id: 'settings', label: 'Settings & routing' },
]
const activeTab = ref('overview'),
  requests = ref([]),
  selectedRequest = ref(null)
const requestDialog = computed({
  get: () => Boolean(selectedRequest.value),
  set: (value) => {
    if (!value) selectedRequest.value = null
  },
})
function navigateTab(event, id) {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
  event.preventDefault()
  const index = tabs.findIndex((tab) => tab.id === id)
  activeTab.value =
    tabs[
      event.key === 'Home'
        ? 0
        : event.key === 'End'
          ? tabs.length - 1
          : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length
    ].id
  document.getElementById(`proxy-tab-${activeTab.value}`)?.focus()
}
async function resetStats() {
  try {
    await axios.post('/api/proxy/stats/reset')
    await refresh({ quiet: true })
  } catch (err) {
    reportFailure('Resetting stats failed', err)
  }
}
const policyFields = [
  { name: 'input_tokens_per_minute', label: 'Input tokens per minute' },
  { name: 'output_tokens_per_minute', label: 'Output tokens per minute' },
  { name: 'daily_token_budget', label: 'Daily token budget (UTC)' },
  { name: 'monthly_token_budget', label: 'Monthly token budget (UTC)' },
  { name: 'max_output_tokens', label: 'Maximum output tokens per request' },
  { name: 'max_concurrent_requests', label: 'Concurrent requests' },
]
const performanceFields = [
  {
    name: 'max_queue_size',
    label: 'Maximum queued requests',
    hint: 'Excess requests receive HTTP 429.',
  },
  {
    name: 'stream_idle_timeout_seconds',
    label: 'Stream inactivity timeout (seconds)',
    step: 0.1,
    min: 0.1,
    placeholder: 'Unlimited',
    hint: 'Maximum wait between upstream chunks.',
  },
  {
    name: 'pool_timeout_seconds',
    label: 'Connection pool wait (seconds)',
    step: 0.1,
    min: 0.1,
    hint: 'Bounds the wait for an upstream connection.',
  },
  {
    name: 'max_connections',
    label: 'Maximum upstream connections',
    hint: 'Shared across running engines.',
  },
  {
    name: 'max_keepalive_connections',
    label: 'Idle keepalive connections',
    hint: 'Must not exceed total connections.',
  },
  {
    name: 'history_retention_days',
    label: 'Request history retention (days)',
    hint: 'Metadata only; older entries are removed automatically.',
  },
  {
    name: 'history_max_rows',
    label: 'Maximum retained request records',
    min: 100,
    hint: 'A second bound on history disk usage.',
  },
  {
    name: 'default_output_tokens',
    label: 'Reserved output tokens when unspecified',
    hint: 'Applied to text requests made with token-budgeted keys.',
  },
]
const keyAccounting = ref('auto')
const keyPolicy = reactive(Object.fromEntries(policyFields.map((field) => [field.name, null])))
const editingKey = ref(null),
  editPolicy = reactive({ scopes: [] }),
  policyError = ref(''),
  savingPolicy = ref(false)
const editingKeyVisible = computed({
  get: () => Boolean(editingKey.value),
  set: (value) => {
    if (!value) editingKey.value = null
  },
})
const summary = ref({}),
  requestTotal = ref(0),
  requestOffset = ref(0),
  requestsLoading = ref(false),
  requestsError = ref('')
const requestFilters = reactive({ search: '', model: '', key_id: '', state: '', status: '' })
const requestStates = [
  'received',
  'queued',
  'loading',
  'forwarding',
  'counting',
  'streaming',
  'completed',
  'rejected',
  'failed',
  'cancelled',
]
const requestDetailFields = [
  ['id', 'Request ID'],
  ['state', 'State'],
  ['status', 'Outcome status'],
  ['http_status', 'Delivered HTTP status'],
  ['key_name', 'Client'],
  ['key_id', 'Key ID'],
  ['requested_model', 'Requested model / alias'],
  ['model', 'Resolved model'],
  ['routing_revision', 'Routing revision'],
  ['engine', 'Engine'],
  ['pid', 'Process ID'],
  ['method', 'Method'],
  ['path', 'Path'],
  ['started_at', 'Started'],
  ['completed_at', 'Completed'],
  ['queue_position', 'Queue position'],
  ['token_count_ms', 'Input token counting'],
  ['input_reserved_source', 'Input accounting source'],
  ['token_count_endpoint', 'Engine counting endpoint'],
  ['admission_ms', 'Key admission'],
  ['accounting_mode', 'Accounting mode'],
  ['queue_ms', 'Queue wait'],
  ['loading_ms', 'Model acquisition'],
  ['first_byte_ms', 'Time to first byte'],
  ['ttft_ms', 'Time to first token'],
  ['generation_ms', 'Generation time'],
  ['response_streaming_ms', 'Response streaming time'],
  ['output_tokens_per_second', 'Output tokens per second'],
  ['latency_ms', 'Total duration'],
  ['input_tokens', 'Input tokens'],
  ['output_tokens', 'Output tokens'],
  ['observed_output_tokens', 'Partial output tokens observed'],
  ['usage_complete', 'Complete protocol usage'],
  ['usage_source', 'Usage source'],
  ['input_reserved', 'Input token reservation'],
  ['output_reserved', 'Output token reservation'],
  ['input_bytes', 'Request bytes'],
  ['output_bytes', 'Response bytes'],
  ['error', 'Failure reason'],
  ['accounting_note', 'Accounting note'],
].map(([name, label]) => ({ name, label }))
function requestDetail(name) {
  const value = selectedRequest.value?.[name]
  if (value == null || value === '') return '—'
  if (name.endsWith('_at')) return formatTime(value)
  if (name.endsWith('_ms')) return metric(value)
  return value
}
function requestSeverity(request) {
  return (
    {
      completed: 'success',
      rejected: 'warn',
      cancelled: 'warn',
      failed: 'danger',
      streaming: 'info',
    }[request.state] || 'secondary'
  )
}
function allowance(value) {
  return value == null ? 'unlimited' : Number(value).toLocaleString()
}
function editKey(key) {
  editingKey.value = key.id
  Object.assign(editPolicy, {
    ...key,
    token_accounting: key.token_accounting || 'auto',
    models: (key.models || []).join(', '),
    scopes: [...(key.scopes || ['inference'])],
    expires: key.expires_at
      ? new Date(
          new Date(key.expires_at).getTime() - new Date(key.expires_at).getTimezoneOffset() * 60000,
        )
          .toISOString()
          .slice(0, 16)
      : '',
  })
  policyError.value = ''
}
async function saveKeyPolicy() {
  savingPolicy.value = true
  try {
    await axios.patch(`/api/inference/keys/${encodeURIComponent(editingKey.value)}`, {
      token_accounting: editPolicy.token_accounting || 'auto',
      name: editPolicy.name.trim(),
      models: editPolicy.models
        .split(',')
        .map((model) => model.trim())
        .filter(Boolean),
      scopes: editPolicy.scopes,
      expires_at: editPolicy.expires ? new Date(editPolicy.expires).toISOString() : null,
      rate_limit_per_minute: policyNumber(editPolicy.rate_limit_per_minute),
      ...Object.fromEntries(
        policyFields.map((field) => [field.name, policyNumber(editPolicy[field.name])]),
      ),
    })
    editingKey.value = null
    toast.add({ severity: 'success', summary: 'Key policy updated', life: 2500 })
    await refresh({ quiet: true })
  } catch (err) {
    policyError.value = err?.response?.data?.detail || err.message
  } finally {
    savingPolicy.value = false
  }
}
let requestSequence = 0
async function loadRequests(reset = false) {
  if (reset) requestOffset.value = 0
  const sequence = ++requestSequence
  requestsLoading.value = true
  try {
    const params = {
      limit: 100,
      offset: requestOffset.value,
      ...Object.fromEntries(Object.entries(requestFilters).filter(([, value]) => value !== '')),
    }
    const { data } = await axios.get('/api/proxy/requests', { params })
    if (sequence !== requestSequence) return
    requests.value = data.requests || []
    requestTotal.value = data.total || 0
    requestsError.value = ''
    if (selectedRequest.value)
      selectedRequest.value =
        requests.value.find((request) => request.id === selectedRequest.value.id) ||
        selectedRequest.value
  } catch (err) {
    if (sequence === requestSequence)
      requestsError.value = err?.response?.data?.detail || err.message
  } finally {
    if (sequence === requestSequence) requestsLoading.value = false
  }
}
function changeRequestPage(delta) {
  requestOffset.value = Math.max(0, requestOffset.value + delta)
  loadRequests()
}
watch(
  () => activeTab.value,
  (tab) => {
    if (tab === 'requests' && loaded.value) loadRequests()
  },
)
const toast = useToast()
const confirm = useConfirm()
const loading = ref(false),
  saving = ref(false),
  creating = ref(false),
  switching = ref(false),
  lifecycleBusy = ref(false)
const loaded = ref(false)
const error = ref(''),
  restartRequired = ref(false),
  keys = ref([]),
  running = ref([]),
  activity = ref([])
const catalog = ref([]),
  network = ref({})
const configuredBackend = ref(''),
  activeBackend = ref('')
const keyName = ref(''),
  createdKey = ref(''),
  createdKeyLabel = ref('Copy this key now — it will not be shown again.'),
  openLogs = ref('')
const logs = reactive({})
const logSearch = ref(''),
  logLevel = ref(''),
  keyNameInput = ref(null)
const keyExpires = ref(''),
  keyRateLimit = ref(null),
  keyModels = ref('')
const aliasRows = ref([])
const draft = reactive({
  max_queue_size: 100,
  stream_idle_timeout_seconds: null,
  pool_timeout_seconds: 10,
  max_connections: 200,
  max_keepalive_connections: 40,
  history_retention_days: 7,
  history_max_rows: 100000,
  default_output_tokens: 4096,
  backend: 'llama_swap',
  proxy_port: 2000,
  idle_ttl_seconds: null,
  max_body_bytes: null,
  allow_anonymous: true,
  max_loaded_models: null,
  global_concurrency: null,
  per_model_concurrency: null,
  queue_timeout_seconds: 30,
  request_timeout_seconds: null,
  readiness_timeout_seconds: 2700,
  drain_timeout_seconds: 30,
  default_model: null,
})
let timer, logTimer
const baseline = ref('')

const liveRunning = computed(() =>
  running.value.filter((item) => !['failed', 'stopped'].includes(item.state)),
)
function policyNumber(value) {
  return value === '' || value == null ? null : value
}
const statusLabel = computed(() =>
  activeBackend.value ? `Proxy active · ${activeBackend.value}` : 'Proxy not running',
)
const statusSeverity = computed(() => (activeBackend.value ? 'success' : 'danger'))

const activityDesc = computed(() => [...activity.value].reverse())
function activitySeverity(type) {
  return (
    {
      ready: 'success',
      starting: 'info',
      loading_progress: 'info',
      drain_timeout: 'warn',
      stopped: 'secondary',
    }[type] || 'secondary'
  )
}
function activityText(item) {
  const bits = []
  if (item.model) bits.push(item.model)
  if (item.type === 'loading_progress' && item.loading_progress != null)
    bits.push(`${Math.round(item.loading_progress)}% loaded`)
  if (item.type === 'stopped' && item.draining) bits.push('while draining')
  if (item.pid) bits.push(`pid ${item.pid}`)
  return bits.join(' · ') || item.type
}

function modelLocation(proxyId) {
  const modelId = catalog.value.find((model) => model.id === proxyId)?.studio_model_id
  return modelId ? { name: 'model-config', params: { id: modelId } } : { name: 'models' }
}

function keyStatus(key) {
  if (key.expires_at && new Date(key.expires_at) <= new Date()) return 'Expired'
  if (key.usage?.daily_remaining === 0 || key.usage?.monthly_remaining === 0)
    return 'Budget exhausted'
  return 'Active'
}
function keySeverity(key) {
  return { Expired: 'danger', 'Budget exhausted': 'warn', Active: 'success' }[keyStatus(key)]
}

function aliasPairs() {
  const aliases = {}
  for (const row of aliasRows.value) {
    const targets = row.targets
      .split(',')
      .map((item) => item.trim())
      .filter(Boolean)
    if (row.alias.trim() && targets.length) aliases[row.alias.trim()] = targets
  }
  return aliases
}
function snapshot() {
  return JSON.stringify({ settings: { ...draft }, aliases: aliasPairs() })
}
const dirty = computed(() => loaded.value && snapshot() !== baseline.value)

async function refresh({ quiet = false, discardDraft = false } = {}) {
  if (!quiet) loading.value = true
  try {
    const [{ data }, keyResponse] = await Promise.all([
      axios.get('/api/proxy'),
      axios.get('/api/inference/keys'),
    ])
    const preserveDraft = quiet && dirty.value && !discardDraft
    if (!preserveDraft) {
      Object.assign(draft, data.settings || {})
      aliasRows.value = Object.entries(data.settings?.routing_aliases || {}).map(
        ([alias, targets]) => ({
          alias,
          targets: (Array.isArray(targets) ? targets : [targets]).join(', '),
        }),
      )
      baseline.value = snapshot()
    }
    if (
      activeTab.value === 'requests' &&
      (requestOffset.value || Object.values(requestFilters).some(Boolean))
    )
      await loadRequests()
    else {
      requests.value = data.requests || []
      requestTotal.value = data.request_total ?? requests.value.length
    }
    summary.value = data.summary || {}
    if (selectedRequest.value)
      selectedRequest.value =
        requests.value.find((request) => request.id === selectedRequest.value.id) ||
        selectedRequest.value
    running.value = data.running || []
    activity.value = data.activity || []
    catalog.value = data.catalog || []
    network.value = data.network || {}
    configuredBackend.value = data.configured_backend || data.settings?.backend || ''
    activeBackend.value = data.active_backend || ''
    if (data.restart_required !== undefined) restartRequired.value = Boolean(data.restart_required)
    keys.value = keyResponse.data?.keys || []
    error.value = ''
    loaded.value = true
    if (openLogs.value && running.value.some((row) => row.model === openLogs.value))
      await loadLogs(openLogs.value)
  } catch (err) {
    error.value = err?.response?.data?.detail || err.message || 'Could not load proxy status.'
  } finally {
    loading.value = false
  }
}

function resetDraft() {
  refresh({ quiet: true, discardDraft: true })
}

async function saveSettings() {
  const rows = aliasRows.value.filter((row) => row.alias.trim())
  if (new Set(rows.map((row) => row.alias.trim().toLowerCase())).size !== rows.length) {
    error.value = 'Alias names must be unique.'
    return
  }
  saving.value = true
  try {
    const { data } = await axios.put(
      '/api/proxy/settings',
      Object.fromEntries(
        Object.entries({ ...draft, routing_aliases: aliasPairs() }).map(([key, value]) => [
          key,
          value === '' ? null : value,
        ]),
      ),
    )
    restartRequired.value = Boolean(data.restart_required)
    toast.add({
      severity: 'success',
      summary: 'Proxy settings saved',
      detail: data.applied_live_fields?.length
        ? 'Routing and access changes applied live. Existing streams are preserved.'
        : undefined,
      life: 3500,
    })
    await refresh({ quiet: true, discardDraft: true })
  } catch (err) {
    error.value = err?.response?.data?.detail || err.message
  } finally {
    saving.value = false
  }
}

function confirmSwitch() {
  requireSingleConfirmation(confirm, {
    header: `Switch proxy to ${draft.backend}`,
    message:
      'The running proxy is stopped (engines drained and terminated), then the configured backend starts.',
    acceptClass: 'p-button-warning',
    accept: async () => {
      switching.value = true
      try {
        // Switch to the persisted configuration the banner describes, never
        // to an unsaved selection still sitting in the form.
        await axios.post('/api/proxy/switch', { backend: configuredBackend.value })
        restartRequired.value = false
        toast.add({
          severity: 'success',
          summary: 'Backend switched',
          detail: configuredBackend.value,
          life: 2500,
        })
        await refresh({ quiet: true })
      } catch (err) {
        error.value = err?.response?.data?.detail || err.message
      } finally {
        switching.value = false
      }
    },
  })
}

async function createKey() {
  if (!keyName.value.trim()) return
  creating.value = true
  try {
    const { data } = await axios.post('/api/inference/keys', {
      ...Object.fromEntries(
        policyFields.map((field) => [field.name, policyNumber(keyPolicy[field.name])]),
      ),
      token_accounting: keyAccounting.value,
      name: keyName.value.trim(),
      expires_at: keyExpires.value ? new Date(keyExpires.value).toISOString() : null,
      models: keyModels.value
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean),
      scopes: ['inference', 'models:read'],
      rate_limit_per_minute: policyNumber(keyRateLimit.value),
    })
    createdKey.value = data.key
    createdKeyLabel.value = 'Key created. Copy it now — it will not be shown again.'
    keyName.value = ''
    keyAccounting.value = 'auto'
    keyExpires.value = ''
    keyModels.value = ''
    keyRateLimit.value = null
    for (const field of policyFields) keyPolicy[field.name] = null
    await refresh({ quiet: true })
  } catch (err) {
    error.value = err?.response?.data?.detail || err.message
  } finally {
    creating.value = false
  }
}

function reportFailure(summary, err) {
  toast.add({
    severity: 'error',
    summary,
    detail: err?.response?.data?.detail || err?.message || 'Request failed',
    life: 5000,
  })
}

function confirmRevoke(key) {
  requireSingleConfirmation(confirm, {
    header: `Revoke “${key.name}”?`,
    message: 'Clients using this key stop working immediately.',
    icon: 'pi pi-exclamation-triangle',
    acceptClass: 'p-button-danger',
    accept: async () => {
      try {
        await axios.delete(`/api/inference/keys/${encodeURIComponent(key.id)}`)
        toast.add({ severity: 'success', summary: 'Key revoked', life: 2500 })
        await refresh({ quiet: true })
      } catch (err) {
        reportFailure(`Revoking “${key.name}” failed`, err)
      }
    },
  })
}

function confirmRotate(key) {
  requireSingleConfirmation(confirm, {
    header: `Rotate “${key.name}”?`,
    message:
      'A new secret is issued and the old one stops working immediately. Update clients before rotating.',
    icon: 'pi pi-refresh',
    acceptClass: 'p-button-warning',
    accept: async () => {
      try {
        const { data } = await axios.post(
          `/api/inference/keys/${encodeURIComponent(key.id)}/rotate`,
        )
        createdKey.value = data.key
        createdKeyLabel.value = 'Key rotated. Copy the new secret — the old one no longer works.'
        await refresh({ quiet: true })
      } catch (err) {
        reportFailure(`Rotating “${key.name}” failed`, err)
      }
    },
  })
}

async function copyKey() {
  try {
    await navigator.clipboard.writeText(createdKey.value)
    toast.add({ severity: 'success', summary: 'Key copied to clipboard', life: 2000 })
  } catch {
    toast.add({
      severity: 'warn',
      summary: 'Clipboard unavailable — copy the key manually',
      life: 4000,
    })
  }
}

async function loadLogs(model) {
  const { data } = await axios.get(`/api/proxy/models/${encodeURIComponent(model)}/logs`, {
    params: { search: logSearch.value || undefined, level: logLevel.value || undefined },
  })
  logs[model] = data.lines || []
}
function debouncedLoadLogs(model) {
  clearTimeout(logTimer)
  logTimer = setTimeout(() => loadLogs(model), 250)
}
async function toggleLogs(model) {
  openLogs.value = openLogs.value === model ? '' : model
  logSearch.value = ''
  logLevel.value = ''
  if (openLogs.value) await loadLogs(model)
}

function confirmClearLogs(model) {
  requireSingleConfirmation(confirm, {
    header: 'Clear captured logs?',
    message: 'The in-memory log ring is emptied. New engine output keeps streaming.',
    icon: 'pi pi-trash',
    acceptClass: 'p-button-danger',
    accept: async () => {
      try {
        await axios.delete(`/api/proxy/models/${encodeURIComponent(model)}/logs`)
        logs[model] = []
      } catch (err) {
        reportFailure('Clearing logs failed', err)
      }
    },
  })
}

function downloadLogs(model) {
  const blob = new Blob([(logs[model] || []).join('\n')], { type: 'text/plain' })
  const link = document.createElement('a')
  link.href = URL.createObjectURL(blob)
  link.download = `${model.replace(/[^a-z0-9._-]+/gi, '_')}.log`
  link.click()
  URL.revokeObjectURL(link.href)
}

async function lifecycle(action) {
  lifecycleBusy.value = true
  try {
    await axios.post(`/api/proxy/lifecycle/${action}`)
    restartRequired.value = false
    toast.add({
      severity: 'success',
      summary: `Proxy ${action === 'apply' ? 'restarted with new settings' : action}`,
      life: 2500,
    })
    await refresh({ quiet: true })
  } catch (err) {
    toast.add({
      severity: 'error',
      summary: `Proxy ${action} failed`,
      detail: err?.response?.data?.detail || err.message,
      life: 5000,
    })
  } finally {
    lifecycleBusy.value = false
  }
}
function confirmRestart() {
  requireSingleConfirmation(confirm, {
    header: 'Restart proxy and apply settings?',
    message:
      'Running engines are drained and relaunched so new limits and timeouts take effect. In-flight requests finish first.',
    icon: 'pi pi-refresh',
    acceptClass: 'p-button-warning',
    accept: () => lifecycle('apply'),
  })
}
function confirmStop() {
  requireSingleConfirmation(confirm, {
    header: 'Stop the proxy?',
    message:
      'Inference clients cannot connect until it starts again. Running engines are drained and terminated.',
    icon: 'pi pi-exclamation-triangle',
    acceptClass: 'p-button-danger',
    accept: () => lifecycle('stop'),
  })
}

function shortRevision(value) {
  return String(value || 'unknown').slice(0, 12)
}
function formatTime(value) {
  return value ? new Date(value * 1000).toLocaleString() : 'Unknown'
}
function formatDate(value) {
  return new Date(value).toLocaleString()
}
function formatAgo(iso) {
  const seconds = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000)
  if (seconds < 60) return 'just now'
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`
  return `${Math.floor(seconds / 86400)} d ago`
}
function metric(value) {
  return value == null ? '—' : `${Number(value).toFixed(1)} ms`
}
function rate(value) {
  return value == null ? '—' : Number(value).toFixed(1)
}
function formatBytes(value) {
  const bytes = Number(value || 0)
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KiB`
  return `${(bytes / 1048576).toFixed(1)} MiB`
}

function tick() {
  if (document.hidden) return
  refresh({ quiet: true })
}
function focusKeyForm() {
  keyNameInput.value?.focus()
}
onMounted(() => {
  refresh()
  timer = setInterval(tick, 5000)
})
onUnmounted(() => {
  clearInterval(timer)
  clearTimeout(logTimer)
})
</script>

<style scoped>
.key-usage {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(12rem, 1fr));
  width: 100%;
  gap: 1rem;
  margin: 0;
}
.key-usage dt,
.key-usage small {
  color: var(--text-secondary);
  font-size: 0.8rem;
}
.key-usage dd {
  margin: 0.3rem 0;
  font-variant-numeric: tabular-nums;
}
.key-usage .p-progressbar {
  height: 0.35rem;
  margin-top: 0.4rem;
}
.usage-source {
  display: block;
  color: var(--text-secondary);
}
.scope-options {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  margin: 1rem 0;
  border: 1px solid var(--border-primary);
}
.proxy-summary {
  padding: 0.75rem 0;
}
.request-filters {
  align-items: end;
}

:global(.layout-wrapper:has(.proxy-page)) {
  height: 100dvh;
  min-height: 0;
  grid-template-rows: auto minmax(0, 1fr) auto;
}
:global(.layout-main:has(.proxy-page)) {
  min-height: 0;
  overflow: hidden;
}
.proxy-page {
  min-height: 0;
  height: 100%;
  overflow: hidden;
}
.proxy-panels {
  min-height: 0;
  overflow: auto;
  display: grid;
  gap: 1rem;
  padding-bottom: 1rem;
}
.proxy-tabs {
  display: flex;
  flex-shrink: 0;
  overflow-x: auto;
  border-bottom: 1px solid var(--border-primary);
}
.proxy-tabs button {
  white-space: nowrap;
  padding: 0.8rem 1rem;
  background: transparent;
  color: var(--text-secondary);
  border: 0;
  border-bottom: 2px solid transparent;
  cursor: pointer;
}
.proxy-tabs button[aria-selected='true'] {
  color: var(--text-primary);
  border-bottom-color: var(--p-primary-color);
}
.request-table {
  overflow: auto;
  max-height: min(55vh, 34rem);
}
.request-table th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: var(--bg-card);
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}
th,
td {
  text-align: left;
  padding: 0.65rem;
  border-bottom: 1px solid var(--border-primary);
}
.request-details div {
  display: grid;
  grid-template-columns: 1fr 2fr;
  gap: 1rem;
  padding: 0.4rem;
  overflow-wrap: anywhere;
}
.request-details dd {
  margin: 0;
}

.proxy-card {
  padding: 1.25rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  background: var(--bg-card);
  display: grid;
  gap: 1rem;
}
.section-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
}
.section-head h2 {
  margin: 0;
  font-size: 1.05rem;
}
.section-head p,
.empty,
.count {
  color: var(--text-secondary);
  margin: 0.3rem 0 0;
}
.count {
  margin-top: 0;
  white-space: nowrap;
}
.backend-state {
  display: flex;
  gap: 1.4rem;
  margin: 0;
  font-size: 0.82rem;
}
.backend-state div {
  display: grid;
  gap: 0.2rem;
}
.backend-state dt {
  color: var(--text-secondary);
}
.backend-state dd {
  margin: 0;
  font-variant-numeric: tabular-nums;
}
.settings-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr));
  gap: 1rem;
}
.settings-grid label {
  display: grid;
  gap: 0.45rem;
  font-size: 0.82rem;
  color: var(--text-secondary);
}
.field {
  width: 100%;
  padding: 0.65rem 0.7rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  background: var(--bg-surface);
  color: var(--text-primary);
  box-sizing: border-box;
}
.field:focus-visible {
  outline: 2px solid var(--accent-primary, var(--accent-blue));
  outline-offset: 1px;
}
.settings-grid small,
.inline-control small {
  color: var(--text-secondary);
  font-weight: 400;
}
.inline-control {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
}
.inline-control label {
  display: grid;
  gap: 0.2rem;
  font-size: 0.85rem;
  cursor: pointer;
}
.runtime-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
}
.save-group,
.lifecycle-group,
.key-actions {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}
.alias-list {
  display: grid;
  gap: 0.6rem;
}
.alias-row {
  display: grid;
  grid-template-columns: 1fr 2fr auto;
  gap: 0.6rem;
  align-items: center;
}
.secret {
  display: grid;
  gap: 0.7rem;
  padding: 1rem;
  border: 1px solid var(--status-warning);
  border-radius: var(--radius-md);
  justify-items: start;
}
.secret code {
  overflow-wrap: anywhere;
  font-size: 0.85rem;
}
.key-list,
.runtime-list,
.key-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.7rem 0;
  border-top: 1px solid var(--border-primary);
  flex-wrap: wrap;
}
.key-summary {
  display: grid;
  gap: 0.25rem;
  min-width: 0;
}
.key-summary code {
  font-size: 0.8rem;
}
.key-summary small {
  color: var(--text-secondary);
}
.key-title {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}
.runtime {
  padding: 1rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  display: grid;
  gap: 0.9rem;
}
.runtime-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
}
.runtime-head div {
  display: grid;
  gap: 0.25rem;
  min-width: 0;
}
.runtime-head span:not(.p-tag) {
  color: var(--text-secondary);
  font-size: 0.82rem;
}
.metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(9.5rem, 1fr));
  gap: 0.8rem 2rem;
  margin: 0;
  font-size: 0.82rem;
}
.metrics div {
  display: grid;
  gap: 0.2rem;
}
.metrics dt {
  color: var(--text-secondary);
}
.metrics dd {
  margin: 0;
  color: var(--text-primary);
  font-variant-numeric: tabular-nums;
}
.state-error {
  color: var(--p-red-500, #f87171);
}
.state-alert {
  color: var(--status-warning);
}
.log-toolbar {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  flex-wrap: wrap;
}
.log-search {
  position: relative;
  flex: 1;
  min-width: 12rem;
  display: flex;
  align-items: center;
}
.log-search i {
  position: absolute;
  left: 0.65rem;
  color: var(--text-secondary);
  font-size: 0.85rem;
  pointer-events: none;
}
.log-search .field {
  padding-left: 2rem;
}
.log-meta {
  color: var(--text-secondary);
  font-size: 0.8rem;
  margin-left: auto;
  white-space: nowrap;
}
.logs {
  max-height: 22rem;
  overflow: auto;
  margin: 0;
  padding: 1rem;
  background: var(--bg-surface);
  border-radius: var(--radius-md);
  white-space: pre-wrap;
  font-size: 0.78rem;
}
.activity-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 0.35rem;
  max-height: 20rem;
  overflow: auto;
}
.activity-list li {
  display: flex;
  align-items: center;
  gap: 0.7rem;
  font-size: 0.82rem;
  padding: 0.3rem 0;
  border-bottom: 1px dashed color-mix(in srgb, var(--border-primary) 55%, transparent);
}
.activity-time {
  color: var(--text-secondary);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.activity-text {
  min-width: 0;
  overflow-wrap: anywhere;
}
.network {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr));
  gap: 1rem;
  margin: 0;
  font-size: 0.85rem;
}
.network div {
  display: grid;
  gap: 0.2rem;
}
.network dt {
  color: var(--text-secondary);
}
.network dd {
  margin: 0;
}

@media (max-width: 800px) {
  .alias-row {
    grid-template-columns: 1fr;
  }
  .runtime-actions {
    align-items: stretch;
  }
}
</style>
