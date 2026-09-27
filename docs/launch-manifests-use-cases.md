# Launch manifests: behavior and use-case specification

This is the normative behavior companion to [the implementation plan](launch-manifests-plan.md). All eight engines migrate in one release: `llama_cpp`, `ik_llama`, `lmdeploy`, `1cat_vllm`, `vllm`, `sglang`, `sglang_v100`, and `audio_cpp`. Engine capabilities differ; the state and apply semantics below do not.

**Universal rules**

Saving changes updates desired configuration only. It never modifies a running process, publishes a launch manifest, rewrites proxy configuration, or unloads another model. Applying means selecting an explicit action from the reviewed plan. A manifest controls future process creation; it cannot modify an existing process's environment or GPU allocation.

Actions used below:

- **Save only:** leave both the published and running revisions unchanged.
- **Restart A:** validate, stop only A, publish its launch revision, start it if it was running, verify readiness, and restore its last verified revision on failure when possible. Active requests to A may be interrupted.
- **Next start:** publish A's new launch revision without stopping it. The next actual launch uses it, whether caused by the user, an inference request, TTL expiry followed by a request, or crash recovery. If A is already starting, the revision its launcher has read governs that attempt; report the observed revision rather than assuming it changed.
- **Global apply:** explicitly review a proxy configuration change that can stop every loaded model. A per-model apply endpoint cannot silently escalate to this action.
- **Reject:** do not change published/running state. Report a validation error, unsupported capability, stale plan, or resource conflict with a reason.

If A is stopped, Restart A means publish only; it does not eagerly load A. Only Next start permits a running model to intentionally differ from its published revision. For every selective action, B/C remain running with their existing revisions and PIDs. New request dispatch can briefly pause during llama-swap's synchronous unload handling; this is not a zero-latency guarantee. Global actions and hardware/container failure are outside this continuity guarantee.

The authoritative classifier compares compiled launch and proxy outputs, not a hardcoded list of form labels. An unknown flag emitted into engine argv is a launch change. An unsupported field is rejected, never silently ignored. A field emitted into proxy YAML is a global change. An edit affecting both outputs requires Global apply. Document every recognized field's owner; this rule covers additional engine parameters discovered in the future.

**Environment contract**

The user-visible modes are **Set value**, **Use default/inherit**, and **Unset**. An explicitly empty string is a value and differs from Unset. Values are passed literally, preserving meaningful whitespace; `$HOME`, `${TOKEN}`, semicolons, and shell syntax do not execute or expand. Reject invalid names, NUL bytes, conflicting set/unset entries, and Studio-reserved keys. Show source/provenance for each effective value without exposing secrets.

The existing `swap_env` map can remain the set-values API for compatibility, despite no longer being serialized into swap YAML. Add an explicit `swap_env_unset` list. Supplying either collection replaces that collection, including an empty map/list; omitting it leaves it unchanged. Removing an override returns the key to its baseline/default; Unset suppresses a baseline value too. Update the current config cleanup/merge code, which drops empty values/collections, so all these operations survive save/reload. Apply the same semantics in templates, command import, and every engine adapter.

Compile a documented deployment environment baseline plus engine/toolkit defaults, then per-model overrides and unsets. Capture launch-relevant inherited values in the generation. The launcher builds its environment from that resolved snapshot, with only narrowly specified per-launch values such as assigned port; it must not silently merge old CUDA or library settings from its llama-swap parent. Baseline capture must preserve required home/temp/locale/path, cache, network, and engine variables with explicit compatibility coverage. Do not export Studio-only credentials to engines merely because they exist in the backend environment.

For ordinary keys, the model override wins. `PATH` and `LD_LIBRARY_PATH` are documented compositional exceptions: required paths for the pinned engine/toolkit first, then user additions, then approved baseline paths, deduplicated without reordering. Show the resulting value. Empty user additions remove those additions; explicit Unset is rejected if required engine paths make it impossible to honor. `CUDA_HOME`/`CUDA_PATH` overrides must describe a compatible toolkit, otherwise validation fails. Adapters must reject conflicting required runtime invariants instead of silently overriding the user's intent. Per-model GPU controls and raw CUDA visibility edits resolve through one canonical representation as defined below.

| ID | User action | Required behavior |
| --- | --- | --- |
| E1 | Add/change an environment variable for A | Save only initially. Restart A or Next start; no proxy YAML change. Child worker processes inherit the new effective environment after launch. |
| E2 | Remove A's override | Restore the current compiled baseline/default for that key; if absent there, explicitly exclude it. Restart A or Next start only if the effective launch spec changes. |
| E3 | Explicitly Unset an inherited variable | Ensure it is absent from A's new process; reject if a declared engine invariant requires it. No effect on B/C or the backend. |
| E4 | Set an empty value, or a value containing spaces, `$`, or quotes | Preserve the exact supported string; no shell expansion. Restart A/Next start if effective output changes. |
| E5 | Edit only map ordering or replace a value with its effective default | If effective spec and proxy output are unchanged, no publication or restart. A future baseline change may make an explicit override and inheritance differ; preserve that intent in saved config. |
| E6 | Edit `LD_LIBRARY_PATH`, `PATH`, toolkit or engine-specific environment | Show the fully resolved environment and validate compatibility. Restart only affected model(s); pinned engine paths must not follow a mutable `current` symlink. |
| E7 | Change a secret used by A at engine startup | Same selective lifecycle as other environment changes; redact values in previews, diffs, receipts, logs, and events. Never mutate a referenced external secret file invisibly behind a supposedly immutable generation. |
| E8 | Change Studio's Hugging Face search/download token | Update that backend credential through its existing workflow. Do not restart inference models unless their explicitly compiled runtime credential changes too. |
| E9 | Edit `.env`, Compose environment, or the parent shell after processes started | Existing OS processes do not acquire those edits automatically. Apply them through the relevant backend/container restart workflow; do not promise selective continuity for container recreation. Recompile desired model baselines afterward; already-published manifests retain their recorded environment until applied. |
| E10 | Change a supported Studio-managed runtime default | Mark exactly the models whose effective launch specs change as pending. No eager process changes or proxy restart; select the affected set for Restart/Next start. |

**GPU assignment contract**

Present **Inherit deployment selection**, **Selected GPUs (ordered)**, and **CPU/no CUDA devices** as distinct choices. An empty selection must not ambiguously mean both inheritance and CPU mode. Selecting every GPU explicitly must preserve that ordered selection rather than silently becoming inheritance.

Persist stable GPU identities where supported, preferably full UUIDs, and display the resolved logical-device mapping. Existing numeric selections must be resolved against the deployment's device enumeration during migration; ambiguous mappings block affected-model validation. Validate device availability inside the actual container/deployment and any configured visibility policy, not just host inventory. Device visibility is not a reservation of memory or exclusive ownership.

CUDA visibility determines enumeration order; selecting physical devices in order `[GPU-2, GPU-0]` makes them logical devices `[0, 1]` in that process. Validate dependent main-device, tensor-split, tensor/pipeline parallelism, and audio-device settings against this effective order. GPU list order is part of the launch spec and must not be sorted away. [NVIDIA CUDA environment-variable reference](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)

Treat the GPU picker and raw `CUDA_VISIBLE_DEVICES` input as two editors of one setting. A raw value must round-trip through the UI, including UUIDs and order; a conflicting request is rejected. CPU mode compiles no CUDA visibility plus engine-specific CPU flags/backend selection. Hiding GPUs alone does not establish that a GPU-only engine can run on CPU. Vulkan or other backend device selectors use their own adapter-specific namespace and validation, not CUDA ordinals.

| ID | User action | Required behavior |
| --- | --- | --- |
| G1 | Move A from GPU 0 to GPU 1 | Save only; Restart A releases its old process before launching on the new GPU. Next start leaves A on its original GPU until its next launch. B/C are not moved or unloaded. |
| G2 | Add/remove GPUs assigned to A | Restart A/Next start after checking visible devices, backend support, parallelism, and split settings. Reject inconsistent combinations before stopping A. |
| G3 | Reorder selected GPUs | Treat as a real launch change. Revalidate logical main-device and split mappings and show the before/after mapping. |
| G4 | Change tensor split, main GPU, GPU layers, tensor/pipeline parallel size, batch size, or GPU memory utilization | Restart A/Next start if emitted into its launch spec. Validate supported combinations; a memory estimate is advisory, not a guarantee. |
| G5 | Remove explicit selection and inherit | Resolve the deployment baseline, not an unrestricted host list or a stale parent-process override. Restart/Next start only when the effective launch spec changes. |
| G6 | Switch A between CPU, CUDA, Vulkan, or another supported backend | Validate the engine build and its device namespace. Restart A/Next start if proxy contract is unchanged. Reject unsupported CPU/backend combinations. |
| G7 | Select a missing, inaccessible, ambiguous, or unsupported GPU/MIG identity | Reject before stop when detectable. Never silently substitute GPU 0 or a different device after enumeration changes. Validate supported MIG forms only where the adapter/runtime supports them. |
| G8 | Assign A to a GPU already used by B | Sharing is permitted if the runtime can support it. Show shared occupancy; do not automatically evict B. Reject definite incompatibilities, otherwise attempt A under the normal failure/rollback policy. |
| G9 | A fails with insufficient GPU memory | Fail A's apply and restore its last verified revision if possible. Leave B/C untouched. Report if the restoration also lacks resources; do not call unload-all to recover. |
| G10 | Change container `--gpus`, device mounts, or `NVIDIA_VISIBLE_DEVICES` | Deployment operation requiring appropriate container recreation. A model environment cannot grant hardware not exposed to the container. Reject it as a model-level GPU-access change. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html) |
| G11 | Activate a different Studio CUDA toolkit | Resolve immutable toolkit paths and mark all models whose launch specs change. Apply only that set; preserve other models' paths and running processes. Revalidate SGLang V100's required toolkit/build compatibility. |
| G12 | GPU removed/reset, driver changed, or host rebooted | External failure, not selective configuration application. Mark affected models unavailable and require revalidation; do not remap identities or claim uninterrupted B/C service on failed shared hardware. |

Example: A runs on GPU 0 with `CUDA_VISIBLE_DEVICES` selecting its UUID; B runs on GPU 1. Saving A's assignment to GPU 1 changes nothing immediately. Restart A stops A, releases its GPU 0 allocations, and tries the new generation on GPU 1. B remains running. If A cannot coexist with B, A's apply fails and Studio attempts to restore A on GPU 0. Next start instead leaves A on GPU 0 now and publishes the GPU 1 choice for its next launch. The UI must show both current and next assignments.

**Other model, engine, artifact, and routing changes**

| ID | User action | Required behavior |
| --- | --- | --- |
| C1 | Change context size, threads, cache type, batching, or an engine CLI flag | Launch-only when compiled into argv/env/sidecars: Restart A/Next start. No reload merely because the flag is newly discovered. |
| C2 | Change an engine startup sampling default | Restart A/Next start if it is a launch argument. Classify by implementation, not by the word “temperature.” |
| C3 | Change sampling/task parameters in one inference request | Send with that request, subject to runtime validation; no manifest publication or restart. |
| C4 | Change saved llama-swap `setParams`, `setParamsByID`, stripping rules, or audio request defaults emitted as filters | Global apply. A launch manifest does not make those proxy settings dynamic. |
| C5 | Edit the configuration of an engine not currently selected for A | Save only. No effective launch/proxy change until that engine is selected. |
| C6 | Switch A's selected engine | Validate artifact/task compatibility and installed version. Restart A/Next start only if the proxy contract remains identical; otherwise Global apply. Never restart other models just because they use the destination engine. |
| C7 | Activate a new version of one engine | Compile the exact affected model set using that engine's active-version selection. Existing generations remain pinned. Runtime-only changes can apply to that set; changed proxy contracts require Global apply. Merely installing a version without activation does nothing to serving. |
| C8 | Rebuild or update engine binaries in place | Stage a new immutable installation identity; existing generations continue using old files. Do not mutate a running/referenced installation and call it a selective restart. |
| C9 | Change A's weights, quantization, model bundle, tokenizer/config files, mmproj, MTP, DFlash, or other startup companion | Stage/version artifacts and validate compatibility. Restart A/Next start if stable model identity and proxy contract remain the same. A new catalog model/route is C14. Preserve rollback dependencies. |
| C10 | Change audio.cpp family/task/backend/device/load/session settings or startup voice presets | Changes compiled into its server sidecar/argv require Restart A/Next start. Changes also emitted into filters or upstream mapping require Global apply. Request-only workspace inputs are C3. |
| C11 | Replace reference audio or other external files read at runtime | Pin generation-owned copies/content identities for persistent configuration, or explicitly classify them as per-request input. Never silently overwrite a referenced generation dependency. A pinned startup dependency follows C9/C10. |
| C12 | Delete engine/model files used by current, published, or rollback generations | Reject until references are safely retired through the relevant model lifecycle. Never invalidate rollback dependencies silently. |
| C13 | Rename a Studio-only display label or edit notes | No runtime action. If a field is deliberately exposed in llama-swap's `/v1/models` metadata, its proxy projection changes and Global apply is required to publish that field. |
| C14 | Add/remove a model route, change its stable proxy ID, alias, or upstream model-name mapping | Global apply. Saving/catalog download may precede this; a new model cannot serve until its route is registered. |
| C15 | Change TTL, concurrency limit, health endpoint, proxy timeout, groups, selectors, or profile definitions in swap YAML | Global apply. Do not move these into a launch manifest and imply stock swap reads them there. |
| C16 | Activate an already-defined routing profile using its existing runtime API | Use that API; do not rewrite YAML solely for activation. Subsequent request-driven model loading/eviction follows the configured routing policy. Editing the profile definition is C15. |
| C17 | Edit runtime settings and an alias/filter in the same save | One mixed plan requiring Global apply. Per-model apply rejects it; the UI must not promise a selective restart for a partial configuration. |
| C18 | Import CLI configuration or apply a model template | Import/save first, then use the same classifier and environment/GPU validation. No special path that auto-publishes or bypasses global-impact detection. |
| C19 | Edit an unrecognized/unsupported option, duplicate Studio-owned port flags, or incompatible capabilities | Reject with the field and reason. Unknown valid engine flags accepted through supported custom-argv handling are launch-only; no silent dropping or arbitrary shell execution. |

**Model state, operation, and recovery cases**

| ID | State/action | Required behavior |
| --- | --- | --- |
| S1 | Save A while it is running, stopped, or loading | Desired only. Current attempt and future starts keep the published revision until an apply action publishes another. |
| S2 | Apply launch-only change to stopped A | Publish, leave stopped, report “Ready for next start”; do not claim runtime readiness was tested. |
| S3 | Restart running A with active requests | Warn that A's requests can be interrupted; stop only A and verify its replacement. Do not automatically retry partially streamed inference requests. |
| S4 | Apply while A is loading | Restart-now cancels/unloads that attempt and selects the new revision. Next-start does not cancel the attempt; report the receipt's revision and any pending restart. |
| S5 | Save a newer edit while an apply is running | Finish the accepted immutable revision; leave the newer desired revision pending. Never clear it with a global stale flag. |
| S6 | Retry the same apply or submit a stale plan | Idempotency returns the existing operation/result. Revision mismatch returns conflict without a second restart. |
| S7 | Request A while it is being stopped/published | Coordinate launch gate and swap lifecycle so a new process selects one complete published generation. Requests may wait or receive the documented temporary error; never start duplicate instances or mixed sidecars. |
| S8 | Stop/delete A or activate/delete its engine during apply | Serialize through shared resource ownership or return busy/conflict. No stale deletion or competing stop/start after resource release. |
| S9 | Invalid manifest, missing file, incompatible environment/GPU detected before stop | Reject and preserve A's current service; no effects on B/C. |
| S10 | Exec or readiness fails during restart-now | Restore the last verified running generation when available, rather than an untested prior next-start pointer. Report apply failure plus rollback outcome; desired settings remain visible for correction. |
| S11 | A later starts a next-start revision and fails | Record that revision's launch failure; do not claim the earlier publication verified readiness. Do not auto-loop between revisions. Offer explicit rollback/retry using retained dependencies. |
| S12 | Rollback also fails | A remains failed/stopped with both errors and retained manifests; no unrelated unload. |
| S13 | Backend restarts during apply | Reconcile journal, pointer, receipts, and process identity before allowing conflicting work. Do not replay a completed restart or mistake PID reuse for a live revision. |
| S14 | Proxy is unreachable or runtime state unknown | Fail/retry restart-now without guessing A is stopped. Next-start may publish after confirming no conflicting operation and valid proxy projection, with running state explicitly unknown. If Studio itself must restart the proxy, show that as a separate global recovery action. |
| S15 | Bulk apply multiple launch-only models | Prevalidate the selected set, execute sequentially, record individual outcomes, and stop further automatic restarts on the first failure. Keep completed models on their verified revisions and remaining models pending; do not invent transactional rollback of the batch. |
| S16 | Cancel apply | Before stopping: leave published/running state unchanged. After stopping/publication: recover to a consistent state, then report cancelled/recovered or recovery failure. Cancellation must not release ownership while recovery is still running. |
| S17 | Discard unapplied saved edits | Restore desired state to the explicitly selected published or running baseline without restarting. If Next start already published the edit, undoing that publication is a new pointer action, not just closing the form. |
| S18 | Revert a next-start choice while old A still runs | Publish the selected retained running revision under the same validation/gate protocol. If it matches A's live revision, no restart is needed; clear only the resolved pending difference. |
| S19 | Manual external manifest/YAML edits | Treat generated files as Studio-owned. Detect drift and block apply pending reconciliation; do not infer that external bytes are trusted revisions or overwrite them without an explicit repair action. |
| S20 | One-time migration or full legacy rollback | One deployment-wide transition covers every deployed model and every supported engine. Preflight failure for any deployed entry aborts before service disruption. No partial engine fallback or hybrid config is published. |

**All-engine implementation and release matrix**

| Engine | Required adapter-specific coverage |
| --- | --- |
| `llama_cpp` | Native executable, cwd/library paths, GGUF/companions, CUDA visibility/order, CPU/backend flags |
| `ik_llama` | Its own scanned flag contract and split/offload controls; no assumption that upstream llama.cpp flags are interchangeable |
| `lmdeploy` | Venv executable, `serve api_server`, server-port substitution, model source, parallelism and environment |
| `1cat_vllm` | Pinned venv/module, venv cwd, worker-process environment, GPU parallelism |
| `vllm` | Upstream venv/module and discovered options; own install identity despite shared implementation helpers |
| `sglang` | Venv/module, GPU/tensor-parallel settings, compatible toolkit environment |
| `sglang_v100` | V100-capable deployment, supported pinned toolkit/build, required architecture/runtime environment |
| `audio_cpp` | Executable/cwd, immutable server sidecars, backend/device mapping, voices and startup assets, filter-vs-sidecar classification |

Every applicable E/G/C/S case needs an automated test mapped to its ID. Tests may share adapters/fixtures but must enumerate all eight registry IDs. Unsupported capabilities require explicit rejection tests. Cross-engine continuity tests must include text/text and text/audio pairs. GPU order, unavailable-device, memory-failure/rollback, empty/unset environment, and child-worker inheritance tests are release gates, not optional follow-up work.

The release switches all supported engines together only after compiler, fake-process, real-runtime smoke, migration, rollback, and UI behavior gates pass. Hardware gaps remain explicit incomplete release evidence; they do not authorize shipping only a subset of engines.
