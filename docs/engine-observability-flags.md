# Engine observability flags

Studio's proxy derives request metrics from the response payload, not by
parsing free-form engine logs. The observer captures OpenAI `usage`,
llama.cpp-family `timings` and native counters, SGLang `meta_info`, and a
top-level `metrics` object. Console logging is used for the log viewer and
diagnostics; Studio does not turn console text into token or throughput rates.

This distinction decides which engine launch flags Studio owns:

- **Applied automatically** when the active build's scanned catalog proves the
  flag exists. These flags add fields the proxy actually consumes.
- **Reserved** even when a build does not expose them. A saved model setting or
  `custom_args` entry cannot redirect, truncate, or silence the log stream the
  viewer depends on.
- **Left to the user** when the flag only changes console verbosity, colors,
  timestamps, or logging of request contents. Those do not change measured
  response fields, and enabling them by default would add noise or expose
  prompts.

## Applied automatically

| Engine                                                | Flags                                     | What the proxy consumes                                                                                                                                                               |
| ----------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `llama_cpp`, `ik_llama`, `unsloth_llama`              | `--perf`                                  | `timings` (`prompt_n`, `prompt_ms`, `prompt_per_second`, `predicted_n`, `predicted_ms`, `predicted_per_second`) and native `tokens_evaluated` / `tokens_predicted` / `tokens_cached`. |
| same                                                  | `--metrics`                               | Reserved as part of the llama-family metrics contract. Prometheus values stay engine-owned and are not mapped onto proxy rates.                                                       |
| `vllm`, `1cat_vllm`                                   | `--enable-prompt-tokens-details`          | `usage.prompt_tokens_details.cached_tokens` (and cache-write fields when the engine reports them).                                                                                    |
| same                                                  | `--enable-force-include-usage`            | Final `usage` on streaming responses, so the stream observer can close its token accounting.                                                                                          |
| same                                                  | `--kv-cache-metrics`                      | Engine log statistics used by the engine's own metrics surface. Reserved because disabling stats would make the flag a no-op.                                                         |
| `sglang`, `sglang_v100`, `sglang_sxm2`, `sglang_sm70` | `--stream-response-default-include-usage` | `usage` on streaming responses.                                                                                                                                                       |
| same                                                  | `--enable-cache-report`                   | `meta_info.cached_tokens` and prompt-cache details.                                                                                                                                   |
| same                                                  | `--enable-metrics`                        | Reserved metrics surface; like llama.cpp, Studio does not scrape it to invent proxy rates.                                                                                            |
| `audio_cpp`                                           | `--log`                                   | Framework progress and timing stream shown in the Studio log viewer. Audio responses do not feed the LLM proxy rate observer.                                                         |

The catalog gate is mandatory. A pin or older build that does not list a flag
in its scanned `--help` catalog starts without it instead of failing on an
unknown argument. `--kv-cache-metrics` was already documented for vLLM; the
implementation now applies it to both vLLM ids. Before this audit it was only
in the 1Cat list.

## Reserved from user configuration

These flags are removed from the emitted structured config and rejected when
they appear in `custom_args`.

| Engine                                                | Reserved flags                                             | Reason                                                                              |
| ----------------------------------------------------- | ---------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `llama_cpp`, `ik_llama`, `unsloth_llama`              | `--log-disable`, `--log-file`                              | Would silence or redirect the process log.                                          |
| `vllm`, `1cat_vllm`                                   | `--log-config-file`, `--disable-log-stats`                 | Would replace the logging configuration or suppress the statistics stream.          |
| `sglang`, `sglang_v100`, `sglang_sxm2`, `sglang_sm70` | `--log-level`, `--log-level-http`, `--log-requests-target` | Could raise the threshold above the useful messages or send request logs to a file. |
| `lmdeploy`                                            | `--log-level`                                              | Could silence the process log.                                                      |
| `audio_cpp`                                           | `--log-file`                                               | Would redirect the process log.                                                     |

`--log-disable` and `--log-file` (llama.cpp) are the two common ways a model
configuration can make the UI look dead while the server is running.
`--log-requests-target` is the SGLang equivalent. `--disable-log-stats` is the
vLLM switch that would defeat `--kv-cache-metrics`.

## Deliberately not managed

| Engine           | Flags                                                                                                                                                                                                                                 | Why                                                                                                                                                                                                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| llama.cpp family | `--verbose`, `--log-verbosity`, `--log-colors`, `--log-prefix`, `--log-timestamps`, `--log-prompts-dir`                                                                                                                               | Presentation/verbosity only; none adds response usage or timings. `--log-prompts-dir` writes prompts to disk and is not a statistics source.                                                                                                                          |
| vLLM / 1Cat      | `--enable-log-requests`, `--enable-log-outputs`, `--enable-log-deltas`, `--max-log-len`, `--uvicorn-log-level`, `--aggregate-engine-logging`                                                                                          | Request/output logging and presentation. Auto-enabling them can expose prompts or generate very large logs. `--aggregate-engine-logging` changes the statistics format, not availability.                                                                             |
| SGLang variants  | `--log-requests`, `--log-requests-format`, `--decode-log-interval`, `--enable-request-time-stats-logging`, `--enable-forward-pass-metrics`, `--enable-mfu-metrics`, `--enable-metrics-for-all-schedulers`, `--export-metrics-to-file` | Console statistics, ZMQ/file export, or Prometheus-only metrics. The observer does not parse these channels.                                                                                                                                                          |
| LMDeploy         | `--disable-metrics`, `--max-log-len`                                                                                                                                                                                                  | `--disable-metrics` only removes the engine-owned Prometheus surface; proxy rates use response usage. `--max-log-len` trims console request text, not measured counts. LMDeploy enables metrics by default and has no request-level usage flag in the api_server CLI. |
| audio.cpp        | `--metrics`                                                                                                                                                                                                                           | Prints an offline-generation wall-time/duration/RTF summary. It is not part of the server response contract used by the LLM proxy.                                                                                                                                    |

## Verification sources

- Scanned `--help` catalogs used by the catalog gate:
  `backend/tests/fixtures/param_scans/llama_cpp.scan.txt`,
  `onecat_vllm.scan.txt`, `sglang_v100.scan.txt`, and
  `audio_cpp.scan.txt`.
- LMDeploy `serve api_server` argument source:
  <https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/cli/serve.py>.
- SGLang cache/usage flags and vLLM usage flags are exercised by the proxy
  reasoning docs and upstream server flags; the local one-cat scan is the
  authoritative catalog gate for the vLLM family in this repository.
- Proxy capture paths and timing allowlists live in
  `backend/proxy/studio/usage.py` and
  `backend/proxy/studio/observation.py`.
