# Native runtime behavior cases

The table describes current configuration, routing, and lifecycle behavior. A catalog ID identifies the managed model; the stable runtime ID identifies its process/manifests. An exposed alias selects routing/request behavior.

## Configuration and publication

| Situation                                              | Behavior                                                                                                                                      |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Edit a model form                                      | The working draft changes. Save persists it.                                                                                                  |
| Save launch settings for A while A and B run           | Desired A settings change; the existing processes keep their current launches.                                                                |
| Apply A with `restart_now` while A runs                | Validate, publish, restart A, and verify readiness. B retains its process.                                                                    |
| Apply A with `restart_now` while A is stopped          | Publish A's revision without assuming a running process needs replacement.                                                                    |
| Apply A with `next_start` while A runs                 | Publish the next revision; the current generation continues.                                                                                  |
| Apply A with `next_start` while A is stopped           | Publish the revision for its next actual process creation.                                                                                    |
| Desired and published launches match                   | No further publication is needed for that launch content. A running generation can still reflect an intentionally different next-start state. |
| Reviewed desired/published state differs at apply time | Reject the expectation conflict and expose current revision information.                                                                      |
| Same idempotency key identifies the same apply intent  | Return the recorded result.                                                                                                                   |
| Same idempotency key carries different intent content  | Reject the conflict.                                                                                                                          |
| Compilation fails                                      | Show concrete reasons and avoid publishing/starting the invalid candidate.                                                                    |
| Environment variable is set to an empty string         | Preserve an explicit empty value.                                                                                                             |
| Environment set/unset collection is empty              | Clear that collection.                                                                                                                        |
| Same variable is set and unset                         | Reject the conflicting environment specification.                                                                                             |
| GPU order/UUIDs are explicitly selected                | Preserve identity/order and validate the engine/device intent.                                                                                |
| Import `--cuda-graph-bs-decode 1 2 3`                  | Import and compile three ordered values.                                                                                                      |
| Scalar CLI option has trailing tokens                  | Consume one scalar value; classify remaining tokens separately.                                                                               |

## Aliases and routing

| Situation                                                     | Behavior                                                                              |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Request uses an ordinary alias                                | Resolve the same real model and apply its engine adapter's upstream model identity.   |
| Request uses a parameter preset                               | Apply that preset's JSON overrides and share the real process.                        |
| Preset sets `enable_thinking` to `false`                      | Preserve the Boolean value.                                                           |
| Preset uses a dotted nested path                              | Set the targeted nested field while retaining unrelated members.                      |
| Preset key ends in `?`                                        | Apply that field only when absent from the incoming request.                          |
| Request uses the base name or another preset                  | Keep that name's own defaults/overrides.                                              |
| Selected default is a request preset                          | Resolve the preset and apply its parameters to the relevant JSON request.             |
| Routing alias has multiple targets                            | Select successive targets round-robin.                                                |
| Routing alias targets request presets                         | Select a target and apply that target's preset.                                       |
| Routing/default changes while a stream is active              | Publish a snapshot for subsequent resolution; the existing stream retains its target. |
| Routing changes while serving-limit changes await restart     | Apply routing live and retain the restart requirement.                                |
| Ordinary/preset name conflicts with another model's name      | Reject conflicting model configuration.                                               |
| Preset list is explicitly empty                               | Persist removal of that model's request presets.                                      |
| Engine-specific audio passthrough receives a parameter preset | Return an explicit restriction instead of silently ignoring the preset.               |

## Supervision, failure, and recovery

| Situation                                                     | Behavior                                                                                       |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Several requests acquire the same cold model                  | Share model-owned startup work.                                                                |
| One startup waiter disconnects                                | Cancel that waiter's work while retaining startup required by other waiters.                   |
| Final unused startup waiter leaves                            | Clean up the otherwise unused cold start.                                                      |
| Acquire arrives while the selected model is stopping/draining | Coordinate with teardown before a replacement generation.                                      |
| Candidate fails readiness during apply                        | Attempt restoration of the selected model's verified revision and report the concrete outcome. |
| Another model streams during A's selective apply              | Preserve the unrelated stream/process.                                                         |
| Batch entry fails or is rejected                              | Stop the sequential batch and mark `stopped_early`; unvisited entries do not run.              |
| Total deadline expires during loading/streaming               | End supervised request work according to the deadline and record its outcome.                  |
| Client disconnects during upstream streaming                  | Cancel upstream work and retain the applicable accounting result.                              |
| Proxy restarts for serving settings                           | Drain its owned engine processes and rebuild listener/runtime settings.                        |
| Process exits unexpectedly                                    | Record failure/logs; an actual subsequent start creates a generation.                          |
| Operation/persistence response is uncertain                   | Keep uncertainty explicit and use state reconciliation/confirmation.                           |
| Portable configuration restores                               | Apply reviewed saved preferences/model settings/templates without launching processes.         |
| History restores one item                                     | Check current document revision and restore the selected saved item.                           |

Tests cover launch compilation, desired/published separation, stale/idempotent intents, environment/GPU contracts, aliases/presets, live route publication, readiness rollback, startup coordination, cancellation, persistence reconciliation, and real CPU process/socket transport. Device-specific engine readiness and inference require the actual model/hardware combination.
