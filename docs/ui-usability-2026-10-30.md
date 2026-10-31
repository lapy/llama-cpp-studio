# Studio usability review — 30 October 2026

## Method and limits

Expert walkthrough of the deployed application at `http://10.0.1.9:8081`, followed
by local implementation and browser verification. This is a heuristic usability
review, not a participant study: no users were recruited and no task completion
rates are claimed. Live inspection was read-only apart from creating a management
login session. No settings, budgets, statistics, models, or running processes on
the deployed instance were changed.

Reviewed the live Proxy overview and request records, expanded footer tools,
Models, Engines, and Search. Checked the revised shell, Proxy, diagnostics,
backup/restore, and change history locally. Local preview screenshots replayed
captured metadata, with the new HTTP outcome counters supplied from that
metadata; they are not screenshots of a deployed update.

Tasks: identify a running engine; explain a high error count; open its requests
and logs; locate backup/restore and diagnostic download; return to the workspace;
navigate with a keyboard; keep controls reachable on narrow screens and at zoom.

## Findings and implemented corrections

| Priority | Evidence / problem | Correction |
| --- | --- | --- |
| High | Ready model displayed 271 “Errors” at first inspection, indistinguishable from engine failures. | Preserve the total counter for compatibility; add independent HTTP 4xx / 5xx counters in gateway and generation statistics. Link model outcomes to a model-filtered request list. |
| High | No direct route from an alarming model counter to the requests that explain it. | Add request outcome filters (4xx, 5xx, both), include rejected/cancelled records, and provide model-specific request and failure actions. |
| High | Expanded footer displaced most of the working viewport with diagnostics, backup, restore and history. | Move tools to `/system` with separate Diagnostics, Backup & restore, and Change history destinations. Keep the footer as status and a System link. |
| High | Proxy model card used approximately 413px of a roughly 1,150px content area. Metrics stacked vertically and actions disappeared below the panel edge. | Replace the shared flex list rule with a full-width grid. Keep core model metrics visible; disclose revision/start information under Process details. Local desktop card measured 1,163px wide and 213px tall at 1440×1000. |
| Medium | Large cumulative shell padding, header padding, tab gaps and nested card gaps delayed useful content. | Reduce shared page/header spacing, compact Proxy tabs and controls, and prevent the panel grid from stretching sparse contents. |
| Medium | Backup tools used a small uppercase footer heading, long unstructured copy and a native file picker beside unrelated diagnostics. | Use distinct Export settings / Restore settings cards, consistent heading styling, shorter explanatory copy and styled file controls. Keep preview, confirmation and recovery behavior. |
| Medium | Mobile footer inherited a column layout and occupied three rows. | Override the inherited layout with a compact status row, ellipsize the version when needed, and keep System reachable. Local mobile footer measured 32px high before increasing the System touch target; browser checks require it to remain under 65px. |
| Medium | Mobile navigation had five destinations assigned to four columns. | Add the System destination and explicitly size six columns. Test narrow screens and both themes. |
| Medium | Model-card names/repositories competed for one line; running actions collided. | Wrap names/repositories and action groups inside narrow cards while preserving full control labels. |
| Medium | A reconnect could leave an empty, automatically opened Activity panel covering retry controls. | Close the panel when its task snapshot becomes empty; keep the mobile dock above footer/navigation controls. |
| Medium | Old restore bookmarks depended on opening tools in the footer. | Redirect `/restore` and existing `#config-backup` links to `/system#config-backup`. |

The distinction between intentional empty workspace and wasted operational space
matters: an empty Search page needs room for results. The strongest density
problems were the narrow runtime card, expanding footer, stacked mobile footer,
and repeated shell/card spacing, rather than every unoccupied pixel.

## Deployed request findings

The first Proxy snapshot showed one ready process, 277 model requests and 271
model HTTP errors; the gateway showed 285 requests and 279 HTTP errors. Later
pagination collected 291 retained records while the deployment remained active:

| HTTP status | Records |
| --- | ---: |
| 200 | 11 |
| 401 | 8 |
| 404 | 272 |
| 5xx | 0 |

The 404 paths included environment/credential filename probes such as `/.env`
and `/firebase-key.json`. These are consistent with automated scanning, not
proof of a model crash. The 401 records were rejected before dispatch. The
model was ready when inspected. Requests can arrive between snapshots, so the
first card counts and the later paginated totals are deliberately distinguished.

Anonymous inference was enabled in the inspected settings. Anonymous requests to
unknown paths were being forwarded to the selected engine and their 404 responses
were counted against that model. This explains the alarming count and why its
short-request latency percentiles were dominated by non-inference traffic. No
access policy was changed on the live instance. Requiring inference keys is an
existing Settings & routing control if anonymous clients are not intended.

The revised UI labels the counters as HTTP outcomes and explicitly says they
cover all traffic. It does not hide probes or reset evidence. General upstream
passthrough remains compatible; introducing an endpoint allowlist requires a
separate compatibility decision for engine-specific APIs. Generation counters
reset independently alongside the existing observation-window reset.

## Verification

- Frontend unit/component suite: 500 tests passed.
- Affected backend accounting, proxy controls and dataplane suite: 119 tests passed.
- Full browser suite: 40 tests passed, including desktop/mobile Proxy checks,
  configuration restore, model journeys, both themes and accessibility flows.
- Added browser checks for full-width model diagnostics, model/outcome filter
  navigation, System destinations, compact footer and horizontal overflow at
  desktop and mobile widths.
- Browser restore and accessibility checks cover previews, confirmation,
  cancellation, focus return, legacy bookmarks and 200% zoom. During verification,
  a seven-pixel file-picker overflow was found and fixed by bounding the field
  to its restore card. Download retry checks also exposed the empty Activity-panel
  obstruction; closing that stale panel restored normal button access.
- Build, frontend lint, stylesheet lint, backend lint, API type check and OpenAPI
  consistency checks pass. Existing mixed static/dynamic import build warnings
  remain.

Changes are local; the deployed instance has not been updated. Hardware model
performance and inference correctness were outside this usability review.

## Control room iteration

The follow-up implementation establishes a stronger operational hierarchy:

- Overview: four headline metrics, a bounded traffic visualization, gateway
  health, then the engine fleet. HTTP outcomes remain separate from readiness.
- Traffic: real retained request timestamps and HTTP statuses, binned across a
  30-minute window. The newest-100 sample limit is explicit. Empty windows stay
  empty; incomplete requests are not counted as successful responses. This is a
  sample visualization, not an exhaustive time-series analytics store.
- Mobile: traffic/health analytics collapse behind a native disclosure so fleet
  information appears earlier. Opening it is explicit and preserves normal
  keyboard navigation order.
- Workspace navigation: Ctrl/Command K opens a searchable destination picker.
  Arrow keys and Enter navigate; Escape returns focus to the launcher. A
  successful destination change focuses the main workspace.
- Access keys: existing clients and their usage come first. Creation happens in
  a focused dialog with visible validation and duplicate-submit protection.
  The one-time credential is presented separately and removed from rendered
  state on dismissal. Rotation uses the same credential presentation.
- Requests: compact method/path cells and grouped outcome, routing, timing and
  accounting inspection. Unavailable optional fields are hidden initially and
  can be disclosed; essential unknown token counts still display as unknown.
- Shell: navigation groups distinguish workspace tools from operations, dialog
  titles use consistent neutral typography, and the proxy fills available width.

Verification for this iteration: 506 frontend tests passed; the full browser
suite passed 42 tests, followed by focused checks for responsive disclosures,
keyboard focus, credential creation and optional diagnostic fields. Build,
frontend lint, stylesheet lint and API type checks pass. Preview screenshots use
controlled fixture data and do not represent a deployed update. There is still
no participant-study evidence or basis for claiming a comparative industry rank.
