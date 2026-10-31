# Frontend standards baseline

The frontend targets the applicable [WCAG 2.2 AA criteria](https://www.w3.org/TR/WCAG22/)
for semantics, names, keyboard operation, focus, contrast, reflow, input errors and
status feedback. Automated checks cover the explicit states below. Passing those
checks is not a declaration of complete WCAG conformance or a replacement for
assistive-technology testing of every engine-specific workflow.

## Clipboard behavior

The deployed LAN HTTP origin was inspected and reported `isSecureContext: false`
and no `navigator.clipboard`. The original inference-key button called that API
directly. Browser access restrictions are described in the
[MDN Clipboard API documentation](https://developer.mozilla.org/en-US/docs/Web/API/Clipboard_API).

All application copy operations now use `frontend/src/utils/clipboard.js`:

- Prefer the asynchronous Clipboard API when it is available.
- If it is absent or rejects, use a temporary, selected readonly textarea and
  check the result of the browser's legacy copy command. This is a compatibility
  path for HTTP installations, not the preferred API. Its deprecation and browser
  limitations are documented by [MDN](https://developer.mozilla.org/en-US/docs/Web/API/Document/execCommand).
- Insert the temporary field within the active dialog so its focus trap does not
  interrupt selection. Remove it afterwards and restore focus and selection.
- Never report a failed/false copy operation as successful. Never read the
  clipboard, log its contents, or store the plaintext secret in telemetry.
- Provide a selected readonly field for manual copying when automatic copying
  is blocked. Keep that feedback inside the dialog.
- Invalidate stale copy feedback when a key is replaced or dismissed. Dismissing
  the one-time key removes its value from the rendered field.

The helper is used by inference keys, Connect, task logs, and audio API examples.
The browser suite verifies actual copy followed by actual paste for both a
trustworthy localhost origin and an insecure `http://studio.test` origin. That
HTTP test bridges application assets to the local development server without
changing the browser's origin classification. It asserts that the modern API is
absent; the native legacy copy command is not mocked. A separate denial case
checks manual selection, native Ctrl+C/paste, and absence of false success.
All clipboard test credentials are synthetic fixture values.

## Shared interaction and semantic rules

- Use native buttons for actions. Model-card disclosure controls no longer wrap
  other interactive controls. Enter/Space behavior comes from the native button.
- Accessible names include visible labels and counts, supporting speech input
  as well as screen readers. Custom audio comboboxes have explicit label IDs.
- Page and section headings form an intentional hierarchy. Empty states expose a
  configurable heading level; backup/history headings fit their containing page.
- Definition lists contain terms and descriptions; tables name their action
  columns. Decorative icons do not replace text labels.
- Page changes update the document title and focus the main workspace. Initial
  loads keep the skip link reachable first. Query-only edits do not steal focus.
  Successful authentication also transfers focus into the workspace.
- Focus outlines remain visible over component-library resets. Tab-strip focus
  stays inside clipped scrolling regions. Small controls meet the tested 24px
  minimum; touch inputs have additional sizing where appropriate.
- Modal dialogs retain their keyboard focus boundary. Background Activity
  controls render beneath the modal layer. Escape/focus return and forward Tab
  wrapping are checked in real browsers.
- Both themes use contrasting primary actions, status labels, secondary actions,
  warnings, and identifier hints. Meaning is also carried in text, not color alone.
- Password visibility is a keyboard-operated native button. Errors are visible,
  linked to the password field, and exposed through `aria-invalid`.
- Responsive views retain usable navigation at 320 CSS pixels. Wide request
  tables scroll within their own region. Existing reduced-motion support remains
  active; accessibility snapshots use reduced motion to inspect settled states.

## Reproducible checks

Use the repository's Node 24 runtime:

```sh
npm ci
npm run lint:frontend
npm run style:check
npm run typecheck:api
npm run format:check
npm run test:frontend
npm run test:browser
npm run build
node frontend/e2e/cold-models-budget.mjs
```

`npm run test:a11y` runs just the standards journeys. The complete browser suite
also runs clipboard, restore, model lifecycle/configuration, keyboard, zoom,
responsive and Proxy workflows.

The accessibility matrix covers Models, Search, populated Audio, Engines, model
configuration, System diagnostics, backup, history, every Proxy tab, key
creation, the one-time secret, revocation confirmation, the workspace navigator,
and the authentication error state. It runs both themes plus selected 320px touch
states. No axe rules are disabled to produce a passing result. Reports retain
`incomplete` results for manual review instead of treating them as proven passes.

The review specifically distinguishes lazy combobox popup relationships, library
focus-trap sentinels, gradient contrast, decorative symbols and clipped inactive
tabs from automatically proven checks. Popup IDs are verified when opened;
combobox names are asserted; dialogs are tabbed repeatedly; and narrow tab strips
are traversed using the keyboard. These checks do not replace a manual screen
reader/browser matrix for all installed engines, generated media or third-party
model UIs.

Browser checks now run for pull requests as well as release tags and manual deep
runs. CI uploads the JSON accessibility reports and browser evidence. No live
keys, settings, running models, or deployments were changed for this work.

## Verified result — 30 October 2026

On Node 24.21.0, 513 frontend tests and 50 browser journeys passed. The 43
accessibility snapshots had zero detected violations for the configured rule
tags. Build, lint, style, API types, formatting and the compressed Models asset
budget passed. Raw reports are produced under `test-results/` and uploaded by CI;
the [checked-in evidence summary](frontend-standards-evidence-2026-10-30.json)
records scope and incomplete rules requiring manual review. Changes remain local
and uncommitted; the deployed HTTP instance has not been updated.

## Workspace resilience follow-up — 31 October 2026

Proxy refreshes preserve local edits. Proxy and model configuration share the same
Stay / Discard / Save dialog, and concurrent leave attempts share a confirmation.
Storage failures do not prevent editing. Proxy settings, key credentials and request
bodies are never written into view URLs. Whitelisted tabs, request metadata IDs,
filters and pagination can be restored from a URL; Models and Engines filters use
the same query-state helper. Proxy tab changes support browser Back/Forward.

Proxy live reads share in-flight work, cancel obsolete request searches, abort on
unmount, pause while hidden or offline, and use bounded exponential retry delays.
Previously received data remains visible after a failed read. Freshness is explicit;
changing check timestamps do not repeatedly announce themselves to screen readers.
System status reads also avoid overlapping refreshes and cancel on application exit.
Mutations are never automatically retried by this mechanism.

Rendering failures retain workspace navigation and provide a retry. Failed route
imports provide explicit Retry navigation and Reload Studio controls. Missing routes
have a meaningful heading and a link back to Models. Engine cards and unsaved-edit
controls now have shared components; engine lifecycle behavior remains in its owner.

The Playwright configuration now runs Chromium, Firefox and WebKit projects. Install
all matching browser binaries with `npx playwright install --with-deps` on a supported
host, or use the existing matching CI container. Local library extraction is an
execution-host workaround, not part of the application or the CI configuration.

### Manual assistive-technology acceptance matrix

These checks still require a person; automated axe and Playwright results do not
constitute a screen-reader certification. Record browser, OS, reader version,
result and issue links for each combination before marking it complete:

| Combination | Status |
| --- | --- |
| Windows, NVDA, Firefox | Not manually tested |
| Windows, NVDA, Chromium | Not manually tested |
| macOS, VoiceOver, Safari | Not manually tested |
| iOS, VoiceOver, Safari | Not manually tested |

For each combination, verify these complete journeys:

1. Unlock the password form, recover from a bad password, follow Skip to content,
   navigate every workspace, and confirm page titles and heading announcements.
2. Edit model and Proxy settings, attempt navigation, exercise Stay, Discard and
   Save, and confirm focus returns to a useful control after each dialog closes.
3. Traverse Proxy tabs and request filters with the keyboard, inspect a request,
   close its details, and restore the same view through reload and browser Back.
4. Create a fixture key, copy and paste it into a disposable local text field,
   test the manual-copy fallback, and verify the secret disappears after closing.
5. Disconnect and reconnect, verify freshness/error announcements without repeated
   timestamp chatter, and check that retry preserves settings edits and table data.
6. Read engine installation logs, validation errors, request table headers and
   empty states. Confirm log updates do not interrupt unrelated reading.
7. Repeat at 200% and 400% zoom, with keyboard-only operation and reduced motion;
   verify focus visibility, scrolling within wide tables and reachable dialog actions.

### Follow-up verification

520 frontend tests pass. The complete Chromium / Firefox / WebKit matrix ran 168
journeys: 166 passed and two failures were corrected. Firefox's three clipboard
journeys pass after fixing the secret-field focus race; WebKit's 19 model journeys
pass after correcting classification of the finite SSE fixture's native EOF
notification. There are no outstanding failed journeys after these targeted reruns.
This reports the full run and its reruns separately, rather than claiming a clean
single full run. Another 24 Proxy browser checks passed across the three engines.

All 129 accessibility snapshots (43 per browser engine) have zero detected
violations. Incomplete rule results still require manual review. The Audio audit
waits for both the collapsed ARIA state and the disappearance of the dropdown.
No accessibility rules are disabled. The exact WebKit network notification from
ending a mocked SSE response is retained as a Playwright fixture annotation;
application exceptions remain test failures.

Build, frontend lint, styles, API types, formatting, the compressed Models asset
budget and `git diff --check` pass. The compressed Models assets measured 246,157
bytes across 19 files. The
[follow-up evidence summary](frontend-resilience-evidence-2026-10-31.json) records
scope and reruns. Changes remain local and uncommitted; they have not been deployed.
