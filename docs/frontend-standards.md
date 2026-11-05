# Frontend interaction and validation reference

Studio's frontend targets accessible names, keyboard operation, focus visibility, semantic structure, input errors, status feedback, reflow, and usable controls in both themes. Automated assertions describe their tested states; manual browser/assistive-technology use supplies additional acceptance evidence.

## Shared visual structure

The application header presents the Studio orbital identity, workspace navigator, proxy state, activity access, and theme control. Page headers use a compact title/action row and an information control for guidance. Shared shell/component styles support desktop and mobile navigation.

Model tables separate identity, runtime status, file size, and actions. Wide request tables scroll inside their own region. System section links retain their width and scroll on narrow screens. Configuration defaults remain inline; longer descriptions belong to accessible information tooltips.

Both themes carry meaning through visible text and status labels as well as color. Reduced-motion preferences apply to relevant animation. Responsive controls remain reachable at narrow widths and browser zoom.

## Semantics, keyboard, and focus

- Actions use native buttons. A disclosure control and another interactive action have separate hit/focus targets.
- Accessible names contain the relevant visible label and count/identity. Icon-only buttons supply an explicit accessible name.
- Headings fit the containing page/section hierarchy. Empty-state heading levels are configurable. Tables label their action columns; definition lists pair terms with descriptions.
- Workspace navigation updates document title and main-content focus. The initial skip link remains reachable. Query-only filtering does not steal focus.
- Tab strips provide the appropriate keyboard selection and scrolling behavior. Active/inactive panels retain their declared relationships.
- Modal dialogs retain their focus boundary and provide Escape/close and useful focus return. Background Activity remains beneath the modal layer.
- Visible focus styling survives component-library resets. Tested small controls meet their declared minimum target sizes; touch layouts provide additional space where appropriate.
- Password visibility is a native keyboard-operable button. Field errors link through the relevant accessible description and invalid state.
- Page guidance and field information controls remain reachable through keyboard focus.

## Clipboard and one-time secrets

`frontend/src/utils/clipboard.js` provides shared copy behavior for keys, Connect, logs, and audio examples. It uses the asynchronous Clipboard API when available. When that API is absent or rejects, a selected readonly temporary field and the browser's copy-command interface provide an additional supported attempt.

The temporary field lives inside the active modal so dialog focus management does not break selection. Cleanup restores relevant focus/selection. A false/failed operation produces failure/manual-copy feedback, not success. Manual selection remains available when automatic copying is blocked.

Key secrets appear only in the relevant creation/rotation display. Dismissing the one-time display removes its rendered value. Copy feedback is bound to the displayed secret. Clipboard content is neither read back for telemetry nor logged as a credential.

Browser clipboard tests exercise copy/paste behavior on localhost and an HTTP fixture origin, including API absence/denial, modal focus, manual selection, and truthful feedback. Fixture keys contain synthetic test values.

## Drafts, navigation, and URLs

Proxy and model configuration use shared Stay / Discard / Save navigation prompts. Concurrent leave attempts share a confirmation. Persistence/storage errors keep the working draft available and expose the relevant refresh/retry choice.

URL state contains permitted view metadata: tabs, filters, pagination, and selected request identifiers. Proxy settings, credentials, and request bodies do not belong in view URLs. Models/Engines filters use the query-state helper, and Proxy tabs support browser Back/Forward.

Form-only edits remain distinct from saved configuration and runtime application. Invalid request-preset JSON prevents save and participates in unsaved-change handling. Successful save and the relevant runtime plan communicate the next available action.

## Live reads and resilience

Proxy live reads share in-flight work, cancel obsolete searches, abort on unmount, pause while hidden/offline, and use bounded retry delays. Read failure retains last available data with an error/freshness indicator. System status avoids overlapping requests and cancels on application exit.

Timestamp changes do not repeatedly announce themselves. Errors and material connection/state changes receive appropriate status feedback. Mutation replay is a separate explicit action rather than a consequence of read retries.

Rendering failures keep navigation available and expose retry. Route-import errors supply Retry navigation and Reload Studio controls. Unknown routes provide a meaningful heading and a path to Models.

## Reproducible frontend checks

Use Node.js 24+ with the installed browser dependencies and configured production build value:

```bash
npm ci
npm run lint:frontend
npm run style:check
npm run typecheck:api
npm run format:check
npm run test:frontend
npm run test:browser
npm run test:a11y
npm run build
node frontend/e2e/cold-models-budget.mjs
```

The Playwright configuration defines Chromium, Firefox, and WebKit projects. `npx playwright install --with-deps` installs matching browsers/dependencies on a supported host. The configured CI browser environment supplies its own matching runtime dependencies.

Browser journeys cover Models, Search, Audio, Engines, Model configuration, System sections, Proxy tabs, key creation/one-time secrets, request details, navigation, authentication errors, restore/history, clipboard, zoom, and narrow layouts. Native-routing browser coverage includes ordinary configuration, request-preset JSON/false values/removal, and multi-target alias saving.

Accessibility checks retain incomplete results for manual assessment. Open combobox relationships, names, dialog focus loops, clipped tab navigation, responsive reflow, and visible focus require their specific assertions; a scan result alone does not establish every interaction.

## Manual acceptance journeys

For a relevant browser/OS/screen-reader combination, record its versions, observed result, and issue links while checking:

1. Authenticate, recover from an input error, follow Skip to content, navigate workspaces, and inspect title/heading announcements.
2. Edit model and Proxy settings; use Stay, Discard, and Save while checking focus return and draft retention.
3. Traverse tabs, filters, tables, and request details; reload and use browser Back/Forward.
4. Create a fixture key, copy/paste to a disposable field, exercise manual selection, and dismiss the secret display.
5. Disconnect/reconnect and inspect freshness/error announcements without timestamp chatter or lost table/draft state.
6. Read install logs, field errors, table headers, empty states, and request-preset JSON validation.
7. Use keyboard-only operation, reduced motion, and 200%/400% zoom; confirm reachable actions and localized scrolling.

Automated checks and these manual journeys assess Studio's own screens. Engine-provided UIs, generated media, and engine-specific option combinations have their own applicable interaction requirements.
