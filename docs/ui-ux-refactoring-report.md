# UI and UX assessment and refactoring

Date: 2026-10-31

## Scope and evidence

Review covers the Vue application shell, model library and catalog, model configuration, audio workspace, engines, proxy, connection dialog, onboarding and activity. The initial review was static. Browser rendering is now available using locally extracted Chromium libraries; screenshots and regression checks accompany implementation. API fixtures exercise representative states, not actual model inference or real hardware capacity.

## Visual direction

Keep the restrained indigo accent and neutral light/dark themes. Make this a calm workspace with clear surfaces, readable metadata and predictable action placement. Give page titles, section headings, model names, field labels and helper text distinct levels. Use color for status and selection, with text supporting each meaning. Avoid decorative gradients, glowing status elements and competing card shadows.

## Additional visual and layout findings

| Finding | User impact | Refactoring |
| --- | --- | --- |
| Shell rules are defined as a horizontal layout, then replaced by a sidebar layout | Competing breakpoints and opaque ownership make responsive bugs easy to introduce | Consolidate shell rules in the workspace stylesheet and retain base reset/content primitives |
| Page descriptions and status badges share a wrapping title group | Titles, explanations and status can lose their reading order | Use a compact title/status toolbar and make explanatory guidance available on demand |
| Library uses multi-column masonry fallback | Visual ordering differs from row-oriented reading and moving cards become hard to track | Use a stable responsive grid, top aligned |
| Cards have several subtly different radii, padding and elevation rules | Weak grouping and visual inconsistency | Use common surface, border, spacing and heading rules; reserve elevation for overlays |
| Metadata and parameter labels shrink to around 10–11px | Important context becomes harder to read, especially on laptops | Raise small text to a readable scale and allow wrapping |
| Engine choice, metadata badges and configuration actions compete for emphasis | Users must scan too many equally prominent controls | Strong primary lifecycle/save actions, quiet secondary tools and grouped advanced actions |
| All library filters are always exposed | Toolbar consumes scarce mobile space | Search/status remain visible; secondary filters move into a disclosure with active-filter chips and a reset |
| Configuration action bar sits at bottom zero behind fixed mobile navigation | Save/apply actions can be obscured | Shared navigation offset, safe-area handling and reduced secondary action footprint |
| Activity floats above content and has inconsistent bottom spacing | It can obscure controls; closing is indirect | Move Activity into the persistent header, anchor its panel inside the viewport, and add explicit close and Escape/focus-return behavior |
| Header proxy link uses an external-link icon even for an internal page | Destination behavior is misleading | Internal router navigation for native proxy; external affordance only for legacy UI |
| Forms mix technical keys, defaults and effective values without a consistent hierarchy | Users struggle to understand what will actually be applied | Human labels first, CLI keys secondary, inherited value and per-field reset visible |
| Dense settings and cards span wide screens without a consistent rhythm | Scanning requires unnecessary horizontal travel | Responsive field/card grids with comfortable gaps and consistent content spacing |

## Combined functional findings

1. Connect must distinguish failed key loading, no configured keys and anonymous-access policy. Revocation needs the same confirmation as Proxy; one-time secrets need dismissal protection.
2. Audio must distinguish a failed model/contract/reference request from a successful empty result and preserve usable previous data.
3. Model state must use a shared presentation across library cards, rows and Audio, including stale and unknown observations.
4. Save and runtime application need consequence-oriented status copy; saving must not imply restarting.
5. Onboarding needs actionable model configuration/start/connect steps and an honest “ready to connect” completion condition, plus a way to reopen it.
6. Discovery must distinguish engine compatibility from memory fit; show unknown fit explicitly without inventing estimates.
7. Library needs result counts, human labels, clear filters and consistent lifecycle action labels across layouts.
8. Workspace navigation must say “Go to…” and support multi-term matching, rather than imply full-text workspace search.
9. Public inference URL is a shared setting and needs explicit save feedback. Request-copy options must distinguish JSON bodies from runnable requests.
10. Numeric control labels and GPU radio keyboard behavior need auditing; helper text should explain practical effects and inherited defaults.
11. Activity needs chronological ordering within lifecycle groups, accurate counts, descriptive action names and predictable dismissal.

## Implementation plan

1. **Foundations:** consolidate shell ownership; establish typography/spacing/surface tokens; improve PageHeader structure; align tabs, panels and fields; resolve responsive navigation/action offsets.
2. **Model journey:** stable library grid; responsive filters/count/chips/reset; common model-state presentation and human-readable engine/task names; consistent lifecycle actions; honest compatibility/memory guidance.
3. **Configuration:** clarify saved/applied status; reorganize action bar; expose inherited defaults and reset; correct numeric input IDs and radio keyboard navigation.
4. **Connection:** explicit loading/error/policy states, guarded key revocation and one-time-secret closure, deliberate shared URL save, full request copying.
5. **Audio and setup:** visible recoverable load failures, common runtime state, actionable setup and reopen control.
6. **Navigation and activity:** accurate navigator naming and matching, correct internal/external proxy navigation, timestamp order, lifecycle counts, task-specific labels, Escape and focus return.
7. **Verification:** focused unit regression tests, complete frontend suite, lint/build/style checks, browser accessibility and mobile reflow/occlusion checks, representative light/dark screenshots. Record limitations and results below.

## Implementation outcome

All seven phases are implemented. Changes preserve the existing Vue/PrimeVue stack and backend contracts.

| Phase | Result |
| --- | --- |
| Foundations | Shell ownership consolidated in `_workspace.css`; old horizontal shell removed from `_base.css`. Shared readable small-text scale, panel spacing, quiet surfaces, inline title/status hierarchy with contextual help, persistent compact header and consistent mobile navigation reservation. |
| Model journey | Stable row-oriented card grid; single-model actions always exposed; labeled lifecycle controls in list view; advanced warmup under More in cards; secondary filters disclosed with counts/chips/reset; human-readable labels; compatibility versus unknown memory fit explained. |
| Configuration | Consequence-oriented saved/applied labels; primary Save/Apply bar above mobile navigation; secondary tools disclosed with Escape support; inherited default/value hints and per-field reset; numeric labels attached to actual inputs; GPU modes support arrow/Home/End keys and one tab stop. Loading models retain an enabled Stop action. |
| Connection | Independent key loading/error and anonymous-policy states; guarded revocation and one-time-secret dismissal; explicit shared public-URL saving and feedback; JSON/cURL selection with correctly quoted runnable request examples; key management placed after the connection/test actions. |
| Audio and setup | Recoverable model, configuration, contract and reference-audio errors; stale state never becomes a ready claim; prior reference choices survive refresh failures; actionable configuration/start/proxy setup steps and a reopen control. Setup completion describes readiness rather than claiming a client connected. |
| Navigation and activity | “Go to…” accurately describes destination search and matches independent terms; “Find models” distinguishes the catalog; native proxy uses router navigation and legacy proxy retains its external affordance. Activity is in the persistent header, orders tasks by lifecycle then time, names task actions and supports explicit Close/Escape/focus return. |
| Verification | Regression coverage covers the revised flows, both color themes, narrow layouts, actual input labels, modal focus and action occlusion. Representative browser captures are linked below. |

## Visual evidence

Screenshots use synthetic model and key data. They illustrate layout, not hardware compatibility or a live deployment.

- [Original library](ui-review/library-before.png)
- [Updated library, dark](ui-review/library-dark.png)
- [Updated library, light](ui-review/library-light.png)
- [Connection dialog with selectable request format](ui-review/connect-dark.png)
- [Mobile configuration and reserved action/navigation space](ui-review/config-mobile.png)

## Validation

- Frontend unit/integration suite: **532 tests passed across 64 files**.
- Chromium browser regression coverage: **39 distinct checks passed across targeted runs** (19 model journeys, 9 keyboard/zoom/reflow checks, 5 standards checks and 6 refactoring checks).
- Automated accessibility audits pass in dark and light themes, including the Connect dialog's invalid-URL state and JSON/cURL controls.
- Reflow checks cover 320, 390, 768, 900 and 1440 CSS pixels across the combined suites. Mobile Save is checked against the navigation bounds and with `elementFromPoint`, so being technically visible beneath another element does not count as passing.
- `npm run lint:frontend`, `npm run style:check`, `npm run build`, and `git diff --check` pass.
- Production build still reports the existing large-chunk and mixed static/dynamic import advisories. Bundle splitting is outside this UI refactoring.

The existing browser assertions were updated for deliberate public-URL saving, corrected input IDs and the enabled Stop control during model loading. Screenshots were inspected after the layout changes. Browser dependencies were supplied outside the repository; no system packages or application dependencies were changed.

## Practical limits

Runtime memory and available installation disk space are not inferred from download size. The interface explicitly explains unknown memory fit and additional temporary disk requirements rather than displaying invented capacity estimates. Browser tests use controlled API fixtures; actual GPU inference, real downloads and physical mobile keyboards require a live deployment. Automated accessibility coverage is complemented by keyboard, zoom, reflow and element-occlusion checks; it does not substitute for assistive-technology user testing.


## Follow-up: model library table layout

The initial refactoring did not fully address the table's independently sized action columns. Different action sets shifted file-size positions between rows. The follow-up gives every desktop row identical Model / Status / File size / Actions columns, right-aligns sizes with tabular figures, labels group totals, and separates actions with a divider. Warmup is secondary in More. At tablet/mobile widths, actions occupy their own row; small screens label file size explicitly beside model identity.

A regression checks consistent size-column alignment and separation from actions at 1440, 768, 390 and 320 pixels with running models, stopped models, long names and grouped variants. The model-library unit suite (9 tests), frontend lint, stylesheet checks and production build pass.

- [Updated desktop table](ui-review/table-1440.png)
- [Updated mobile table](ui-review/table-390.png)


## Follow-up: compact page and application headers

The earlier title/description/metadata stack consumed too much height across routine pages. Shared page headers now use a toolbar: title and badges inline, actions alongside, smaller heading type, 8px bottom padding, and tighter page-top spacing. Permanent subtitles move into an accessible info disclosure that does not shift the page when opened; Escape closes it and returns focus. Model configuration uses the same inline identity treatment.

The application header is reduced from 76px to 56px in normal layouts. Small screens omit the decorative mark to preserve control space. A ResizeObserver measures the actual header height, keeping sidebar and focus-scroll offsets correct when notices or large text cause wrapping. Touch controls retain their existing minimum sizes.

Browser regression coverage checks all seven main page headers at desktop and mobile widths, desktop page-header height no greater than 64px, a normal application-header height of 56px, no horizontal overflow, and opening/closing contextual help. This follow-up also rechecks both-theme accessibility and mobile action placement. All 13 Chromium browser checks and 12 focused unit tests passed, along with frontend lint, stylesheet validation, production build and whitespace checks.

- [Compact library header, desktop](ui-review/compact-models-1440.png)
- [Compact proxy header, desktop](ui-review/compact-proxy-1440.png)
- [Compact library header, mobile](ui-review/compact-models-390.png)
