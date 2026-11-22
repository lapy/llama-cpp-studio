# UI alignment foundation

Status: **complete** — foundation and all routed-page migrations are integrated
on `feat/single-operator-hardening`. Branch history: `ws/ui-foundation`, then
`ws/ui-pages-models`, `ws/ui-pages-config`, `ws/ui-pages-engines`,
`ws/ui-pages-system` and `ws/ui-pages-proxy-audio`.

This document is the visual and structural consistency contract for Studio's
routed pages. It complements [`frontend-standards.md`](frontend-standards.md),
which stays the accessibility and interaction contract. The release-wide
workstream plan is in
[`single-operator-hardening-plan.md`](single-operator-hardening-plan.md) (WS5).

Scope of this workstream:

- Add shared primitives and tokens so pages can be migrated without copying
  card/section/toolbar CSS.
- Migrate `NotFoundView` and the application shell as the reference
  implementation.
- Audit every page and record its concrete inconsistencies and migration map.
- Do **not** migrate Models, Search, ModelConfig, Engines, Proxy, System or
  Audio in this branch: WS1/WS2 and the page workstreams own those files.

Line references below are from the foundation branch base and will drift as
page migrations land. Treat them as evidence for the audit, not stable anchors.

## Shared rules

Every routed page must satisfy these ten rules. "Primitive" names the shared
implementation a page uses.

| #   | Rule                                                                                                                                                                                                                                  | Primitive                                            |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| 1   | **Page shell.** Every routed page uses `page-shell` (or `page-shell--wide` for table-heavy workspaces) and exactly one `PageHeader` with title, description, optional meta and actions. One primary action per page.                  | `page-shell`, `PageHeader`                           |
| 2   | **Sections.** All page sections use one `PageSection` pattern: heading, optional description, optional meta/actions, consistent padding/gap/border. No page-local `system-card`/`proxy-card`/`ev-section` lookalikes after migration. | `PageSection`, `.page-section*`                      |
| 3   | **State.** Loading uses `LoadingState`; empty uses `EmptyState` with a next action; stale/error banners use the shared `state-banner` (or `StateBanner`) with retry. No silent blank panels.                                          | `LoadingState`, `EmptyState`, `StateBanner`          |
| 4   | **Controls.** PrimeVue components with shared sizes; one primary button per section; icon-only buttons have `aria-label`s; labels are sentence case and consistent (`Save`, `Save settings`, `Retry`, `Refresh`).                     | PrimeVue + shared sizes                              |
| 5   | **Tables and toolbars.** One toolbar pattern (search, filters, summary, actions), one table density, labelled action columns, empty/loading inside the table shell, horizontal scroll contained.                                      | `PageToolbar`, `.table-shell*`                       |
| 6   | **Forms.** Shared field spacing and label/hint/error placement, two columns at desktop and one at mobile, consistent save bar and unsaved-change prompt.                                                                              | `.page-form*`                                        |
| 7   | **Dialogs.** Shared header/body/footer structure, width scale, focus return, destructive confirmation, one-time secret treatment.                                                                                                     | `dialog-width-*`, `.dialog-stack`, `.dialog-actions` |
| 8   | **Type and color.** Page `h1`, section `h2`, sub-section `h3`; tokens only; both themes; no hardcoded colors; consistent icon sizes and empty-state illustrations.                                                                    | tokens, `PageSection` heading levels                 |
| 9   | **Responsive.** No horizontal overflow at 320/390 px; actions remain reachable; tables scroll locally; touch targets meet the standards.                                                                                              | shared media rules                                   |
| 10  | **Usefulness.** Every page answers: what is this, what is the primary action, what state am I in, what do I do next. Empty and failed states always offer the next step.                                                              | `PageHeader`, `EmptyState`, `StateBanner`            |

## Page migration record

| Page             | Integrated changes                                                                                                                                                                                                                                                                                                                               | Evidence                                                                                                                                                  |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Models           | Shared `StateBanner`/`PageToolbar`/`table-shell`; extracted `LibraryToolbar` and `LibraryListView`; shared Hugging Face token notice/dialog; first-model section; lazy cold-path chunks.                                                                                                                                                         | 8 targeted files / 85 tests; full frontend suite; `/models` cold budget 247,036 B ≤ 280,000; `ui-refactor` Models journeys.                               |
| Search           | `PageHeader` result/import actions; `PageToolbar` filters; `PageSection` results; shared empty/loading/error states; extracted variant/install/audio/token dialogs.                                                                                                                                                                              | Targeted Search suites; `ui-refactor` Search journey; 320/390 checks in standards.                                                                        |
| ModelConfig      | Descriptor-driven `SectionTabs` with Overview / Runtime / engine tabs / Names / Launch; new read-only `ModelConfigNamesSummary`; shared `PageToolbar` save bar, `StateBanner` states, dialog/form patterns; 3,740 lines (was 3,820).                                                                                                             | 11 targeted files / 94 tests; Chromium model journey 19/19; targeted axe 8 scans / 0 violations; 18 screenshots at 320/390/1440 both themes, no overflow. |
| Engines          | `PageSection`/`PageToolbar`/`SectionTabs`/state/dialog patterns across the card grid, filters, version table, install/build/update dialogs and notes; extracted structure tests.                                                                                                                                                                 | 10 engine component/view files / 92 tests; full frontend suite; `EnginesView.structure.test.js`.                                                          |
| Proxy            | Tabs on `SectionTabs`; sections on `PageSection`; lifecycle/filter rows on `PageToolbar`; banners on `StateBanner`; request table in `table-shell`; extracted overview, keys, requests and activity panels; ProxyView 3,039 → 1,212 lines.                                                                                                       | 10 files / 126 tests; full frontend suite; `proxy.spec` 12/12 Chromium; standards axe 5/5; 320/390/1440 screenshots.                                      |
| System           | One `PageHeader`, five `SectionTabs` links (Hardware / Storage / Diagnostics / Backup & restore / Change history), each a `PageSection`; StoragePanel normalized; HardwarePanel split into metrics + CUDA panels; Storage validates the inventory payload and shows a retryable error instead of crashing the route when the API is unavailable. | 6 targeted files / 46 tests; full frontend suite; dev-box targeted run; module budget reduced.                                                            |
| Audio            | `PageHeader` description; compact model/context section; pill `SectionTabs`; task panels and results as `PageSection`; `StateBanner` replaces ad-hoc notices; new transcript panel.                                                                                                                                                              | Audio/proxy files / 126 tests; full frontend suite; `standards` and `ui-refactor` Audio journeys.                                                         |
| NotFound / shell | Foundation reference migration: shared `PageHeader`/`EmptyState`/`StateBanner` and route-failure handling.                                                                                                                                                                                                                                       | Foundation unit tests; `recovery.spec`.                                                                                                                   |

Shared-contract repairs found by the page wave:

- `_patterns.css`: `.page-section__heading-group` now resets to `flex: 0 0 auto`
  when the header stacks at ≤640 px. Without it, every migrated section
  reserved ~200 px of empty header height on mobile.
- `useRovingTabs`: scrolls the newly focused tab into the scrollport, restoring
  the keyboard contract for clipped tab strips at narrow widths.
- `GpuAdmissionPanel`: the `role="status"` live region wraps the `<dl>` instead
  of being placed on it, fixing `aria-allowed-role`/`dlitem` axe failures.

## Screenshot review

Integrated Chromium review at 320/390/1440 px in both themes (headless; the
PrimeUI development-license badge is a local build artifact, not a page
element):

| Page        | Reviewed states                                                  | Result                                                                                                              |
| ----------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Models      | library table/cards, filters, empty and stale banners            | One header, one primary action, contained table scroll, consistent cards                                            |
| Search      | initial, results, empty, error, dialogs                          | Toolbar and result sections match Models; dialogs share header/footer                                               |
| ModelConfig | Overview, Names, Runtime filters, Launch/save bar                | One tab strip, one save bar, no overflow; Names panel is the single identity explanation                            |
| Engines     | cards, filters, versions, install/build dialogs                  | One header, sectioned cards/table, consistent dialog widths and actions                                             |
| Proxy       | Overview, Requests, Events, Access keys, Settings & routing      | Consistent tab strip, metric cards, table shell and dialog stack; GPU admission card uses the same section language |
| System      | Hardware, Storage, Diagnostics, Backup & restore, Change history | Five sections share tab, heading, action and card geometry                                                          |
| Audio       | model/context bar, task tabs, speech and transcript panels       | Task tabs and sections match the rest of the workspace; stacked fields at 390 px                                    |
| NotFound    | route failure                                                    | Shared shell, meaningful heading, return action                                                                     |

No page-local card/section lookalike remains for a migrated page. The review did
not treat screenshots produced by unrelated test runs as intentional changes.

## Page outcomes required by the plan

| Page        | Required outcome                                                                                       |
| ----------- | ------------------------------------------------------------------------------------------------------ |
| Models      | Clear library/table switch, toolbar, first-model CTA, running/attention states, consistent row actions |
| Search      | Filter/toolbar consistency, result cards/table, empty/loading/error, import action                     |
| ModelConfig | Section nav (Engine / Runtime / Names / Launch), Names & routing summary, consistent save/review bar   |
| Engines     | Card overview + version table consistency, install/build dialogs, filters, empty states                |
| Proxy       | Tabs, overview/settings/keys, shared sections, GPU admission card, request table                       |
| System      | Shared section tabs, Hardware/Storage/Diagnostics/Backup/History visual parity                         |
| Audio       | Model/context header, task tabs/panels, result states, voice/reference controls                        |
| NotFound    | Shared shell, meaningful heading, route back to Models                                                 |

## Foundation primitives

All primitives live in `frontend/src/components/common/` and are global CSS in
`frontend/src/styles/_patterns.css` (imported last by
`frontend/src/styles/index.css`). Tokens live in
`frontend/src/styles/_variables.css` under "UI foundation".

### PageSection

`PageSection.vue` replaces every hand-built section/card.

| Prop / slot   | Behavior                                                                                  |
| ------------- | ----------------------------------------------------------------------------------------- |
| `title`       | Section `h2` (default). Wires `aria-labelledby` on the `<section>`.                       |
| `headingTag`  | `h1`–`h6`; use `h3` for a sub-section inside a section.                                   |
| `description` | Muted one-line section purpose. Keep it to a sentence; detail belongs in an info control. |
| `id`          | Anchor target for deep links (for example `#storage`). The heading id is `<id>-title`.    |
| `variant`     | `default` card, `plain` (no surface, for a page-level band), `compact` (dense panels).    |
| `#title`      | Custom heading markup; the component then does not invent `aria-labelledby`.              |
| `#meta`       | Tags/status next to the heading.                                                          |
| `#actions`    | Section-level actions; one primary action per section.                                    |
| default slot  | Body; becomes `.page-section__body`.                                                      |

CSS: `.page-section`, `.page-section--plain`, `.page-section--compact`,
`.page-section__header`, `.page-section__heading-group`, `.page-section__title`,
`.page-section__description`, `.page-section__meta`, `.page-section__actions`,
`.page-section__body`, `.page-subheading`.

### SectionTabs

`SectionTabs.vue` replaces page-local tab strips and section links.

- Tabs mode (`items` without `to`): `role="tablist"`, `role="tab"`, roving
  `tabindex`, arrow/Home/End keys via the existing
  `composables/useRovingTabs.js`. Pass `idPrefix` to keep a page's existing
  `*-tab-<id>` / `*-panel-<id>` wiring; omit it for strips without panels.
- Links mode (any item has `to`): `<nav>` with `RouterLink`s and
  `aria-current="page"` on the selected section, matching System's hash
  sections.
- `variant`: `underline` (Proxy/System shape) or `pills` (ModelConfig/Audio
  shape). `responsiveGrid` collapses to two columns instead of scrolling.
- Counts use the `count` item field and render in `.section-tabs__count`.

Migration note for ModelConfig: keep the existing engine-descriptor tab set
(`ENGINE_DESCRIPTORS[...].configTabs`) as `items`; only the strip markup and
keyboard contract move to the primitive.

### PageToolbar

`PageToolbar.vue` is the one toolbar shell (search, filter group, summary,
actions). Pair it with the existing `.compact-fields` / `.compact-toolbar`
density classes rather than adding page-local control sizes.

- `label` gives the toolbar group an accessible name.
- `plain` drops the surface where the toolbar already sits inside a card.
- Slots: `#search`, default (filters), `#summary`, `#actions`.

CSS: `.page-toolbar`, `.page-toolbar--plain`, `.page-toolbar__search`,
`.page-toolbar__group`, `.page-toolbar__summary`, `.page-toolbar__actions`.

### StateBanner

`StateBanner.vue` is the structured form of the existing `.state-banner`
class. It keeps the base warning palette and adds severity modifiers.

- `severity`: `info`, `success`, `warning` (default), `error`.
- `role`: errors default to `alert`; other severities default to `status`
  with `aria-live="polite"`. Override for recoverable errors and for
  non-blocking warnings that should not interrupt.
- `icon`: default per severity; pass `""` to hide it.
- Slots: default message, `#actions` for retry/next-step buttons.

A page must not mix `.state-banner` and PrimeVue `Message` for the same class
of state after migration; `Message` stays for inline field/row guidance where
a banner would be too heavy.

### Loading and empty states

`LoadingState.vue` and `EmptyState.vue` were already shared. `EmptyState` now
accepts `role="alert"` for failed loads and `headingTag="h1"` for full-page
failure surfaces; the heading validator accepts `h1`–`h6`.

### Forms, tables, dialogs, navigation actions

- `.page-form`, `.page-form__field`, `.page-form__field--full`,
  `.page-form__label`, `.page-form__hint`, `.page-form__error`,
  `.page-form__actions`, `.page-form--single`. Two columns from 900 px; one
  column and full-width action buttons on narrow/touch layouts.
- `.table-shell`, `.table-shell__scroll`, `.table-shell__state`,
  `.table-shell__footer`. The scroll region owns `overflow-x`; sticky headers
  use `--table-header-bg`.
- `.dialog-stack` and `.dialog-actions` standardize dialog body/footer
  stacking; the width scale stays `dialog-width-xs|sm|md|lg`.
- `.page-action-link` is a `RouterLink` that reads as the page's next action
  while keeping link semantics (used by NotFound). Do not wrap a `Button` in a
  link.

### Tokens

Added to `_variables.css` (theme-independent; colors continue to use the
existing semantic tokens):

| Token                                                                                | Purpose                                  |
| ------------------------------------------------------------------------------------ | ---------------------------------------- |
| `--text-lg`, `--text-xl`, `--text-2xl`                                               | Complete the existing `xs/sm/base` scale |
| `--section-radius`, `--section-padding`, `--section-gap`, `--section-header-gap`     | Section geometry                         |
| `--section-title-size`, `--section-title-weight`, `--subsection-title-size`          | Section type scale                       |
| `--toolbar-radius`, `--toolbar-padding`, `--toolbar-gap`, `--toolbar-control-height` | Toolbar geometry                         |
| `--form-gap`, `--form-column-gap`                                                    | Form rhythm                              |
| `--table-radius`, `--table-header-bg`                                                | Table shell                              |

Rules 8 still applies: page migrations must use these tokens and existing
color/status tokens, never raw hex values or one-off pixel scales.

## Page-by-page audit

Cross-cutting patterns in the current tree (all confirmed in source):

- **Cards:** `.card`, `.model-card`, `.config-card`, `.config-section`,
  `.overview-card`, `.system-card`, `.proxy-card`, `.ev-section`,
  `.engine-card`, `.attention-card`, `.metric-card`, `.backup-action-card`,
  `.voice-preset-card` — at least thirteen card vocabularies.
- **Section headers:** `.page-header`, `.section-head` (Proxy),
  `.ev-section-header` (Engines), `.system-section-head` (System),
  `.config-section-heading` (ModelConfig), and bare `.section-label` divs
  (ModelConfig/Audio).
- **State:** `.state-banner`, PrimeVue `Message`, `.token-warning`,
  `.persistence-notice`, `.system-error`, `.empty` text nodes, `.empty-state`
  (legacy global), `.empty-state-mini`, plus `EmptyState`.
- **Toolbars:** `.library-toolbar`, `.search-bar.compact-toolbar`,
  `.engine-filters.compact-toolbar`, `.request-filters.compact-toolbar`,
  `.config-toolbar`, `.runtime-actions` — five distinct shapes behind the
  shared `.compact-*` density.
- **Tables:** Model library `role="table"` spans, legacy Search `.result-row`
  divs, Proxy native `<table>`, Engines `VersionTable` div rows — four table
  implementations, none using a shared table shell.
- **Tabs:** `.proxy-tabs` buttons, `.system-tabs` links, `.config-section-tabs`
  buttons (ModelConfig), and a duplicated `.config-section-tabs` markup in
  Audio — four strips, three visual treatments.
- **Card padding/radius:** `--spacing-lg`/`--radius-xl` (`.card`),
  `--panel-padding`/`--radius-lg` (`.proxy-card`, `.system-card`),
  `0.75rem 0.9rem` (`.config-card`), `1rem`/`--radius-md` (`.system-card`
  override) — same visual role, four sizes.

### Cross-cutting: type, spacing, responsive, themes

These apply to every page and are the reason migrations must use tokens
instead of page-local values.

- **Page width.** Six workspaces opt into `page-shell--wide` (Models L2,
  Search L2, ModelConfig L2, Engines L2, Proxy L2, Audio L2). System alone uses
  the capped default shell (`SystemView.vue` L2), so its content column is
  narrower than every other page at desktop widths. The shell contract allows
  both; the decision (wide only for table-heavy workspaces) must be recorded
  before page migrations choose one.
- **Page header collapse.** `.page-header` is defined twice with opposite
  narrow-screen behavior: `_utilities.css` L15–61 and L245–267 make it a column
  at ≤768 px, then `_workspace.css` L78–101 and L462–475 put it back in a row
  with auto-width actions. The final behavior depends on import order, not on
  an explicit contract. Long titles plus three actions (Audio, Proxy, Models)
  leave little room at 320 px even though the overflow tests pass.
- **Section type scale.** Section titles are `1rem` (`.section-head h2` in
  Proxy L2763–2766, `.ev-section-title h2` in Engines L3883–3889,
  `.system-section-head h2` in System L208–211), `clamp(1rem, 1.2vw, 1.15rem)`
  (`ModelConfigSectionHeading.vue` L22–28), and now
  `--section-title-size: 1.0625rem` (`PageSection`). Body/caption sizes
  scatter across `0.8/0.82/0.85/0.875/0.9/0.9375rem`; the new `--text-lg`
  token only helps once pages stop hardcoding sizes.
- **Control height.** The shell sets `.p-button` to `2.75rem`
  (`_workspace.css` L168–175), but ModelConfig compresses page buttons to
  `2rem` and small buttons to `1.75rem` (`_model-config-pane.css`
  L610–631) and the config pane shrinks inputs to `2rem`. Tables and dense
  toolbars use `2.125rem` (`.compact-*`, `_density.css`). The same visual
  primary action is four different heights depending on the page.
- **Theme coverage.** Most newer surfaces use semantic tokens, but older
  rules bake in colors: status tags in `_components.css` L140–156 and
  L396–424 use raw `rgba(...)` rather than `--status-*` tokens; the icon-only
  button hover/focus in `_components.css` L1245–1257 uses
  `rgba(255, 255, 255, ...)` and a blue focus ring that are only sensible in
  dark mode; `_model-config-pane.css` and `ModelConfig.vue` keep stale hex
  fallbacks inside `var()` (`#161b2e`, `#22d3ee`, `#9ca3af`) that drift from
  the themed indigo palette; Proxy's active tab uses the PrimeVue runtime token
  `--p-primary-color` (L2713–2716) while every other active state uses
  `--accent-primary`. The brand orbit in `AppHeader.vue` L89–128 is the one
  intentional hardcoded palette (identity), and should stay documented as
  such.
- **Responsive strategies differ per page.** System tabs become a two-column
  grid at ≤600 px (System L181–192); Proxy tabs scroll horizontally with
  snap points (Proxy L2690–2716); config tabs wrap and then stretch at ≤768 px
  (`_model-config-pane.css` L592–607); Audio defines its own two-column task
  grid at ≥900 px (L1674–1681); Engines' card grid and filter bar rely on
  `auto-fill` + `.compact-*`; Proxy's request table is the only table with a
  fixed `min-width: 48rem` (L2717–2732). The migration should converge on
  `SectionTabs` responsive behavior, `.page-form` breakpoints, and
  `.table-shell` scroll rather than preserving four strategies.
- **Touch targets.** `_workspace.css` L306–317 raises coarse-pointer targets
  to 44 px, but page-local button overrides (ModelConfig above, Engines dialog
  `size="small"` buttons, Proxy row actions) can shrink them again. The
  standards touch journey audits `/models`, `/search`, `/audio`, `/engines`,
  `/proxy` and `/system#config-backup` at 320 px but not the ModelConfig
  configuration page or the Proxy request table, so an override there would not
  be caught automatically.

### Models — `frontend/src/views/ModelLibrary.vue`

Current structure: `page-shell page-shell--wide`; `PageHeader` with model-count
tag, refresh icon action and "Add model" primary (L2–31); custom token warning
(L34–45); `SetupChecklist`; two state banners (L49–57); `LoadingState` (L59);
load-failure `EmptyState` (L61–68); `FirstModelPanel` for the empty library
(L70); custom `.library-toolbar` with native search input and native selects
(L73–137); bespoke results summary row with select-all (L139–155); custom
filter toggle (L91–99); raw-button view switch (L129–136); chip row;
no-match `EmptyState` (L192–198); list view built from `role="table"` spans
(L200–...); card grid otherwise.

Inconsistencies:

1. The toolbar is native HTML (`input`, `select`, `button`) with
   `.library-toolbar` styles (L1346+), while Search uses PrimeVue
   `InputText`/`Select`/`Button` and Engines uses a third native-control bar.
2. `.token-warning` is defined in this view (L1349–1370) and redefined with
   different responsive rules in ModelSearch (L3213–3235, L3609+).
3. Two different stale/error banner shapes (L49–57) instead of one component;
   both include a retry action, so migration is mechanical.
4. The filter toggle is a custom button with `aria-expanded` (L91–99), while
   Search uses a PrimeVue outlined button with the same attributes.
5. The view switch is two unstyled buttons with `aria-pressed`; no shared
   segmented-control look; labels are "Cards"/"List".
6. Table view is a hand-built grid with `role="table"` and `span`
   `role="columnheader"` (L201–212); sizes `library-list__size-heading` and
   actions have custom widths. Search's legacy result table and Proxy's request
   table solve the same problem differently.
7. The empty-library first-run state uses `FirstModelPanel` (its own card)
   while the no-match state uses `EmptyState`; both are "empty library" states
   with different visuals.
8. Card/section styling: model cards come from global `.model-card`
   (`_components.css`), not from a section primitive, and trailing actions use
   page-local `.model-actions` rules.

Migration map: `PageHeader` stays; toolbar → `PageToolbar` with
`#search`/default filters/`#summary`/`#actions`; token warning → `StateBanner
severity="info"` (one shared "Hugging Face token required" component used by
Models and Search); stale/error → `StateBanner`; filter toggle → toolbar
action; view switch → one segmented control primitive or shared classes; list
view → `.table-shell`; `FirstModelPanel` → `EmptyState` with "Search models"
next action or keep as a section but wrap in `PageSection`.

### Search — `frontend/src/views/ModelSearch.vue`

Current structure: `page-shell page-shell--wide`; `PageHeader` title and
description only (L2–6); `ModelSearchControls` own `<section
aria-label="Catalog search">` with search input, task select, Search primary,
filters toggle, reset text button and "Import audio bundle" text button
(`ModelSearchControls.vue` L1–76); token warning (L37–49); unavailable-provider
banner without retry (L51–59); stale-results banner with retry (L61–64); two
`EmptyState`s (L66–80); a plain `.search-hint` paragraph for the pre-search
state (L82–84); inline `LoadingState` (L86); catalog card grid with its own
`.results-header`, sort and pagination (L88–171); legacy `.results-table`
renderer with a second `.results-header` (L174–...).

Inconsistencies:

1. **No page-level primary action.** "Import audio bundle" is the only
   creation path and is a tertiary text button buried in the search toolbar
   (`ModelSearchControls.vue` L67–75); every other workspace puts its creation
   action in `PageHeader` (Models "Add model", Audio "Start", Engines none).
2. The pre-search state is prose (L82–84) with no next action; rule 3/10 wants
   an `EmptyState` (or a guidance section) with a concrete next step.
3. Two result renderers with duplicated `.results-header`/sort markup
   (L98–123 vs L175–190). The legacy renderer is documented as a
   compatibility adapter but is still reachable, so both must migrate or the
   dead one must be removed with its tests.
4. `LoadingState` uses the `inline` form here but the centered form on Models,
   Audio and Proxy for the same "first load of the page" state.
5. Provider unavailability is rendered twice: once as a banner (L51–59) and
   once as tags in the results header (L113–121).
6. `.search-hint` and `.results-table`/`.result-row` styles are page-local;
   none of the result surfaces use a shared table shell.
7. `PageHeader` has no meta (no result count) while Models/Proxy/Audio put
   count/status tags in `#meta`.

Migration map: `PageHeader` gains the import action (primary) and result-count
meta; `ModelSearchControls` becomes `PageToolbar` (search/filters/actions
split) with `#actions` holding Search and Import; provider banner →
`StateBanner severity="warning"`; initial state → `EmptyState` ("Search by
model name…" with a focus-search or example action); legacy renderer either
migrates to `.table-shell` or is removed with its tests; one loading form.

### ModelConfig — `frontend/src/views/ModelConfig.vue` and `components/models/*`

Current structure: `page-shell page-shell--wide`; loading/empty states before
the model resolves (L3–23); `ModelConfigHeader` wraps `PageHeader` with a
custom `#title` slot containing the `h1` and all meta tags, and a `#start`
back button (`ModelConfigHeader.vue` L2–59); `ModelConfigActionBar` is a
page-local `section role="region" aria-label="Configuration actions"` bar
(L37–67, `ModelConfigActionBar.vue` L2–114); four `state-banner` blocks
(L69–97); a custom `.persistence-notice` paragraph with raw buttons (L99–108);
tab strip `.config-section-tabs` driven by overview + engine descriptor tabs
(L110–130, `pageTabs` L1117–1125); section heading component used only for
Runtime (L185–188) and Launch (L311–314); `.config-card` blocks; `.section-label`
divs for the Overview engine picker (`ModelConfigOverview.vue` L42) and the
Launcher preview (L379–427); many inline `Message` components (L233–281,
L356+); dialogs for command preview, templates, and change review.

Inconsistencies:

1. **Header shape differs.** Meta tags live inside the `#title` slot instead of
   `#meta`, and the back control uses `#start` — the only page using that slot.
   Two tags render inside a custom `.config-page-title` wrapper rather than the
   shared `.page-header__meta` row.
2. **Two heading grammars.** `ModelConfigSectionHeading` renders `h2` +
   description for Runtime/Launch, but the Overview engine picker and the
   Launcher preview use `div.section-label` (uppercase in
   `_model-config-pane.css`, overridden back to sentence case in
   `ModelConfig.vue` L3392–3403), and `ModelConfigOverview` mixes `h3` section
   titles with `.section-label`. A page-local `.config-card` (L3366–3371)
   shadows the shared `.config-card` in `_model-config-pane.css` L29–34 with
   different padding.
3. **Action bar drift.** Save/Apply/Start/Review are one bar above the content;
   Proxy has a save group + lifecycle group inside the Settings section;
   Audio puts per-task actions inside each card. Labels differ
   ("Save changes" here vs "Save settings" in Proxy) and the unsaved-changes
   prompt is shared, which is the one part that is already consistent.
4. **Persistence state is its own idiom.** `.persistence-notice` includes
   retry/refresh buttons and `role="status"`, but is not a `state-banner`, so
   it has no warning/error semantics and no shared layout.
5. **Tabs are descriptor-driven.** The plan's fixed "Engine / Runtime / Names /
   Launch" nav does not match the current `Overview + engine configTabs`
   contract (`ENGINE_DESCRIPTORS`, L1117–1125). Migration must render the
   existing items through `SectionTabs` rather than hardcoding four labels;
   the "Names & routing" summary lives in `ModelConfigOverview`.
6. **Forms.** Runtime parameters use `.params-grid`
   (`_model-config-pane.css`); the config pane redefines density, and page
   fields use several wrappers (`.param-field`, `.form-field`, `.toggle-field`,
   `.workbench-block`). There is no shared label/hint/error contract for the
   long tabs.
7. Sections are visually cards but have no section descriptions or actions
   slots; guidance is expressed through `Message` components at various
   depths.

Migration map: `ModelConfigHeader` → `PageHeader` with `#start` only for back,
`#meta` for tags, `#actions` for updates/audio; `.config-card` → `PageSection`
(h2 headings, optional actions); Overview/Names `.section-label` → `PageSection`
with `headingTag="h3"` or `.page-subheading`; tab strip → `SectionTabs` with
`idPrefix="config"` and the descriptor tab list; `.persistence-notice` →
`StateBanner severity="error" role="status"`; action bar → one `PageToolbar`
save-bar variant with the same labels as Proxy (`Save`, `Apply`, `Review`).

### Engines — `frontend/src/views/EnginesView.vue`

Current structure: `page-shell page-shell--relaxed page-shell--wide`;
`PageHeader` title/description only (L2–6); one collapsible `.ev-section`
(L9–73) whose header is a full-width toggle button and whose actions hold
refresh (L31–40); `EngineFiltersBar` with native selects and a link-style reset
inside the section (L44–48, `EngineFiltersBar.vue`); `.engine-grid` of
`EngineCard`s plus descriptor-driven `GenericEnginePanel`s (L49–70); a single
large `Dialog` hosting per-engine sections styled with `ev-section
ev-section--modal` (L258–399); `VersionTable` for installed versions (div rows,
not a table); many toast-only failure paths.

Inconsistencies:

1. **Section is not a section.** `.ev-section` (L3820–3915) is a custom
   card with its own header/actions/body paddings and a gradient body, then
   reused inside dialogs (`ev-section--modal`) for a different purpose. This is
   exactly the page-local lookalike rule 2 removes.
2. **Refresh placement.** Models/Proxy/Audio put refresh in `PageHeader`
   actions; here it is inside the section's actions, and the page header has no
   actions at all.
3. **Toolbar.** `EngineFiltersBar` is a third native-control toolbar with an
   inline `Reset filters` text button, no search and no summary; filters sit
   inside the section body rather than the page toolbar.
4. **No page-level state contract.** There is no `LoadingState`, `EmptyState`
   or `state-banner` anywhere in the view (checked by import and template);
   filtered-out cards use `v-show` (L54) leaving a blank grid instead of an
   empty state, and load failures surface only through toasts and per-dialog
   messages.
5. **Empty states are duplicated minis.** `VersionTable` styles
   `.empty-state-mini` (L273–287) and `EnginesView` styles it again
   (L4168–4180) with different typography; the dialog version tables use
   `empty-message` strings (L308, L377, L473).
6. **Dialogs.** One maximizable `Dialog` contains each engine's entire
   install/build UI; nested `ev-section` panels carry their own headings. There
   is no shared dialog body/footer structure, and width classes are applied per
   dialog.
7. **Cards.** `.engine-grid` + `EngineCard` have `:deep(.engine-card)` sizing
   (L3924+), separate from model/proxy/system cards.
8. Heading levels vary: `h2` inside the section toggle, `h2` inside dialog
   panels, and component-local headings in `VersionTable`/`GenericEnginePanel`.

Migration map: section → `PageSection` with a proper `#actions` (refresh,
expand) and an accessible collapse control if the collapse behavior stays;
filters → `PageToolbar`; add `LoadingState` while the catalog/versions load and
an `EmptyState` when filters exclude every engine; version list → `.table-shell`
with the same empty state inside the shell; dialogs → shared
`EngineDialogHeader` + `.dialog-stack`/`.dialog-actions` and the width scale;
`EngineCard` visuals → shared section/card tokens.

### Proxy — `frontend/src/views/ProxyView.vue`

Current structure: `page-shell page-shell--wide`; `PageHeader` with status
tags and refresh (L2–19); `UnsavedChangesDialog`; `FreshnessStatus`; three
PrimeVue `Message` banners for error/stopped/restart (L29–47); page
`LoadingState` (L49); custom `.proxy-tabs` tablist with manual
`navigateTab` (L52–69, L1280–1295); `.proxy-panels` scroll container plus
`:global` layout overrides (L2634–2659); `.proxy-card` sections with
`.section-head` headers (L76+, styles L2744–2776); a save/lifecycle action
cluster (L87–130); keys/events/requests `EmptyState`s and inline `Message`s;
`RequestFiltersBar` inside a `.proxy-card` (L1114–1122); native `<table>` in a
scroll wrapper with sticky header (L1126–1187); `.empty` paragraphs for
no-request/no-activity (L1105, L1188).

Inconsistencies:

1. **Two banner systems.** Page-level errors/stopped/restart use PrimeVue
   `Message` (L29–47), while Models/Search/Config/Audio use `.state-banner` for
   the same class of state. Restart-required is a warning with an action, so it
   maps directly to `StateBanner`.
2. **Tab strip is bespoke.** `.proxy-tabs` reimplements the tab contract and
   hand-rolls arrow/Home/End navigation (L1280–1295) although
   `composables/useRovingTabs.js` exists; System and ModelConfig/Audio have
   their own strips too. The tab count badge is a separate `.tab-count` style.
3. **Section header/action split.** `.section-head` renders the `h2` and
   description, but actions live in a following `.runtime-actions` row, so a
   section's action is not in the section header (unlike `PageSection`).
4. **Table state outside the shell.** The request table has no empty/loading
   row inside it; emptiness is an `.empty` paragraph after the table (L1188)
   and loading is a toolbar button spinner plus a separate banner.
5. **Page height exception.** Proxy is the only page that overrides the layout
   wrapper/main with `:global` rules (L2634–2649) to create an internal scroll
   region. A migration must preserve this exception or move it into the shell
   deliberately; it must not leak into other pages.
6. **Three card-like levels.** `.proxy-card` sections, `.proxy-summary` grid,
   and nested request rows each define their own padding/border.
7. **Filter toolbar** is a `<form role="search">` with a `.field` control set
   (shared density) but sits inside the section and has its own action group.

Migration map: tabs → `SectionTabs idPrefix="proxy"` (existing URL/query state
unchanged); messages → `StateBanner`; `.proxy-card` → `PageSection` with
`#meta` for counts and `#actions` for save/lifecycle; request table →
`.table-shell` with `LoadingState`/`EmptyState` inside the shell and the
pagination in `.table-shell__footer`; `.runtime-actions` → shared save-bar
pattern (same labels as ModelConfig: `Save settings` stays the settings label
if chosen as canonical); keep the page-height exception documented in the shell
CSS rather than duplicated.

### System — `frontend/src/views/SystemView.vue`

Current structure: default `page-shell`; `PageHeader` without meta/actions
(L2–6); `.system-tabs` RouterLink section nav (L7–26) with page-local styles
(L162–192); a `.system-card` wrapper around `NeedsAttentionPanel` (L27–29),
another around `ConfigBackupPanel` (L73–75) and one around
`ConfigHistoryPanel` (L76); Hardware renders a bare `HardwarePanel`; the
Diagnostics section defines its own `system-section-head` with an `h2` and a
download link (L35–40) and a bespoke `dl.health-grid` (L41–61); an error
paragraph uses `.system-error` with `role="alert"` (L62–67).

Inconsistencies:

1. **Section nav is bespoke links.** `.system-tabs` duplicates the link-strip
   contract and CSS; `SectionTabs` links mode exists for exactly this shape.
2. **Nested cards.** The page wraps panels in `.system-card`, while the panels
   render their own cards (`.attention-card`, `.backup-action-card`,
   `.metric-card` in `HardwarePanel`). Result: double borders/padding and
   different radii; Hardware has no page-level wrapper at all. WS2 adds a fifth
   `Storage` section, so this must be settled before that lands.
3. **Only one section is labelled.** Service health has `aria-labelledby`;
   Hardware, Backup, History and the attention panel are unlabelled sections
   or plain divs, so landmark navigation changes per section.
4. **Heading source varies.** Diagnostics uses an `h2` in page markup;
   Backup/History delegate headings to child components (`heading-tag` prop),
   Hardware renders metric headings; no consistent `h2` per section.
5. **No page-level state contract.** There is no `LoadingState`,
   `EmptyState` or `state-banner`; failures are `.system-error` prose or
   child-local messages. Refresh is not offered anywhere on the page even
   though status can be stale (App refreshes on a timer and on visibility).
6. **Section URL contract** (`#hardware`, `#config-backup`, `#history`, and the
   router's legacy `#system`/`#ev-section-system` redirects) must be preserved
   when the strip moves to `SectionTabs`; WS2 adds `#storage`.

Migration map: `PageHeader` keeps title/description and gains refresh (status
is shared with the shell, so a shared "Refresh status" action is enough);
`.system-tabs` → `SectionTabs` links mode with the same `to` targets; each
section → `PageSection` with `id` matching its hash and a single `h2`; remove
`.system-card` wrappers and let panels render body content inside the section;
`.system-error` → `StateBanner severity="error"`; add page-level loading/empty
states only where the data is genuinely absent (not for panels that render
their own honest states).

### Audio — `frontend/src/views/AudioWorkspace.vue`

Current structure: `page-shell page-shell--relaxed page-shell--wide`;
`PageHeader` title only, with meta tags and refresh/configure/start actions
(L2–46); page loading and four `EmptyState`s for boot/disabled/missing/no-model
(L48–88); state banners and `Message`s for workspace/config/contract/reference
errors (L91–124); a `config-card config-card--compact` model bar with a
`.section-label` "Model" (L125–173) including reference-audio buttons and a
file input inside a hint paragraph (L147–172); duplicated
`.config-section-tabs` markup with `onRovingTabKeydown` (L175–195); per-task
`.config-card` panels with `.section-label` headings (L219–230, L329–...,
L420–..., L448–...); inline `Message`s for ASR/no-fields states (L426, L450);
page-local two-column layout at ≥900 px via
`.audio-task-layout:has(> :nth-child(2))` (L1674–1681); voice/reference
controls inside `AudioModelConfig` with their own `Message`s.

Inconsistencies:

1. **No page description/help.** `PageHeader` has a title only, unlike every
   other workspace; the page's purpose and audio.cpp contract caveats are
   explained only by inline messages and the model bar.
2. **Primary action is state-dependent and split.** Refresh, Configure and
   Start all live in the header; per-task Generate/Synthesize/Transcribe
   actions live inside each card, so "one primary action per page" has no clear
   reading. Configure is outlined while Start is filled; when a model is
   running the header has no primary action.
3. **Duplicate tab markup.** The tab strip copies ModelConfig's markup and
   relies on the shared `_model-config-pane.css` classes; the two pages must be
   migrated together to `SectionTabs` or they will diverge.
4. **Section labels instead of headings.** Speech/Transcribe/Transcript/Model
   are `div.section-label` elements, so the page has no `h2` hierarchy and
   screen-reader heading navigation cannot reach the task panels.
5. **Mixed state idioms on one page.** Five `.state-banner` blocks plus four
   `Message` components, including `Message severity="info"` for contract
   guidance and `Message severity="warn"` for a not-running model.
6. **Actions inside hint text.** Reference-audio buttons and the hidden file
   input sit inside `<p class="config-muted-hint">` (L147–172); invalid HTML is
   avoided only because the buttons are inline-level.
7. **Model selection and request controls are adjacent but not grouped.**
   Model/engine version/inference id/reference audio (workspace scope) and task
   text/parameters (request scope) share the same visual level, making the
   model/context header hard to read.
8. **Page-local responsive layout.** `.audio-task-layout:has(...)` is the only
   page that defines its own two-column task grid; it must either become a
   shared layout variant or be documented as an audio-specific exception.

Migration map: `PageHeader` gains a one-sentence description/metadata and a
stable primary action (Start when stopped, otherwise the task action inside the
active panel — document the rule); model bar → `PageSection variant="compact"`
(or a shared "context bar" primitive) with `#meta` for family/task/runtime and
`#actions` for configure/refresh; model/request split → two `PageSection`s
(context and task); tab strip → `SectionTabs variant="pills"`; `.section-label`
→ `PageSection`/`.page-subheading`; state banners → `StateBanner`; ASR/no-fields
messages → `StateBanner` or inline `Message` with one documented rule; keep the
two-column task layout only if the shared form/section grid cannot express it.

### NotFound — `frontend/src/views/NotFoundView.vue` (migrated in this branch)

Before: a bare `<section class="page-shell">` with `h1`, one sentence and a
plain `RouterLink` — no `PageHeader`, no description/help control, no
action styling, and no tests.

After (reference implementation):

- `page-shell` + `PageHeader` (`h1` "Page not found", description available
  through the shared info control).
- `EmptyState` with a heading, description and next action.
- `.page-action-link` router link to Models, keeping link semantics and the
  browser journey's accessible name ("Return to Models").
- `frontend/src/views/NotFoundView.test.js` asserts the shared shell,
  `PageHeader`, `EmptyState` and link target.

## Reference implementation: application shell

`frontend/src/App.vue` route-failure state (previously a raw `<section
role="alert">` with an `h2` and two bare `<button>`s) now renders the shared
`EmptyState role="alert" heading-tag="h1"` with PrimeVue `Button`s for "Retry
navigation" and "Reload Studio". This keeps the recovery contract from
`frontend/e2e/recovery.spec.mjs` (heading and both buttons) while using the
same state primitives as pages. `App.test.js` covers the rendered failure
state; no route behavior changed.

The shell header/footer already use shared layout classes; no markup change
was needed there. Their visible chrome stays governed by
`_workspace.css`/`_base.css` tokens.

## Migration sequence and guardrails

1. This branch (`ws/ui-foundation`) delivers primitives, tokens, the audit, and
   the NotFound/shell reference. It does not touch feature pages.
2. WS1/WS2 build new panels (GPU admission, Storage) with `PageSection`,
   `SectionTabs`, `StateBanner`, `PageToolbar` from the start. If a feature
   branch was cut before this foundation merges, its new panels are treated as
   page-local sections and migrated by the corresponding `ws/ui-pages-*`
   branch, like every other section.
3. `ws/ui-pages-models`, `ws/ui-pages-engines`,
   `ws/ui-pages-proxy-audio` migrate pages after feature branches land, using
   NotFound and the documented reference as the pattern.
4. Every page migration keeps the existing routes, URL state, accessible names
   and browser journeys; it replaces styling/markup only.
5. Run `frontend/e2e/ui-refactor.spec.mjs` for refreshed screenshots after
   migrations and record intentional visual changes; do not commit incidental
   PNG churn.
6. Remove a page-local class only when no remaining selector/test references
   it. Update `docs/frontend-standards.md` only if an interaction contract
   changes (it should not).

## Acceptance checklist

Foundation (this branch):

- [x] Shared `PageSection`, `SectionTabs`, `PageToolbar`, `StateBanner`
      primitives with colocated unit tests.
- [x] Shared tokens for section/toolbar/form/table geometry and type scale.
- [x] One shared CSS entry point for the patterns, imported after the existing
      layers.
- [x] NotFound migrated to the shared shell as the reference page.
- [x] App-shell route failure migrated to `EmptyState` with tests.
- [x] `docs/ui-alignment.md` records the audit, rules and checklist.
- [x] Existing routes, API shapes and URL state untouched.
- [x] Page migrations (Models, Search, ModelConfig, Engines, Proxy, System, Audio).

Per-page migration acceptance (apply to each of Models, Search, ModelConfig,
Engines, Proxy, System, Audio):

- [ ] Exactly one `PageHeader` with title, description and one primary action.
- [ ] Every section is a `PageSection` (no page-local card/section lookalike
      remains for that page).
- [ ] Loading, empty, stale and error states use `LoadingState`, `EmptyState`
      and `StateBanner`; every failed/empty state offers a next action.
- [ ] One toolbar pattern (`PageToolbar` + `.compact-*` density) and one table
      shell with contained horizontal scroll.
- [ ] Labels are sentence case and consistent (`Save`, `Save settings`,
      `Retry`, `Refresh`).
- [ ] Page `h1` → section `h2` → sub-section `h3`; no `div` standing in for a
      heading.
- [ ] Tokens only; both themes pass the axe run.
- [ ] No horizontal overflow at 320/390 px; actions reachable; tables scroll
      locally.
- [ ] Unsaved-change/Stay/Discard/Save behavior unchanged.
- [ ] Existing browser journeys and a11y scans pass (`npm run test:browser`,
      `npm run test:a11y`), with a screenshot review table in the PR.

Release acceptance (from the plan, unchanged):

- [x] Unit tests for the migrated pages/components.
- [ ] Browser journeys and accessibility checks pass. Chromium journeys and the
      `standards` axe run pass locally; the full integrated Firefox/WebKit run is
      recorded with the release evidence.
- [x] No page-local duplicate section styles remain after all page migrations.
- [x] No horizontal overflow at the plan's widths in the captured Chromium
      reviews (320/390/1440; Models, ModelConfig, Proxy, Audio).
- [x] Screenshot review table recorded in the integration PR.

## Verification

Foundation checks run on this branch:

```bash
npm run lint:frontend
npm run style:check
npm run format:check
npm run test:frontend -- frontend/src/components/common/PageSection.test.js \
  frontend/src/components/common/SectionTabs.test.js \
  frontend/src/components/common/PageToolbar.test.js \
  frontend/src/components/common/StateBanner.test.js \
  frontend/src/views/NotFoundView.test.js frontend/src/App.test.js
```

Not covered by the foundation branch:

- Playwright screenshot review (`frontend/e2e/ui-refactor.spec.mjs`) is
  meaningful only after page migrations. Its shared-headers journey does not
  visit the NotFound route; `frontend/e2e/recovery.spec.mjs` covers the
  NotFound heading and the "Return to Models" link.
- Browser/a11y journeys for the migrated pages remain in the page-workstream
  acceptance.
- No hardware-dependent behavior is involved in this workstream.
