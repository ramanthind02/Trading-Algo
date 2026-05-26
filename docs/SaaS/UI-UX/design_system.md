# Design System (MVP)

Implementation guide for **QuantFoundry-Web**. Canonical brand: [brand.md](brand.md) · [`logo (1)/Brand.html`](logo%20(1)/Brand.html) v1.0.

**App theme:** dark shell — **Ink** surfaces, **Paper** text, **Ember** accents. No blue/purple SaaS defaults.

## CSS variables (copy into app)

```css
:root {
  /* Brand tokens */
  --ink:        oklch(0.18 0.01 270);
  --ink-2:      oklch(0.28 0.012 270);
  --ink-3:      oklch(0.38 0.012 270);
  --paper:      oklch(0.97 0.005 80);
  --paper-2:    oklch(0.94 0.006 80);
  --line:       oklch(0.86 0.008 80);
  --line-2:     oklch(0.80 0.008 80);
  --muted:      oklch(0.55 0.01 270);
  --ember:      oklch(0.66 0.17 45);
  --ember-deep: oklch(0.54 0.16 38);

  /* App aliases (dark UI) */
  --bg-page:      var(--ink);
  --bg-surface:   var(--ink-2);
  --bg-elevated:  var(--ink-3);
  --text-primary: var(--paper);
  --text-secondary: oklch(0.78 0.01 270);
  --text-muted:   var(--muted);
  --border-subtle: oklch(0.28 0.012 270);
  --border-strong: oklch(0.38 0.012 270);
}
```

| Token | OKLCH | Hex (ref) | App usage |
|-------|-------|-----------|-----------|
| Ink | `0.18 0.01 270` | `#22232B` | Page background, primary button text |
| Ink-2 | `0.28 0.012 270` | — | Cards, sidebar |
| Ink-3 | `0.38 0.012 270` | — | Hover, inputs, strong borders |
| Paper | `0.97 0.005 80` | `#F7F5EF` | Primary text on dark |
| Paper-2 | `0.94 0.006 80` | — | Light marketing backgrounds only |
| Muted | `0.55 0.01 270` | — | Labels, captions |
| Ember | `0.66 0.17 45` | `#D57044` | Brackets, cursor, links, primary CTA |
| Ember-deep | `0.54 0.16 38` | — | Hover, pressed, validation zone tint |

Support neutrals between Ink and Paper use hue **270** (cool slate) on the dark end and **80** (warm cream) on the light end — never flat gray.

## Ember (primary accent)

| Token | OKLCH | Usage |
|-------|-------|-------|
| `ember` | `0.66 0.17 45` | Primary buttons, links, focus, active nav, stepper current |
| `ember-deep` | `0.54 0.16 38` | Hover, active press, deeper zone segment |
| `ember-muted` | `0.66 0.17 45 / 0.12` | Selected row, subtle fill |
| `ember-border` | `0.55 0.14 42 / 0.35` | Accent card border on focus |

Brackets and cursor in logo are always Ember — do not substitute other hues.

## Typography

```css
body {
  font-family: "Space Grotesk", system-ui, sans-serif;
  font-weight: 500;
  font-size: 14px;
  -webkit-font-smoothing: antialiased;
}
.mono, code, .label {
  font-family: "JetBrains Mono", monospace;
}
```

| Role | Font | Size / weight | Other |
|------|------|---------------|-------|
| Page title | Space Grotesk | 22–24px / 600 | letter-spacing -0.5px |
| Section title | JetBrains Mono | 11px / 500 | uppercase, letter-spacing 1.6px, `muted` |
| Card title | Space Grotesk | 16px / 600 | |
| Body | Space Grotesk | 14–15px / 500 | |
| Meta, timestamps | JetBrains Mono | 11–12px / 400 | tabular nums for data |
| Tabular data | JetBrains Mono or Space Grotesk | 13px | `font-variant-numeric: tabular-nums` |

**Wordmark in UI:** use `exports/lockup-dark.svg` or icon + “QuantFoundry” in Space Grotesk 600 — do not rebuild manually unless using brand SVGs.

## Spacing & layout

| Token | Value | Brand ref |
|-------|-------|-----------|
| `page-max-width` | 1200px | Brand `.page` |
| `page-padding` | 24px (app); 48px (marketing) | |
| `section-gap` | 32px | |
| `card-padding` | 20–24px | Brand cards 22–28px |
| `card-gap` | 16–18px | Brand grid gap 18px |

## Radius & borders

| Element | Value |
|---------|-------|
| Card | 10px (brand); 8px acceptable in dense app tables |
| Button | 4–6px |
| Status pill | 9999px |
| Input | 6px |
| Border | 1px `border-subtle` — no heavy shadows |

## Logo in app chrome

| Placement | Asset |
|-----------|-------|
| Sidebar expanded | `logo (1)/exports/lockup-dark.svg` or icon + wordmark-dark |
| Sidebar collapsed | `icon-dark.svg` or `favicon-32.png` |
| Browser tab | `favicon-32.png`, `favicon-192.png`, `apple-touch-icon.png` |

Minimum icon size: full QF from **24px**; below 24px use brackets + cursor only (brand rule).

## Components

### Primary button

```css
background: var(--ember);
color: var(--ink);
font-weight: 600;
border-radius: 6px;
```
Hover: `ember-deep`. One primary per card on dashboard.

### Secondary / ghost

Border `border-subtle`; text `text-primary`; hover `ember-muted` fill.

### Links

`color: var(--ember)`; hover underline or `ember-deep` (brand nav pattern).

### Section label

```css
font-family: "JetBrains Mono", monospace;
font-size: 11px;
letter-spacing: 1.6px;
text-transform: uppercase;
color: var(--muted);
```

### Sticky top bar (optional)

```css
background: oklch(0.18 0.01 270 / 0.85);
backdrop-filter: blur(10px);
border-bottom: 1px solid var(--border-subtle);
```

## Status pills

Always include text label; never color alone.

| State | Background / text |
|-------|-------------------|
| Running | `ember-muted` + `ember` text |
| Stopped | `ink-3` + `text-secondary` |
| Warning | `oklch(0.78 0.16 75)` tint (warm, on-brand) |
| Error | `oklch(0.58 0.20 25)` (deep red, not green/blue) |

## Zone timeline (Research)

Warm spectrum only — [brand.md](brand.md):

| Zone | OKLCH |
|------|-------|
| Train | `ember` |
| Validation | `ember-deep` |
| Portfolio zone (locked) | `oklch(0.50 0.12 55)` or muted amber band + lock icon |

## Accessibility

| Requirement | Implementation |
|-------------|----------------|
| Focus ring | 2px `ember` outline, 2px offset |
| Contrast | Paper on Ink / Ink on Ember ≥ WCAG AA |
| Motion | `prefers-reduced-motion` for expand/collapse |

## Light mode

Not MVP for app. Marketing site may use `paper-2` page bg per Brand.html. If app light mode ships later, invert `--bg-*` and `--text-*` aliases; keep Ember unchanged.

## Progressive disclosure

Collapsed by default; Ember for current step in research stepper; detail pages for heavy config.

## Related

- [brand.md](brand.md) — marks, exports, usage rules  
- [dashboard/mvp_ui.md](dashboard/mvp_ui.md) — minimal components  
- [`../product_vision.md`](../product_vision.md) — product tone
