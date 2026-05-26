# QuantFoundry Brand (v1.0)

**Canonical source:** [`logo (1)/Brand.html`](logo%20(1)/Brand.html) · exports in [`logo (1)/exports/`](logo%20(1)/exports/)  
**App implementation:** [design_system.md](design_system.md)

Brand finalized May 2026. OKLCH-first; three tokens (**Ink**, **Paper**, **Ember**) plus support neutrals.

## Mark system

| Asset | Use |
|-------|-----|
| **Lockup** (`lockup-dark.svg`) | Sidebar, marketing hero on dark |
| **Icon** (`icon-dark-512.png` / `stamp-ink`) | Favicon, compact chrome |
| **Wordmark** (`wordmark-dark.svg`) | Auth pages, email headers |

Construction: `[Quant|Foundry]` — Ember brackets `[` `]`, Ember cursor bar between words, Space Grotesk letters.

**Do not:** recolor brackets/cursor away from Ember; stretch or rotate the icon; use blue/purple UI accents.

## Color tokens (CSS)

```css
:root {
  --ink:        oklch(0.18 0.01 270);   /* #22232B */
  --ink-2:      oklch(0.28 0.012 270);
  --ink-3:      oklch(0.38 0.012 270);
  --paper:      oklch(0.97 0.005 80);   /* #F7F5EF */
  --paper-2:    oklch(0.94 0.006 80);
  --line:       oklch(0.86 0.008 80);
  --line-2:     oklch(0.80 0.008 80);
  --muted:      oklch(0.55 0.01 270);
  --ember:      oklch(0.66 0.17 45);    /* #D57044 */
  --ember-deep: oklch(0.54 0.16 38);
}
```

Support ramp (body copy, dividers): `ink` → `ink-2` → `ink-3` → `muted` → `oklch(0.78 0.01 270)` → `oklch(0.92 0.006 80)` → `paper`.

## Typography

| Family | Role | Weights |
|--------|------|---------|
| **Space Grotesk** | UI, headings, wordmark | 400 / 500 / 600 / 700 |
| **JetBrains Mono** | Labels, code, metadata, brackets in UI copy | 400 / 500 |

Google Fonts:

```html
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet" />
```

| Style | Spec |
|-------|------|
| Section label | JetBrains Mono 11px, uppercase, letter-spacing 1.6px, `muted` |
| Body | Space Grotesk 500, 14–15px |
| Page title | Space Grotesk 600, 22–24px, letter-spacing -0.5px |
| Wordmark (marketing) | Space Grotesk 600, letter-spacing -2.4% |

## Favicons (web app)

From brand bundle — deploy to app `public/`:

```html
<link rel="icon" type="image/png" sizes="32x32" href="/favicon-32.png" />
<link rel="icon" type="image/png" sizes="192x192" href="/favicon-192.png" />
<link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png" />
```

Source files: `logo (1)/exports/favicon-*.png`, `apple-touch-icon.png`.

## UI chrome (from brand patterns)

| Element | Spec |
|---------|------|
| Card radius | 10–12px |
| Button radius | 4–6px |
| Card border | 1px `line` (light) / `ink-3` on dark surfaces |
| Primary button | `ember` background, `ink` text |
| Link | `ember`; hover: underline or `ember-deep` |
| Sticky header | `paper-2` @ 85% + blur (light docs); app uses `ink` @ 85% + blur |

## App theme

QuantFoundry-Web uses the **dark app shell** (Ink surfaces, Paper text) — matches `lockup-dark`, `icon-dark`, `wordmark-dark`. Light Paper theme is reserved for marketing/docs (Brand.html), not the trading UI.

## File index

See Brand.html § Export bundle — 36 SVG/PNG files under `logo (1)/exports/`.
