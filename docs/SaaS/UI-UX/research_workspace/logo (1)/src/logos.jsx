// QuantFoundry — D direction (QF brackets) refined.
// The wordmark is "Quant" and "Foundry" framed by ember brackets,
// with an ember cursor bar between the two words.
// The ICON is the same construction compressed — "Q" and "F" framed
// by ember brackets, with an ember cursor bar between Q and F.
// They are the same logo at two zoom levels.

const INK = "oklch(0.18 0.01 270)";
const PAPER = "oklch(0.97 0.005 80)";
const EMBER = "oklch(0.66 0.17 45)";
const MUTED = "oklch(0.55 0.01 270)";

// ═════════════════════════════════════════════════════════
// THE WORDMARK — bracketed, ember cursor between words
// ═════════════════════════════════════════════════════════
function BracketedWordmark({ color = INK, accent = EMBER, size = 1 }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 * size, fontFamily: "Space Grotesk", fontWeight: 600, fontSize: 46 * size, letterSpacing: -1.1 * size, color }}>
      <span className="mono" style={{ fontWeight: 500, color: accent, fontSize: 50 * size }}>[</span>
      <span>Quant</span>
      <span style={{ width: 4 * size, height: 38 * size, background: accent, display: "inline-block" }} />
      <span style={{ fontWeight: 500, opacity: 0.85 }}>Foundry</span>
      <span className="mono" style={{ fontWeight: 500, color: accent, fontSize: 50 * size }}>]</span>
    </div>
  );
}

// ═════════════════════════════════════════════════════════
// THE ICON — D, refined. Six small variants below.
// Recommended primary: D2 (cursor bar between Q and F).
// All share viewBox 0 0 100 100 so they swap cleanly.
// ═════════════════════════════════════════════════════════

// D1 — Baseline: ember brackets, ink QF, no cursor.
function IconD1({ size = 1, color = INK, accent = EMBER }) {
  return (
    <svg viewBox="0 0 100 100" width={100 * size} height={100 * size}>
      <path d="M 18 16 L 8 16 L 8 84 L 18 84" fill="none" stroke={accent} strokeWidth="6" strokeLinecap="square" />
      <path d="M 82 16 L 92 16 L 92 84 L 82 84" fill="none" stroke={accent} strokeWidth="6" strokeLinecap="square" />
      <text x="50" y="66" textAnchor="middle" fontFamily="Space Grotesk" fontWeight="700" fontSize="44" letterSpacing="-2" fill={color}>QF</text>
    </svg>
  );
}

// D2 — Recommended: cursor bar between Q and F. Letters
// use the wordmark's exact font: Space Grotesk 600 (Q),
// Space Grotesk 500 @ 0.85 opacity (F). Anchored by the
// inner edges so the cursor sits dead centered.
function IconD2({ size = 1, color = INK, accent = EMBER }) {
  return (
    <svg viewBox="0 0 100 100" width={100 * size} height={100 * size}>
      <path d="M 18 16 L 8 16 L 8 84 L 18 84" fill="none" stroke={accent} strokeWidth="6" strokeLinecap="square" />
      <path d="M 82 16 L 92 16 L 92 84 L 82 84" fill="none" stroke={accent} strokeWidth="6" strokeLinecap="square" />
      <text x="45" y="66" textAnchor="end" fontFamily="Space Grotesk" fontWeight="600" fontSize="46" letterSpacing="-1.1" fill={color}>Q</text>
      <rect x="48" y="31" width="4" height="38" fill={accent} />
      <text x="55" y="66" textAnchor="start" fontFamily="Space Grotesk" fontWeight="500" fontSize="46" letterSpacing="-1.1" fill={color} opacity="0.85">F</text>
    </svg>
  );
}

// D3 — All ink: ink brackets, ink letters, single ember dot
// for accent. Quieter, more conservative.
function IconD3({ size = 1, color = INK, accent = EMBER }) {
  return (
    <svg viewBox="0 0 100 100" width={100 * size} height={100 * size}>
      <path d="M 18 16 L 8 16 L 8 84 L 18 84" fill="none" stroke={color} strokeWidth="6" strokeLinecap="square" />
      <path d="M 82 16 L 92 16 L 92 84 L 82 84" fill="none" stroke={color} strokeWidth="6" strokeLinecap="square" />
      <text x="45" y="66" textAnchor="end" fontFamily="Space Grotesk" fontWeight="600" fontSize="46" letterSpacing="-1.1" fill={color}>Q</text>
      <rect x="48" y="46" width="4" height="8" fill={accent} />
      <text x="55" y="66" textAnchor="start" fontFamily="Space Grotesk" fontWeight="500" fontSize="46" letterSpacing="-1.1" fill={color} opacity="0.85">F</text>
    </svg>
  );
}

// D4 — All ember: brackets and letters all in the accent.
// Boldest single-color treatment.
function IconD4({ size = 1, color = INK, accent = EMBER }) {
  return (
    <svg viewBox="0 0 100 100" width={100 * size} height={100 * size}>
      <path d="M 18 16 L 8 16 L 8 84 L 18 84" fill="none" stroke={accent} strokeWidth="6" strokeLinecap="square" />
      <path d="M 82 16 L 92 16 L 92 84 L 82 84" fill="none" stroke={accent} strokeWidth="6" strokeLinecap="square" />
      <text x="45" y="66" textAnchor="end" fontFamily="Space Grotesk" fontWeight="600" fontSize="46" letterSpacing="-1.1" fill={accent}>Q</text>
      <rect x="48" y="31" width="4" height="38" fill={accent} />
      <text x="55" y="66" textAnchor="start" fontFamily="Space Grotesk" fontWeight="500" fontSize="46" letterSpacing="-1.1" fill={accent} opacity="0.85">F</text>
    </svg>
  );
}

// D5 — Solid stamp: filled ink rounded square, paper letters
// reversed inside, ember cursor between. App-icon strong.
function IconD5({ size = 1, color = INK, accent = EMBER }) {
  return (
    <svg viewBox="0 0 100 100" width={100 * size} height={100 * size}>
      <rect x="0" y="0" width="100" height="100" rx="18" fill={color} />
      <path d="M 24 22 L 16 22 L 16 78 L 24 78" fill="none" stroke={PAPER} strokeWidth="5" strokeLinecap="square" />
      <path d="M 76 22 L 84 22 L 84 78 L 76 78" fill="none" stroke={PAPER} strokeWidth="5" strokeLinecap="square" />
      <text x="45" y="65" textAnchor="end" fontFamily="Space Grotesk" fontWeight="600" fontSize="38" letterSpacing="-0.9" fill={PAPER}>Q</text>
      <rect x="48" y="34" width="4" height="32" fill={accent} />
      <text x="55" y="65" textAnchor="start" fontFamily="Space Grotesk" fontWeight="500" fontSize="38" letterSpacing="-0.9" fill={PAPER} opacity="0.85">F</text>
    </svg>
  );
}

// D6 — Solid stamp, ember letters: filled ink rounded square
// with ember QF inside. Premium, glowing.
function IconD6({ size = 1, color = INK, accent = EMBER }) {
  return (
    <svg viewBox="0 0 100 100" width={100 * size} height={100 * size}>
      <rect x="0" y="0" width="100" height="100" rx="18" fill={color} />
      <path d="M 24 22 L 16 22 L 16 78 L 24 78" fill="none" stroke={accent} strokeWidth="5" strokeLinecap="square" />
      <path d="M 76 22 L 84 22 L 84 78 L 76 78" fill="none" stroke={accent} strokeWidth="5" strokeLinecap="square" />
      <text x="45" y="65" textAnchor="end" fontFamily="Space Grotesk" fontWeight="600" fontSize="38" letterSpacing="-0.9" fill={accent}>Q</text>
      <rect x="48" y="34" width="4" height="32" fill={PAPER} />
      <text x="55" y="65" textAnchor="start" fontFamily="Space Grotesk" fontWeight="500" fontSize="38" letterSpacing="-0.9" fill={accent} opacity="0.85">F</text>
    </svg>
  );
}

// The recommended primary, aliased for clarity throughout
const IconD = IconD2;

// ═════════════════════════════════════════════════════════
// PRESENTATION HELPERS
// ═════════════════════════════════════════════════════════

function PaperFrame({ children, label, size = 1, dark = false, extraLabel }) {
  const bg = dark ? INK : PAPER;
  const fg = dark ? PAPER : INK;
  return (
    <div style={{
      width: "100%", height: "100%",
      display: "flex", flexDirection: "column",
      background: bg, color: fg,
      position: "relative",
    }}>
      <div style={{ position: "absolute", top: 18, left: 22, right: 22, display: "flex", justifyContent: "space-between" }}>
        <span className="mono" style={{ fontSize: 11, letterSpacing: 1.6, opacity: 0.55 }}>{label}</span>
        <span className="mono" style={{ fontSize: 11, letterSpacing: 1.6, opacity: 0.35 }}>{extraLabel ?? (dark ? "DARK" : "LIGHT")}</span>
      </div>
      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: 40 }}>
        {React.cloneElement(children, { color: fg, size })}
      </div>
    </div>
  );
}

// Hero: full lockup icon + wordmark
function HeroLockup({ Icon = IconD2, dark = false }) {
  const bg = dark ? INK : PAPER;
  const fg = dark ? PAPER : INK;
  return (
    <div style={{ width: "100%", height: "100%", background: bg, color: fg, display: "flex", flexDirection: "column", padding: 36, boxSizing: "border-box" }}>
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <span className="mono" style={{ fontSize: 11, letterSpacing: 1.6, opacity: 0.55 }}>PRIMARY LOCKUP</span>
        <span className="mono" style={{ fontSize: 11, letterSpacing: 1.6, opacity: 0.35 }}>{dark ? "DARK" : "LIGHT"}</span>
      </div>
      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", gap: 36 }}>
        <Icon size={1.4} color={fg} />
        <div style={{ width: 1, height: 100, background: fg, opacity: 0.16 }} />
        <BracketedWordmark color={fg} />
      </div>
    </div>
  );
}

// Compress / expand visualization: icon → wordmark with an
// arrow making the relationship explicit.
function CompressExpand({ dark = false }) {
  const bg = dark ? INK : PAPER;
  const fg = dark ? PAPER : INK;
  return (
    <div style={{ width: "100%", height: "100%", background: bg, color: fg, display: "flex", flexDirection: "column", padding: 36, boxSizing: "border-box" }}>
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <span className="mono" style={{ fontSize: 11, letterSpacing: 1.6, opacity: 0.55 }}>COMPRESSED · EXPANDED</span>
        <span className="mono" style={{ fontSize: 11, letterSpacing: 1.6, opacity: 0.35 }}>{dark ? "DARK" : "LIGHT"}</span>
      </div>
      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", gap: 48 }}>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 14 }}>
          <IconD2 size={1.6} color={fg} />
          <span className="mono" style={{ fontSize: 11, letterSpacing: 1.6, opacity: 0.6 }}>[QF]</span>
        </div>
        <svg viewBox="0 0 80 24" width="80" height="24">
          <line x1="4" y1="12" x2="68" y2="12" stroke={fg} strokeWidth="2" opacity="0.5" />
          <polyline points="60,4 72,12 60,20" fill="none" stroke={fg} strokeWidth="2" opacity="0.5" />
        </svg>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 14 }}>
          <BracketedWordmark color={fg} />
          <span className="mono" style={{ fontSize: 11, letterSpacing: 1.6, opacity: 0.6 }}>[QuantFoundry]</span>
        </div>
      </div>
    </div>
  );
}

// D variants comparison row
function VariantsRow({ dark = false }) {
  const bg = dark ? INK : PAPER;
  const fg = dark ? PAPER : INK;
  const items = [
    ["D1", "All ink",         IconD1],
    ["D2", "Cursor (rec.)",   IconD2],
    ["D3", "Quiet accent",    IconD3],
    ["D4", "All ember",       IconD4],
    ["D5", "Stamp (paper)",   IconD5],
    ["D6", "Stamp (ember)",   IconD6],
  ];
  return (
    <div style={{ width: "100%", height: "100%", background: bg, color: fg, display: "flex", flexDirection: "column", padding: 32, boxSizing: "border-box" }}>
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <span className="mono" style={{ fontSize: 11, letterSpacing: 1.6, opacity: 0.55 }}>D · VARIANTS</span>
        <span className="mono" style={{ fontSize: 11, letterSpacing: 1.6, opacity: 0.35 }}>{dark ? "DARK" : "LIGHT"}</span>
      </div>
      <div style={{ flex: 1, display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 20, alignItems: "center", paddingTop: 18 }}>
        {items.map(([code, name, Icon]) => (
          <div key={code} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 14 }}>
            <Icon size={1.1} color={fg} />
            <div className="mono" style={{ fontSize: 11, letterSpacing: 1.4, opacity: 0.6 }}>{code}</div>
            <div style={{ fontFamily: "Space Grotesk", fontSize: 13, fontWeight: 500, opacity: 0.85 }}>{name}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Construction grid showing the geometry behind D2.
function Construction() {
  const fg = INK;
  const grid = MUTED;
  return (
    <div style={{ width: "100%", height: "100%", background: PAPER, color: fg, display: "flex", flexDirection: "column", padding: 32, boxSizing: "border-box" }}>
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <span className="mono" style={{ fontSize: 11, letterSpacing: 1.6, opacity: 0.55 }}>D2 · CONSTRUCTION</span>
        <span className="mono" style={{ fontSize: 11, letterSpacing: 1.6, opacity: 0.35 }}>100 × 100 · 8u GRID</span>
      </div>
      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", gap: 60 }}>
        <svg viewBox="-4 -4 108 108" width="320" height="320">
          {/* grid */}
          {Array.from({ length: 13 }, (_, i) => i * 8).map((v) => (
            <g key={v}>
              <line x1={v} y1="0" x2={v} y2="100" stroke={grid} strokeWidth="0.4" opacity="0.25" />
              <line x1="0" y1={v} x2="100" y2={v} stroke={grid} strokeWidth="0.4" opacity="0.25" />
            </g>
          ))}
          {/* safe area */}
          <rect x="0" y="0" width="100" height="100" fill="none" stroke={grid} strokeWidth="0.6" opacity="0.5" strokeDasharray="2 2" />
          {/* the icon */}
          <path d="M 18 16 L 8 16 L 8 84 L 18 84" fill="none" stroke={EMBER} strokeWidth="6" strokeLinecap="square" />
          <path d="M 82 16 L 92 16 L 92 84 L 82 84" fill="none" stroke={EMBER} strokeWidth="6" strokeLinecap="square" />
          <text x="45" y="66" textAnchor="end" fontFamily="Space Grotesk" fontWeight="600" fontSize="46" letterSpacing="-1.1" fill={INK}>Q</text>
          <rect x="48" y="31" width="4" height="38" fill={EMBER} />
          <text x="55" y="66" textAnchor="start" fontFamily="Space Grotesk" fontWeight="500" fontSize="46" letterSpacing="-1.1" fill={INK} opacity="0.85">F</text>
          {/* annotations */}
          <line x1="8" y1="92" x2="18" y2="92" stroke={MUTED} strokeWidth="0.6" />
          <text x="13" y="98" textAnchor="middle" fontFamily="JetBrains Mono" fontSize="3.6" fill={MUTED}>10</text>
          <line x1="82" y1="92" x2="92" y2="92" stroke={MUTED} strokeWidth="0.6" />
          <text x="87" y="98" textAnchor="middle" fontFamily="JetBrains Mono" fontSize="3.6" fill={MUTED}>10</text>
          <line x1="-2" y1="16" x2="-2" y2="84" stroke={MUTED} strokeWidth="0.6" />
          <text x="-2" y="52" textAnchor="middle" fontFamily="JetBrains Mono" fontSize="3.6" fill={MUTED} transform="rotate(-90 -2 52)">68</text>
        </svg>
        <div style={{ maxWidth: 320, fontFamily: "Space Grotesk", fontSize: 14, color: "oklch(0.32 0.01 270)", lineHeight: 1.55 }}>
          <div style={{ fontWeight: 600, marginBottom: 10, color: INK }}>Geometry</div>
          <ul style={{ paddingLeft: 18, margin: 0 }}>
            <li>100 × 100 viewBox, all measurements in 2-unit increments</li>
            <li>Brackets: 6u stroke, 10u inset from edge, 68u tall</li>
            <li>Letters: Space Grotesk 700, 44pt, –2 letter-spacing</li>
            <li>Cursor: 4 × 36u ember bar, optically centered</li>
            <li>16u of clear space around when used in a lockup</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────
// REAL-CONTEXT MOCKS
// ─────────────────────────────────────────────────────────

// macOS-style dock with our icon among others
function DockMock() {
  return (
    <div style={{ width: "100%", height: "100%", background: "linear-gradient(180deg, oklch(0.62 0.07 235), oklch(0.45 0.08 245))", display: "flex", alignItems: "flex-end", justifyContent: "center", paddingBottom: 32, boxSizing: "border-box" }}>
      <div style={{
        display: "flex", alignItems: "center", gap: 12,
        padding: "10px 14px",
        background: "oklch(0.96 0.005 80 / 0.42)",
        backdropFilter: "blur(20px)",
        WebkitBackdropFilter: "blur(20px)",
        borderRadius: 22,
        border: "1px solid oklch(1 0 0 / 0.4)",
        boxShadow: "0 12px 40px oklch(0 0 0 / 0.25)",
      }}>
        {/* placeholder app icons */}
        {["#22c55e", "#3b82f6", "#a855f7"].map((c, i) => (
          <div key={i} style={{ width: 56, height: 56, borderRadius: 14, background: c, opacity: 0.92 }} />
        ))}
        {/* our app icon (D5 stamp) */}
        <div style={{ width: 56, height: 56, borderRadius: 14, overflow: "hidden", boxShadow: "0 2px 6px rgba(0,0,0,0.2)" }}>
          <IconD5 size={0.56} />
        </div>
        {["#ef4444", "#f59e0b"].map((c, i) => (
          <div key={i} style={{ width: 56, height: 56, borderRadius: 14, background: c, opacity: 0.92 }} />
        ))}
      </div>
    </div>
  );
}

// Browser tab with our favicon
function BrowserTabMock() {
  return (
    <div style={{ width: "100%", height: "100%", background: "oklch(0.94 0.005 80)", padding: 24, boxSizing: "border-box", fontFamily: "Space Grotesk" }}>
      <div style={{ display: "flex", gap: 0 }}>
        {/* inactive tab */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 14px", background: "oklch(0.88 0.005 80)", borderRadius: "8px 8px 0 0", fontSize: 12, color: MUTED, maxWidth: 180, overflow: "hidden", whiteSpace: "nowrap" }}>
          <div style={{ width: 14, height: 14, background: "oklch(0.7 0.1 200)", borderRadius: 3 }} />
          Dashboard · Acme
        </div>
        {/* active tab — ours */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 14px", background: PAPER, borderRadius: "8px 8px 0 0", fontSize: 12, color: INK, fontWeight: 500, maxWidth: 240, marginLeft: 4 }}>
          <div style={{ width: 16, height: 16 }}>
            <IconD2 size={0.16} />
          </div>
          [QuantFoundry] · Strategies
        </div>
        {/* inactive tab */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 14px", background: "oklch(0.88 0.005 80)", borderRadius: "8px 8px 0 0", fontSize: 12, color: MUTED, maxWidth: 180, marginLeft: 4 }}>
          <div style={{ width: 14, height: 14, background: "oklch(0.7 0.1 30)", borderRadius: 3 }} />
          Inbox (12)
        </div>
      </div>
      <div style={{ background: PAPER, padding: "16px 18px", borderRadius: "0 8px 8px 8px", display: "flex", alignItems: "center", gap: 10, marginTop: -1 }}>
        <span className="mono" style={{ fontSize: 11, color: MUTED, padding: "4px 8px", background: "oklch(0.92 0.005 80)", borderRadius: 6 }}>app.quantfoundry.io</span>
        <span className="mono" style={{ fontSize: 11, color: MUTED }}>/strategies/momentum-v3</span>
      </div>
    </div>
  );
}

// Phone home screen with our app among others
function PhoneHomeMock() {
  return (
    <div style={{ width: "100%", height: "100%", background: "linear-gradient(160deg, oklch(0.32 0.06 280), oklch(0.22 0.05 270))", padding: 26, boxSizing: "border-box", display: "flex", flexDirection: "column", justifyContent: "center" }}>
      <div className="mono" style={{ fontSize: 10, color: "oklch(1 0 0 / 0.5)", letterSpacing: 1.6, marginBottom: 14, textAlign: "center" }}>HOME · 09:41</div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14, justifyItems: "center" }}>
        {/* row 1 */}
        {[
          { c: "#0ea5e9", label: "Mail" },
          { c: "#10b981", label: "Notes" },
          { c: "#a855f7", label: "Music" },
          { c: "#f59e0b", label: "Maps" },
        ].map((a, i) => (
          <div key={"r1-" + i} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
            <div style={{ width: 52, height: 52, borderRadius: 12, background: a.c }} />
            <span style={{ fontSize: 10, color: "white", fontFamily: "Space Grotesk" }}>{a.label}</span>
          </div>
        ))}
        {/* row 2 — ours + others */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
          <div style={{ width: 52, height: 52, borderRadius: 12, overflow: "hidden" }}>
            <IconD5 size={0.52} />
          </div>
          <span style={{ fontSize: 10, color: "white", fontFamily: "Space Grotesk", fontWeight: 600 }}>QuantFoundry</span>
        </div>
        {[
          { c: "#ef4444", label: "Photos" },
          { c: "#6366f1", label: "Calendar" },
          { c: "#22c55e", label: "Health" },
        ].map((a, i) => (
          <div key={"r2-" + i} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
            <div style={{ width: 52, height: 52, borderRadius: 12, background: a.c }} />
            <span style={{ fontSize: 10, color: "white", fontFamily: "Space Grotesk" }}>{a.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// Profile / social avatar usage
function AvatarMock() {
  return (
    <div style={{ width: "100%", height: "100%", background: PAPER, padding: 30, boxSizing: "border-box", display: "flex", flexDirection: "column", gap: 18, fontFamily: "Space Grotesk", color: INK }}>
      <span className="mono" style={{ fontSize: 11, letterSpacing: 1.6, opacity: 0.55 }}>AVATAR · TIMELINE</span>
      {/* row 1 — us */}
      <div style={{ display: "flex", gap: 14, alignItems: "flex-start" }}>
        <div style={{ width: 44, height: 44, borderRadius: "50%", overflow: "hidden", flexShrink: 0, background: INK, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <IconD6 size={0.36} />
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ display: "flex", gap: 6, alignItems: "baseline" }}>
            <span style={{ fontWeight: 600, fontSize: 14 }}>QuantFoundry</span>
            <span className="mono" style={{ fontSize: 11, color: MUTED }}>@quantfoundry · 2h</span>
          </div>
          <div style={{ fontSize: 14, marginTop: 4, color: "oklch(0.28 0.012 270)" }}>
            Momentum v3 just shipped. Sharpe 1.84 on out-of-sample. Open beta opens Monday.
          </div>
        </div>
      </div>
      {/* row 2 — someone else */}
      <div style={{ display: "flex", gap: 14, alignItems: "flex-start", opacity: 0.7 }}>
        <div style={{ width: 44, height: 44, borderRadius: "50%", flexShrink: 0, background: "oklch(0.72 0.08 30)" }} />
        <div style={{ flex: 1 }}>
          <div style={{ display: "flex", gap: 6, alignItems: "baseline" }}>
            <span style={{ fontWeight: 600, fontSize: 14 }}>Maya Chen</span>
            <span className="mono" style={{ fontSize: 11, color: MUTED }}>@maya · 1h</span>
          </div>
          <div style={{ fontSize: 14, marginTop: 4, color: "oklch(0.32 0.01 270)" }}>
            Replying with backtest results. The drawdown profile is wild.
          </div>
        </div>
      </div>
    </div>
  );
}

// Scale ladder: 16 → 256 on a single row
function ScaleLadder({ dark = false }) {
  const bg = dark ? INK : PAPER;
  const fg = dark ? PAPER : INK;
  const sizes = [16, 24, 40, 64, 128, 256];
  return (
    <div style={{ width: "100%", height: "100%", background: bg, color: fg, display: "flex", flexDirection: "column", padding: 32, boxSizing: "border-box" }}>
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <span className="mono" style={{ fontSize: 11, letterSpacing: 1.6, opacity: 0.55 }}>D2 · SCALE</span>
        <span className="mono" style={{ fontSize: 11, letterSpacing: 1.6, opacity: 0.35 }}>{dark ? "DARK" : "LIGHT"}</span>
      </div>
      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "space-around" }}>
        {sizes.map((s) => (
          <div key={s} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10 }}>
            <div style={{ width: s, height: s, display: "flex", alignItems: "center", justifyContent: "center" }}>
              <IconD2 size={s / 100} color={fg} />
            </div>
            <span className="mono" style={{ fontSize: 10, opacity: 0.55 }}>{s}px</span>
          </div>
        ))}
      </div>
    </div>
  );
}

Object.assign(window, {
  BracketedWordmark,
  IconD, IconD1, IconD2, IconD3, IconD4, IconD5, IconD6,
  PaperFrame, HeroLockup, CompressExpand, VariantsRow,
  Construction, DockMock, BrowserTabMock, PhoneHomeMock,
  AvatarMock, ScaleLadder,
  INK, PAPER, EMBER, MUTED,
});
