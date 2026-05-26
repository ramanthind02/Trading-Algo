const { DesignCanvas, DCSection, DCArtboard } = window;

function App() {
  return (
    <DesignCanvas
      title="QuantFoundry — [QF] icon + wordmark"
      subtitle="The icon is the wordmark compressed. The wordmark is the icon expanded. Same construction, same vocabulary."
    >
      <DCSection id="hero" title="Primary lockup">
        <DCArtboard id="hero-light" label="Light · primary" width={1180} height={320}>
          <HeroLockup />
        </DCArtboard>
        <DCArtboard id="hero-dark" label="Dark · reversed" width={1180} height={320}>
          <HeroLockup dark />
        </DCArtboard>
      </DCSection>

      <DCSection id="concept" title="The compress / expand idea">
        <DCArtboard id="concept-light" label="Concept · light" width={1180} height={300}>
          <CompressExpand />
        </DCArtboard>
        <DCArtboard id="concept-dark" label="Concept · dark" width={1180} height={300}>
          <CompressExpand dark />
        </DCArtboard>
      </DCSection>

      <DCSection id="variants" title="D · variants — pick the cursor treatment">
        <DCArtboard id="variants-light" label="Variants · light" width={1180} height={300}>
          <VariantsRow />
        </DCArtboard>
        <DCArtboard id="variants-dark" label="Variants · dark" width={1180} height={300}>
          <VariantsRow dark />
        </DCArtboard>
      </DCSection>

      <DCSection id="construction" title="Construction & geometry">
        <DCArtboard id="construction" label="D2 — 100 × 100, 8u grid" width={1180} height={420}>
          <Construction />
        </DCArtboard>
      </DCSection>

      <DCSection id="in-the-wild" title="In the wild — real contexts">
        <DCArtboard id="ctx-dock" label="macOS dock" width={760} height={300}>
          <DockMock />
        </DCArtboard>
        <DCArtboard id="ctx-tab" label="Browser tab + favicon" width={760} height={300}>
          <BrowserTabMock />
        </DCArtboard>
        <DCArtboard id="ctx-phone" label="Phone home screen" width={520} height={420}>
          <PhoneHomeMock />
        </DCArtboard>
        <DCArtboard id="ctx-avatar" label="Social / timeline avatar" width={760} height={320}>
          <AvatarMock />
        </DCArtboard>
      </DCSection>

      <DCSection id="scale" title="Scale test — 16 → 256 px">
        <DCArtboard id="scale-light" label="Scale · light" width={1180} height={220}>
          <ScaleLadder />
        </DCArtboard>
        <DCArtboard id="scale-dark" label="Scale · dark" width={1180} height={220}>
          <ScaleLadder dark />
        </DCArtboard>
      </DCSection>

      <DCSection id="rationale" title="Notes">
        <DCArtboard id="notes" label="Design notes" width={1180} height={380}>
          <div style={{ width: "100%", height: "100%", background: PAPER, padding: "44px 56px", boxSizing: "border-box", fontFamily: "Space Grotesk", color: INK, lineHeight: 1.55 }}>
            <div className="mono" style={{ fontSize: 11, letterSpacing: 1.8, color: MUTED, marginBottom: 18 }}>SYSTEM</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 40 }}>
              <div>
                <div style={{ fontWeight: 600, marginBottom: 8 }}>The system in one sentence</div>
                <div style={{ fontSize: 15, color: "oklch(0.32 0.01 270)" }}>The brand is two words separated by an ember cursor, wrapped in ember brackets. The icon is the abbreviation of those two words, separated by the same cursor, wrapped in the same brackets. One construction, two zoom levels.</div>
              </div>
              <div>
                <div style={{ fontWeight: 600, marginBottom: 8 }}>Why it works</div>
                <div style={{ fontSize: 15, color: "oklch(0.32 0.01 270)" }}>A separate icon and wordmark usually require the viewer to learn two things. Here, they learn one — the bracketed pattern — and apply it at whichever scale they see. The icon doesn't decorate the wordmark; it <em>is</em> the wordmark, compressed.</div>
              </div>
              <div>
                <div style={{ fontWeight: 600, marginBottom: 8 }}>Palette</div>
                <div style={{ display: "flex", gap: 10, marginBottom: 10 }}>
                  <div style={{ width: 36, height: 36, background: INK, borderRadius: 4 }} />
                  <div style={{ width: 36, height: 36, background: PAPER, borderRadius: 4, border: "1px solid oklch(0.85 0.008 80)" }} />
                  <div style={{ width: 36, height: 36, background: EMBER, borderRadius: 4 }} />
                </div>
                <div className="mono" style={{ fontSize: 12, color: MUTED }}>
                  ink &nbsp; oklch(.18 .01 270)<br/>
                  paper oklch(.97 .005 80)<br/>
                  ember oklch(.66 .17 45)
                </div>
              </div>
            </div>
          </div>
        </DCArtboard>
      </DCSection>
    </DesignCanvas>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
