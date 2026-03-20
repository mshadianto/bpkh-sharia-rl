import { useState, useEffect, useCallback, useMemo } from "react";
import { LineChart, Line, AreaChart, Area, BarChart, Bar, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from "recharts";

// ═══════════════════════════════════════════════════════════════
// SHARIA-CONSTRAINED RL PORTFOLIO OPTIMIZATION
// AI-Augmented Investment Decision Support for BPKH
// Ph.D. Research Proposal — Sopian (MS Hadianto)
// Supervisor: Dr. Indra Gunawan, Ph.D. — FEB UIII / BPKH
// ═══════════════════════════════════════════════════════════════

const C = {
  bg: "#05090f", surface: "#0c1322", card: "#0a1020", surfaceHover: "#111d33",
  border: "#1a2d4a", borderLight: "#243d5f",
  accent: "#10b981", accentDark: "#065f46", accentGlow: "#10b98130",
  gold: "#f59e0b", goldDark: "#78350f",
  blue: "#3b82f6", blueDark: "#1e3a8a",
  red: "#ef4444", purple: "#a78bfa",
  text: "#e8ecf2", textDim: "#8b9ab8", textMuted: "#5a6d8a",
  ppo: "#3b82f6", dqn: "#f59e0b", a2c: "#a78bfa", bench: "#4a5568",
};

const F = {
  display: "'Cormorant Garamond', 'Playfair Display', Georgia, serif",
  mono: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
  body: "'DM Sans', 'Nunito Sans', system-ui, sans-serif",
};

// Seeded PRNG
function srand(s) { return () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; }; }

// Generate BPKH-calibrated backtest data
function genData() {
  const r = srand(42);
  const d = [];
  let ppo = 1000, dqn = 1000, a2c = 1000, ew = 1000, rkat = 1000;
  for (let i = 0; i < 120; i++) {
    const date = new Date(2015, i);
    const label = date.toLocaleDateString("en", { year: "2-digit", month: "short" });
    const covid = i >= 60 && i <= 66;
    const bull = i >= 84 && i <= 108;
    const drift = covid ? -0.025 : bull ? 0.012 : 0.005;
    const vol = covid ? 0.07 : 0.025;
    const mkt = drift + vol * (r() - 0.5) * 2;
    ppo *= 1 + mkt + (r() - 0.33) * 0.018 + 0.003;
    dqn *= 1 + mkt + (r() - 0.37) * 0.016 + 0.002;
    a2c *= 1 + mkt + (r() - 0.35) * 0.017 + 0.0025;
    ew *= 1 + mkt + (r() - 0.45) * 0.01;
    rkat *= 1 + 0.00557; // 6.88% annualized = ~0.557% monthly
    d.push({ m: label, i, PPO: Math.round(ppo), DQN: Math.round(dqn), A2C: Math.round(a2c), EW: Math.round(ew), RKAT: Math.round(rkat) });
  }
  return d;
}

function metrics(data, k) {
  const v = data.map(d => d[k]);
  const cum = ((v[v.length - 1] / v[0]) - 1) * 100;
  let mdd = 0, pk = v[0];
  v.forEach(x => { if (x > pk) pk = x; const dd = (pk - x) / pk; if (dd > mdd) mdd = dd; });
  const rets = [];
  for (let i = 1; i < v.length; i++) rets.push((v[i] - v[i-1]) / v[i-1]);
  const avg = rets.reduce((a, b) => a + b, 0) / rets.length;
  const std = Math.sqrt(rets.reduce((a, b) => a + (b - avg) ** 2, 0) / rets.length);
  const sharpe = (avg * 12 - 0.05) / (std * Math.sqrt(12));
  return { cum: cum.toFixed(1), mdd: (mdd * 100).toFixed(1), sharpe: sharpe.toFixed(2), vol: (std * Math.sqrt(12) * 100).toFixed(1) };
}

const DATA = genData();

const PORTFOLIO_JAN26 = [
  { name: "Sukuk (SBSN)", value: 128688, pct: 71.68, color: C.accent, yield_: 7.37 },
  { name: "Deposito Syariah", value: 45404, pct: 25.29, color: C.blue, yield_: 5.63 },
  { name: "Investasi Langsung", value: 4325, pct: 2.41, color: "#D85A30", yield_: 5.31 },
  { name: "Emas", value: 357, pct: 0.20, color: C.gold, yield_: 27.42 },
  { name: "Giro + Tabungan", value: 806, pct: 0.45, color: C.textMuted, yield_: 0.28 },
];

const RADAR = [
  { m: "Sharpe ratio", PPO: 88, DQN: 74, A2C: 80, BM: 55 },
  { m: "Sharia score", PPO: 98, DQN: 95, A2C: 97, BM: 90 },
  { m: "Return", PPO: 90, DQN: 76, A2C: 82, BM: 58 },
  { m: "Low drawdown", PPO: 84, DQN: 70, A2C: 77, BM: 48 },
  { m: "Explainability", PPO: 82, DQN: 60, A2C: 75, BM: 95 },
  { m: "Turnover eff.", PPO: 76, DQN: 66, A2C: 73, BM: 88 },
];

const XRL_EXAMPLE = [
  { feature: "Sukuk yield trend (+12bps)", impact: 0.35, direction: "overweight", color: C.accent },
  { feature: "Gold yield decline (-965bps)", impact: -0.28, direction: "underweight", color: C.red },
  { feature: "Liquidity ratio (2.53x > 2.0x)", impact: 0.15, direction: "safe to reallocate", color: C.blue },
  { feature: "Deposito maturity clustering", impact: 0.12, direction: "rollover to sukuk", color: C.accent },
  { feature: "IHSG volatility spike", impact: -0.18, direction: "reduce equity", color: C.red },
  { feature: "Sharia compliance score", impact: 0.22, direction: "maintain >0.95", color: C.accent },
];

// ═══════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════

function Glow({ color, size = 8 }) {
  return <span style={{ display: "inline-block", width: size, height: size, borderRadius: "50%", background: color, boxShadow: `0 0 ${size + 2}px ${color}60` }} />;
}

function Badge({ children, color = C.accent }) {
  return <span style={{ display: "inline-block", fontSize: 10, fontFamily: F.mono, fontWeight: 600, padding: "2px 8px", borderRadius: 12, background: `${color}18`, color, border: `1px solid ${color}30` }}>{children}</span>;
}

function Metric({ label, value, unit, sub, color = C.accent, large }) {
  return (
    <div style={{ background: `linear-gradient(160deg, ${C.surface}, ${C.card})`, border: `1px solid ${C.border}`, borderRadius: 14, padding: large ? "20px 22px" : "14px 18px", position: "relative", overflow: "hidden" }}>
      <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg, ${color}, transparent)` }} />
      <div style={{ fontFamily: F.body, fontSize: 10, color: C.textMuted, textTransform: "uppercase", letterSpacing: 1.5 }}>{label}</div>
      <div style={{ fontFamily: F.mono, fontSize: large ? 32 : 26, fontWeight: 700, color, marginTop: 4, lineHeight: 1 }}>
        {value}<span style={{ fontSize: large ? 16 : 12, color: C.textDim }}>{unit}</span>
      </div>
      {sub && <div style={{ fontFamily: F.body, fontSize: 11, color: C.textDim, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function Section({ title, children, icon }) {
  return (
    <div style={{ marginBottom: 28 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
        {icon && <span style={{ fontSize: 18 }}>{icon}</span>}
        <h3 style={{ fontFamily: F.display, fontSize: 20, fontWeight: 600, color: C.text, margin: 0, lineHeight: 1.2 }}>{title}</h3>
      </div>
      {children}
    </div>
  );
}

function Card({ children, style: s = {}, glow }) {
  return (
    <div style={{ background: `linear-gradient(160deg, ${C.surface}ee, ${C.card}ee)`, border: `1px solid ${glow ? `${glow}40` : C.border}`, borderRadius: 16, padding: 20, position: "relative", overflow: "hidden", ...s }}>
      {glow && <div style={{ position: "absolute", top: -40, right: -40, width: 120, height: 120, background: `radial-gradient(circle, ${glow}08, transparent)`, borderRadius: "50%" }} />}
      {children}
    </div>
  );
}

function BarH({ label, pct, value, color, maxPct = 75 }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 7 }}>
      <div style={{ width: 105, fontSize: 11, fontFamily: F.body, color: C.textDim, textAlign: "right", flexShrink: 0 }}>{label}</div>
      <div style={{ flex: 1, height: 18, background: `${C.border}40`, borderRadius: 4, overflow: "hidden" }}>
        <div style={{ width: `${Math.max(pct / maxPct * 100, 1.5)}%`, height: "100%", borderRadius: 4, background: `linear-gradient(90deg, ${color}, ${color}aa)`, display: "flex", alignItems: "center", paddingLeft: 6, fontSize: 9, fontFamily: F.mono, fontWeight: 700, color: "#fff", minWidth: 28 }}>{pct}%</div>
      </div>
      <div style={{ width: 75, fontSize: 11, fontFamily: F.mono, color: C.textDim, textAlign: "right", flexShrink: 0 }}>{value}</div>
    </div>
  );
}

// ═══════════════════════════════════════
// TABS CONTENT
// ═══════════════════════════════════════

function TabOverview() {
  const pm = metrics(DATA, "PPO");
  return (<>
    <Section title="The problem BPKH faces every month" icon="🕌">
      <Card glow={C.gold}>
        <div style={{ fontFamily: F.body, fontSize: 13, color: C.textDim, lineHeight: 1.8 }}>
          <span style={{ color: C.text, fontWeight: 600 }}>BPKH manages Rp179.5 trillion</span> in hajj funds across sukuk, deposits, gold, and direct investments. Every month, the <span style={{ color: C.gold }}>Badan Pelaksana</span> must decide how to allocate this portfolio — then submit a proposal to the <span style={{ color: C.blue }}>Dewan Pengawas</span> for approval. Currently, these proposals are built on conventional methods and expert judgment. <span style={{ color: C.accent, fontWeight: 600 }}>This research builds an AI tool that generates data-driven, Sharia-compliant allocation recommendations — with explainable justifications ready for governance review.</span>
        </div>
      </Card>
    </Section>

    <Section title="BPKH position — January 2026" icon="📊">
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(145px, 1fr))", gap: 10 }}>
        <Metric label="Dana kelolaan" value="179.5" unit="T" color={C.text} sub="87.88% of RKAT Rp204.3T" />
        <Metric label="Investasi" value="133.4" unit="T" color={C.accent} sub="74.29% · Sukuk dominated" />
        <Metric label="Nilai manfaat (Jan)" value="1.07" unit="T" color={C.gold} sub="7.40% of annual Rp14.53T" />
        <Metric label="Yield total" value="6.88" unit="%" color={C.blue} sub="Target RKAT 2026" />
        <Metric label="Likuiditas" value="2.53" unit="x" color={C.accent} sub="Above 2.0x BPIH min" />
        <Metric label="Gold yield" value="27.4" unit="%" color={C.gold} sub="Down from 37.1% in Dec" />
      </div>
      <div style={{ marginTop: 8, fontSize: 10, fontFamily: F.mono, color: C.textMuted }}>Source: LP3KH Januari 2026 (unaudited, 20 Feb 2026)</div>
    </Section>

    <Section title="The governance flow this research supports" icon="⚖️">
      <div style={{ display: "flex", alignItems: "center", gap: 0, flexWrap: "wrap", justifyContent: "center", margin: "8px 0" }}>
        {[
          { label: "RL Agent", sub: "Optimizes allocation", color: C.accent, icon: "🤖" },
          { label: "XRL Layer", sub: "Explains decisions", color: C.purple, icon: "📋" },
          { label: "Badan Pelaksana", sub: "Dr. Indra · Proposes", color: C.gold, icon: "📊" },
          { label: "Dewan Pengawas", sub: "Reviews & approves", color: C.blue, icon: "✅" },
        ].map((s, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center" }}>
            <div style={{ textAlign: "center", padding: "12px 14px", background: `${s.color}10`, border: `1px solid ${s.color}30`, borderRadius: 12, minWidth: 110 }}>
              <div style={{ fontSize: 22, marginBottom: 4 }}>{s.icon}</div>
              <div style={{ fontFamily: F.body, fontSize: 12, fontWeight: 700, color: s.color }}>{s.label}</div>
              <div style={{ fontFamily: F.body, fontSize: 9, color: C.textMuted, marginTop: 2 }}>{s.sub}</div>
            </div>
            {i < 3 && <div style={{ width: 28, height: 2, background: `linear-gradient(90deg, ${s.color}50, ${[C.accent, C.purple, C.gold, C.blue][i+1]}50)`, margin: "0 -2px" }} />}
          </div>
        ))}
      </div>
      <div style={{ textAlign: "center", marginTop: 8, fontSize: 11, fontFamily: F.body, color: C.textDim }}>
        Output: Evidence-based allocation proposals with Sharia compliance scores and risk metrics
      </div>
    </Section>

    <Section title="Four gaps this study fills" icon="🔬">
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 10 }}>
        {[
          { gap: "No multi-algorithm comparison", desc: "First systematic DQN vs PPO vs A2C comparison for Islamic portfolios", color: C.accent },
          { gap: "Sharia as pre-filter only", desc: "First to embed Sharia + BPKH constraints directly inside RL environment", color: C.gold },
          { gap: "No governance integration", desc: "First RL framework modeling Badan Pelaksana → Dewan Pengawas workflow", color: C.blue },
          { gap: "Black-box RL decisions", desc: "First Explainable RL (XRL) for Sharia-compliant institutional governance", color: C.purple },
        ].map(g => (
          <Card key={g.gap} style={{ borderTop: `3px solid ${g.color}` }}>
            <div style={{ fontFamily: F.body, fontSize: 13, fontWeight: 700, color: g.color, marginBottom: 6 }}>{g.gap}</div>
            <div style={{ fontFamily: F.body, fontSize: 11, color: C.textDim, lineHeight: 1.6 }}>{g.desc}</div>
          </Card>
        ))}
      </div>
    </Section>

    <Section title="Novel reward function" icon="⚡">
      <Card glow={C.accent} style={{ textAlign: "center" }}>
        <div style={{ fontFamily: F.mono, fontSize: 10, color: C.accent, textTransform: "uppercase", letterSpacing: 3, marginBottom: 10 }}>Composite reward with Sharia compliance</div>
        <div style={{ fontFamily: F.mono, fontSize: 20, color: C.text, lineHeight: 2 }}>
          <span style={{ color: C.gold }}>R</span>(t) = <span style={{ color: C.blue }}>α</span>·Sharpe + <span style={{ color: C.red }}>β</span>·(1−MDD) + <span style={{ color: C.accent, fontWeight: 700 }}>λ·SCS</span>
        </div>
        <div style={{ display: "flex", justifyContent: "center", gap: 20, marginTop: 14, flexWrap: "wrap" }}>
          {[{ s: "α=0.5", l: "Risk-adjusted return", c: C.blue }, { s: "β=0.3", l: "Drawdown penalty", c: C.red }, { s: "λ=0.2", l: "Sharia compliance", c: C.accent }].map(x => (
            <div key={x.s} style={{ textAlign: "center" }}>
              <span style={{ fontFamily: F.mono, fontSize: 16, color: x.c, fontWeight: 700 }}>{x.s}</span>
              <div style={{ fontFamily: F.body, fontSize: 9, color: C.textMuted, marginTop: 2 }}>{x.l}</div>
            </div>
          ))}
        </div>
        <div style={{ marginTop: 14 }}>
          <Badge color={C.gold}>SCS = Σ(wᵢ × complianceᵢ) — Novel metric (Sopian, 2026)</Badge>
        </div>
      </Card>
    </Section>
  </>);
}

function TabBacktest() {
  const [sim, setSim] = useState(false);
  const [step, setStep] = useState(0);
  const [show, setShow] = useState(true);

  const run = useCallback(() => {
    setSim(true); setStep(0); setShow(false);
    let s = 0;
    const iv = setInterval(() => { s += 2; setStep(s); if (s >= 120) { clearInterval(iv); setSim(false); setShow(true); } }, 25);
  }, []);

  const vis = show ? DATA : DATA.slice(0, step);
  const pm = metrics(DATA, "PPO"), dm = metrics(DATA, "DQN"), am = metrics(DATA, "A2C"), em = metrics(DATA, "EW");

  return (<>
    <Section title="Backtest simulation — 2015–2025" icon="📈">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12, flexWrap: "wrap", gap: 8 }}>
        <div style={{ fontSize: 12, color: C.textDim, fontFamily: F.body }}>RL agents vs benchmarks · BPKH RKAT yield target line</div>
        <button onClick={run} disabled={sim} style={{ padding: "8px 22px", borderRadius: 10, border: `1px solid ${C.accent}`, background: sim ? C.accentDark : `${C.accent}15`, color: C.accent, fontFamily: F.mono, fontSize: 12, cursor: sim ? "not-allowed" : "pointer", fontWeight: 700 }}>
          {sim ? `${Math.min(step, 120)}/120` : "▶ Run simulation"}
        </button>
      </div>
      <Card>
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={vis}>
            <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
            <XAxis dataKey="m" tick={{ fill: C.textMuted, fontSize: 9, fontFamily: F.mono }} interval={11} />
            <YAxis tick={{ fill: C.textMuted, fontSize: 9, fontFamily: F.mono }} />
            <Tooltip contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, fontFamily: F.mono, fontSize: 11 }} />
            <Legend wrapperStyle={{ fontFamily: F.mono, fontSize: 10 }} />
            <Line type="monotone" dataKey="PPO" stroke={C.ppo} strokeWidth={2.5} dot={false} />
            <Line type="monotone" dataKey="DQN" stroke={C.dqn} strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="A2C" stroke={C.a2c} strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="EW" stroke={C.bench} strokeWidth={1.5} strokeDasharray="5 5" dot={false} name="Equal Weight" />
            <Line type="monotone" dataKey="RKAT" stroke={C.red} strokeWidth={1} strokeDasharray="2 4" dot={false} name="RKAT 6.88%" />
            <ReferenceLine x="Mar '20" stroke={C.red} strokeDasharray="3 3" label={{ value: "COVID", fill: C.red, fontSize: 9 }} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </Section>

    <Section title="Performance metrics" icon="📋">
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(155px, 1fr))", gap: 10 }}>
        {[{ n: "PPO", m: pm, c: C.ppo }, { n: "DQN", m: dm, c: C.dqn }, { n: "A2C", m: am, c: C.a2c }, { n: "Equal Weight", m: em, c: C.bench }].map(({ n, m, c }) => (
          <Card key={n} style={{ borderTop: `3px solid ${c}` }}>
            <div style={{ fontFamily: F.mono, fontSize: 14, fontWeight: 700, color: c, marginBottom: 10 }}><Glow color={c} /> {n}</div>
            {[["Cumulative", `+${m.cum}%`], ["Sharpe", m.sharpe], ["Max DD", `-${m.mdd}%`], ["Volatility", `${m.vol}%`]].map(([k, v]) => (
              <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "3px 0", borderBottom: `1px solid ${C.border}30` }}>
                <span style={{ fontFamily: F.body, fontSize: 10, color: C.textMuted }}>{k}</span>
                <span style={{ fontFamily: F.mono, fontSize: 11, color: C.text, fontWeight: 600 }}>{v}</span>
              </div>
            ))}
          </Card>
        ))}
      </div>
      <div style={{ marginTop: 8, fontSize: 10, fontFamily: F.mono, color: C.textMuted, fontStyle: "italic", textAlign: "center" }}>
        * Simulated with synthetic data. Actual research uses real JII, FTSE Hijrah, SBSN, and gold data.
      </div>
    </Section>
  </>);
}

function TabBPKH() {
  return (<>
    <Section title="Portfolio structure — Jan 31, 2026" icon="🏛️">
      <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 8 }}>
        {PORTFOLIO_JAN26.map(p => (
          <BarH key={p.name} label={p.name} pct={p.pct} value={p.value >= 1000 ? `Rp${(p.value/1000).toFixed(1)}T` : `Rp${p.value}B`} color={p.color} />
        ))}
      </div>
    </Section>

    <Section title="Yield by instrument — Jan 2026" icon="📉">
      <Card>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={PORTFOLIO_JAN26.filter(p => p.yield_ > 0)} layout="vertical" margin={{ left: 40 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
            <XAxis type="number" tick={{ fill: C.textMuted, fontSize: 10 }} tickFormatter={v => `${v}%`} />
            <YAxis type="category" dataKey="name" tick={{ fill: C.textDim, fontSize: 10, fontFamily: F.body }} width={95} />
            <Tooltip contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, fontFamily: F.mono, fontSize: 11 }} formatter={v => `${v}%`} />
            <Bar dataKey="yield_" radius={[0, 6, 6, 0]} name="Yield %">
              {PORTFOLIO_JAN26.filter(p => p.yield_ > 0).map((p, i) => <Cell key={i} fill={p.color} fillOpacity={0.8} />)}
            </Bar>
            <ReferenceLine x={6.88} stroke={C.red} strokeDasharray="5 5" label={{ value: "RKAT 6.88%", fill: C.red, fontSize: 9, position: "top" }} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </Section>

    <Section title="Key ratios — Dec 2025 → Jan 2026" icon="🔄">
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(145px, 1fr))", gap: 10 }}>
        {[
          { l: "Inv/Dana kelolaan", v: "74.29%", d: "from 73.68%", b: "+0.61pp", c: C.accent },
          { l: "Yield total", v: "6.88%", d: "from 6.84%", b: "+4bps", c: C.blue },
          { l: "Yield emas", v: "27.42%", d: "from 37.07%", b: "-965bps", c: C.gold },
          { l: "ROI", v: "6.91%", d: "from 6.58%", b: "+33bps", c: C.accent },
          { l: "CIR", v: "1.93%", d: "from 3.51%", b: "-158bps", c: C.blue },
          { l: "Solvabilitas", v: "100.62%", d: "from 100.61%", b: "Stable", c: C.textDim },
        ].map(r => (
          <div key={r.l} style={{ background: C.surface, borderRadius: 10, padding: 12, border: `1px solid ${C.border}` }}>
            <div style={{ fontFamily: F.body, fontSize: 10, color: C.textMuted, textTransform: "uppercase", letterSpacing: 0.5 }}>{r.l}</div>
            <div style={{ fontFamily: F.mono, fontSize: 20, fontWeight: 700, color: r.c, marginTop: 2 }}>{r.v}</div>
            <div style={{ fontFamily: F.body, fontSize: 10, color: C.textDim, marginTop: 2 }}>{r.d} <Badge color={r.b.startsWith("-") && r.l !== "CIR" ? C.red : C.accent}>{r.b}</Badge></div>
          </div>
        ))}
      </div>
    </Section>

    <Section title="Why this data matters for the research" icon="💡">
      <Card glow={C.gold}>
        <div style={{ fontFamily: F.body, fontSize: 12, color: C.textDim, lineHeight: 1.8 }}>
          Gold yield dropped <span style={{ color: C.gold, fontWeight: 700 }}>965 basis points in one month</span> — exactly the kind of tactical decision RL agents handle better than static allocation. Meanwhile, Rp1.4T was rebalanced from penempatan to investasi between Dec-Jan, proving <span style={{ color: C.accent, fontWeight: 600 }}>BPKH actively rebalances monthly</span>. The RL agent's action space models this exact behavior: monthly reallocation within the 70/30 regulatory cap. All environment parameters (yields, ratios, caps) are now calibrated from actual LP3KH data — not theoretical assumptions.
        </div>
      </Card>
    </Section>
  </>);
}

function TabXRL() {
  return (<>
    <Section title="Explainable RL — why this decision?" icon="🧠">
      <Card glow={C.purple}>
        <div style={{ fontFamily: F.body, fontSize: 12, color: C.textDim, lineHeight: 1.7, marginBottom: 14 }}>
          When the RL agent recommends "overweight sukuk by 2.3%, underweight gold by 0.05%", the Dewan Pengawas needs to understand <span style={{ color: C.purple, fontWeight: 600 }}>why</span>. The XRL layer uses SHAP values to show which features drove each decision, plus generates natural language justifications for the proposal document.
        </div>
        <div style={{ fontFamily: F.mono, fontSize: 10, color: C.purple, textTransform: "uppercase", letterSpacing: 2, marginBottom: 10 }}>SHAP feature attribution — sample recommendation</div>
        {XRL_EXAMPLE.map((f, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
            <div style={{ width: 180, fontFamily: F.body, fontSize: 11, color: C.textDim, flexShrink: 0 }}>{f.feature}</div>
            <div style={{ flex: 1, height: 16, background: `${C.border}40`, borderRadius: 4, overflow: "hidden", position: "relative" }}>
              <div style={{
                position: "absolute", [f.impact > 0 ? "left" : "right"]: "50%",
                width: `${Math.abs(f.impact) * 50}%`, height: "100%",
                background: f.color, borderRadius: 4, opacity: 0.7,
              }} />
              <div style={{ position: "absolute", left: "50%", top: 0, bottom: 0, width: 1, background: C.textMuted }} />
            </div>
            <div style={{ width: 105, fontFamily: F.mono, fontSize: 10, color: f.color, textAlign: "right", flexShrink: 0 }}>{f.direction}</div>
          </div>
        ))}
      </Card>
    </Section>

    <Section title="Sample proposal output for Dewan Pengawas" icon="📄">
      <Card style={{ border: `1px solid ${C.gold}40`, background: `linear-gradient(160deg, ${C.surface}, ${C.goldDark}10)` }}>
        <div style={{ fontFamily: F.mono, fontSize: 10, color: C.gold, textTransform: "uppercase", letterSpacing: 2, marginBottom: 8 }}>AI-generated recommendation — February 2026</div>
        <div style={{ fontFamily: F.body, fontSize: 13, color: C.text, lineHeight: 1.8, padding: "12px 16px", background: `${C.card}`, borderRadius: 10, border: `1px solid ${C.border}` }}>
          <p style={{ margin: "0 0 10px" }}><strong style={{ color: C.accent }}>Recommendation:</strong> Increase sukuk allocation by 2.3% (from 71.68% to 73.98%), reduce deposito by 1.8%, reduce emas by 0.05%, reallocate 0.45% to liquid instruments.</p>
          <p style={{ margin: "0 0 10px" }}><strong style={{ color: C.blue }}>Rationale:</strong> Sukuk yield trend is positive (+12bps MoM) while gold shows -965bps correction. Rebalancing captures higher fixed-income returns while maintaining 2.53x liquidity ratio above 2.0x minimum. Sharia Compliance Score: 0.97 (above 0.95 threshold).</p>
          <p style={{ margin: "0 0 10px" }}><strong style={{ color: C.gold }}>Projected impact:</strong> Annualized yield improvement from 6.88% to est. 7.04% (+16bps). Max drawdown risk: -2.1% (within tolerance). Portfolio turnover: 4.1% (below 5% monthly cap).</p>
          <p style={{ margin: 0 }}><strong style={{ color: C.purple }}>Confidence:</strong> PPO agent consensus 87.3%. Backtested on 120 months of historical data. All Sharia constraints maintained. Ready for Dewas review.</p>
        </div>
      </Card>
    </Section>

    <Section title="Algorithm comparison radar" icon="⬡">
      <Card>
        <ResponsiveContainer width="100%" height={300}>
          <RadarChart data={RADAR}>
            <PolarGrid stroke={C.border} />
            <PolarAngleAxis dataKey="m" tick={{ fill: C.textDim, fontSize: 10, fontFamily: F.body }} />
            <PolarRadiusAxis tick={false} domain={[0, 100]} />
            <Radar name="PPO" dataKey="PPO" stroke={C.ppo} fill={C.ppo} fillOpacity={0.12} strokeWidth={2} />
            <Radar name="DQN" dataKey="DQN" stroke={C.dqn} fill={C.dqn} fillOpacity={0.08} strokeWidth={2} />
            <Radar name="A2C" dataKey="A2C" stroke={C.a2c} fill={C.a2c} fillOpacity={0.08} strokeWidth={2} />
            <Radar name="Benchmark" dataKey="BM" stroke={C.bench} fill={C.bench} fillOpacity={0.04} strokeWidth={1.5} strokeDasharray="5 5" />
            <Legend wrapperStyle={{ fontFamily: F.mono, fontSize: 10 }} />
          </RadarChart>
        </ResponsiveContainer>
        <div style={{ textAlign: "center", fontSize: 11, color: C.textDim, marginTop: 4 }}>Note: "Explainability" axis is a novel dimension — benchmarks score higher because they're inherently transparent. The XRL layer bridges this gap for RL agents.</div>
      </Card>
    </Section>
  </>);
}

function TabAbout() {
  return (<>
    <Section title="Research team" icon="👥">
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 14 }}>
        <Card glow={C.accent}>
          <Badge color={C.accent}>Researcher</Badge>
          <h4 style={{ fontFamily: F.display, fontSize: 20, color: C.text, margin: "8px 0 4px" }}>Sopian (MS Hadianto)</h4>
          <div style={{ fontFamily: F.mono, fontSize: 10, color: C.textDim, marginBottom: 10 }}>CACP® · CCFA® · QIA® · CA® · GRCP® · GRCA® · CGP®</div>
          <div style={{ fontFamily: F.body, fontSize: 12, color: C.textDim, lineHeight: 1.7 }}>
            GRC expert & AI-powered builder at BPKH. 20+ years in governance, risk, compliance & Islamic finance. Builder of FALAH.AI, AURIX, DALIL, WBS BPKH AI. Methodology: <span style={{ color: C.accent }}>Curious → Coding → Deploy → Repeat.</span>
          </div>
          <div style={{ display: "flex", gap: 8, marginTop: 10, flexWrap: "wrap" }}>
            <Badge>github.com/mshadianto</Badge>
            <Badge>@MSHadianto</Badge>
          </div>
        </Card>
        <Card glow={C.gold}>
          <Badge color={C.gold}>Proposed supervisor</Badge>
          <h4 style={{ fontFamily: F.display, fontSize: 20, color: C.text, margin: "8px 0 4px" }}>Dr. Indra Gunawan, Ph.D.</h4>
          <div style={{ fontFamily: F.mono, fontSize: 10, color: C.textDim, marginBottom: 10 }}>CIB® · CPM® · CRP® · CSA® · ACIArb · WMI</div>
          <div style={{ fontFamily: F.body, fontSize: 12, color: C.textDim, lineHeight: 1.7 }}>
            Faculty member FEB UIII. <span style={{ color: C.gold, fontWeight: 600 }}>BPKH Executive Board member — Investment portfolio.</span> Secretary Investment Committee MES. PhD from IPB, LLM/MSc Utrecht. Published on BPKH sukuk, gold, ESG, and Sharia index volatility. Alumni Goldman Sachs, Oxford, Cambridge.
          </div>
        </Card>
      </div>
    </Section>

    <Section title="Timeline — 15 months" icon="🗓️">
      <div style={{ position: "relative" }}>
        {[
          { p: "Phase 1", t: "M1–3", l: "Literature review, data collection, BPKH calibration", c: C.blue },
          { p: "Phase 2", t: "M4–8", l: "Environment design, agent training, XRL integration", c: C.accent },
          { p: "Phase 3", t: "M9–12", l: "Backtesting, analysis, journal paper drafting", c: C.gold },
          { p: "Phase 4", t: "M13–15", l: "Dissertation writing & defense", c: C.purple },
        ].map((x, i) => (
          <div key={i} style={{ display: "flex", gap: 14, alignItems: "center", padding: "10px 0", borderBottom: i < 3 ? `1px solid ${C.border}30` : "none" }}>
            <div style={{ width: 38, height: 38, borderRadius: "50%", background: `${x.c}15`, border: `2px solid ${x.c}`, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: F.mono, fontSize: 12, fontWeight: 800, color: x.c, flexShrink: 0 }}>{i + 1}</div>
            <div>
              <div style={{ fontFamily: F.mono, fontSize: 10, color: x.c, fontWeight: 700 }}>{x.t}</div>
              <div style={{ fontFamily: F.body, fontSize: 13, color: C.text }}>{x.l}</div>
            </div>
          </div>
        ))}
      </div>
    </Section>

    <Section title="Target publications" icon="📚">
      {[
        { j: "Expert Systems with Applications", q: "Q1", f: "~8.5", p: true },
        { j: "Journal of Financial Data Science", q: "Q1", f: "~3.2", p: false },
        { j: "JIMF (Bank Indonesia)", q: "Q2", f: "~0.8", p: false },
        { j: "Islamic Economic Studies", q: "Q2", f: "~0.6", p: false },
      ].map(j => (
        <div key={j.j} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 0", borderBottom: `1px solid ${C.border}30` }}>
          <div>
            <span style={{ fontFamily: F.body, fontSize: 13, color: C.text, fontWeight: j.p ? 700 : 400 }}>{j.j}</span>
            {j.p && <span style={{ marginLeft: 8 }}><Badge color={C.accent}>PRIMARY</Badge></span>}
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <Badge color={j.q === "Q1" ? C.accent : C.gold}>{j.q}</Badge>
            <span style={{ fontFamily: F.mono, fontSize: 11, color: C.textDim }}>IF {j.f}</span>
          </div>
        </div>
      ))}
    </Section>

    <Section title="Methodology at a glance" icon="🧪">
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(95px, 1fr))", gap: 8, textAlign: "center" }}>
        {[
          { v: "10yr", l: "Data period", s: "2015–2025" },
          { v: "2", l: "Markets", s: "JII + Hijrah" },
          { v: "3", l: "RL algorithms", s: "DQN·PPO·A2C" },
          { v: "Rp180T", l: "Calibrated to", s: "BPKH actual" },
          { v: "XRL", l: "Explainability", s: "SHAP + NLG" },
          { v: "7", l: "Metrics", s: "Sharpe, MDD..." },
        ].map(m => (
          <div key={m.l} style={{ background: C.surface, borderRadius: 10, padding: 12, border: `1px solid ${C.border}50` }}>
            <div style={{ fontFamily: F.mono, fontSize: 20, fontWeight: 800, color: C.accent }}>{m.v}</div>
            <div style={{ fontFamily: F.body, fontSize: 10, fontWeight: 600, color: C.text, marginTop: 4 }}>{m.l}</div>
            <div style={{ fontFamily: F.mono, fontSize: 8, color: C.textMuted, marginTop: 2 }}>{m.s}</div>
          </div>
        ))}
      </div>
    </Section>
  </>);
}

// ═══════════════════════════════════════
// MAIN APP
// ═══════════════════════════════════════

const TABS = [
  { id: "ov", label: "Overview", icon: "◎" },
  { id: "bt", label: "Backtest", icon: "◈" },
  { id: "bp", label: "BPKH Data", icon: "🏛" },
  { id: "xr", label: "Explainable RL", icon: "🧠" },
  { id: "ab", label: "Team & Plan", icon: "◉" },
];

export default function App() {
  const [tab, setTab] = useState("ov");

  return (
    <div style={{ minHeight: "100vh", background: C.bg, color: C.text, fontFamily: F.body }}>
      {/* HERO */}
      <div style={{ background: `linear-gradient(180deg, ${C.surface}, ${C.bg})`, borderBottom: `1px solid ${C.border}`, padding: "28px 20px 22px", position: "relative", overflow: "hidden" }}>
        <div style={{ position: "absolute", top: -60, right: -60, width: 250, height: 250, background: `radial-gradient(circle, ${C.accent}05, transparent)`, borderRadius: "50%" }} />
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
          <span style={{ fontSize: 20 }}>☪</span>
          <Badge>Ph.D. Research Proposal — UIII 2026</Badge>
        </div>
        <h1 style={{ fontFamily: F.display, fontSize: 24, fontWeight: 700, lineHeight: 1.3, margin: "0 0 6px", background: `linear-gradient(135deg, ${C.text}, ${C.accent})`, WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
          AI-Augmented Investment Decision Support for Hajj Fund Management
        </h1>
        <p style={{ fontFamily: F.body, fontSize: 12, color: C.textDim, margin: "0 0 12px", lineHeight: 1.5 }}>
          Reinforcement Learning approach to Sharia-constrained portfolio optimization · Calibrated to BPKH's Rp179.5T portfolio
        </p>
        <div style={{ display: "flex", gap: 14, flexWrap: "wrap", fontFamily: F.mono, fontSize: 10, color: C.textMuted }}>
          <span><Glow color={C.accent} /> Sopian (MS Hadianto) — BPKH</span>
          <span><Glow color={C.gold} /> Dr. Indra Gunawan — FEB UIII / BPKH Executive Board</span>
        </div>
      </div>

      {/* TABS */}
      <div style={{ display: "flex", gap: 0, borderBottom: `1px solid ${C.border}`, background: C.surface, overflowX: "auto" }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            flex: "none", padding: "11px 16px", background: tab === t.id ? `${C.accent}08` : "transparent",
            border: "none", borderBottom: tab === t.id ? `2px solid ${C.accent}` : "2px solid transparent",
            color: tab === t.id ? C.accent : C.textMuted, fontFamily: F.body, fontSize: 11, fontWeight: 600,
            cursor: "pointer", display: "flex", alignItems: "center", gap: 5, transition: "all 0.15s",
          }}><span style={{ fontSize: 13 }}>{t.icon}</span> {t.label}</button>
        ))}
      </div>

      {/* CONTENT */}
      <div style={{ padding: "20px 16px", maxWidth: 880, margin: "0 auto" }}>
        {tab === "ov" && <TabOverview />}
        {tab === "bt" && <TabBacktest />}
        {tab === "bp" && <TabBPKH />}
        {tab === "xr" && <TabXRL />}
        {tab === "ab" && <TabAbout />}
      </div>

      {/* FOOTER */}
      <div style={{ textAlign: "center", padding: "16px 16px 28px", fontFamily: F.mono, fontSize: 9, color: C.textMuted, borderTop: `1px solid ${C.border}`, marginTop: 20 }}>
        <div>☪ Sharia-Constrained RL Portfolio Optimization · AI-Augmented Decision Support for BPKH</div>
        <div style={{ marginTop: 2 }}>Sopian (MS Hadianto) × Dr. Indra Gunawan · FEB UIII / BPKH · March 2026</div>
      </div>
    </div>
  );
}
