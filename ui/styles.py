"""Global CSS theme — dark futuristic AI SaaS (frontend only)."""

GLOBAL_STYLES = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --ns-bg: #05070a;
    --ns-bg-elevated: #0b1018;
    --ns-bg-card: rgba(12, 18, 28, 0.72);
    --ns-border: rgba(0, 229, 255, 0.14);
    --ns-border-soft: rgba(255, 255, 255, 0.06);
    --ns-cyan: #00e5ff;
    --ns-cyan-dim: rgba(0, 229, 255, 0.55);
    --ns-purple: #a855f7;
    --ns-purple-dim: rgba(168, 85, 247, 0.5);
    --ns-green: #22c55e;
    --ns-red: #f87171;
    --ns-text: #f1f5f9;
    --ns-text-muted: #94a3b8;
    --ns-text-dim: #64748b;
    --ns-radius: 16px;
    --ns-radius-sm: 10px;
    --ns-shadow: 0 24px 64px rgba(0, 0, 0, 0.45);
    --ns-glow-cyan: 0 0 40px rgba(0, 229, 255, 0.18);
    --ns-glow-purple: 0 0 48px rgba(168, 85, 247, 0.12);
    --ns-font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --ns-mono: 'JetBrains Mono', ui-monospace, monospace;
}

html, body, [class*="css"] {
    font-family: var(--ns-font) !important;
}

.stApp {
    background:
        radial-gradient(ellipse 80% 50% at 15% -10%, rgba(0, 229, 255, 0.07), transparent 55%),
        radial-gradient(ellipse 60% 40% at 85% 5%, rgba(168, 85, 247, 0.06), transparent 50%),
        var(--ns-bg) !important;
    color: var(--ns-text) !important;
}

.block-container {
    max-width: 1200px !important;
    padding-top: 0.5rem !important;
    padding-bottom: 3rem !important;
    animation: ns-fade-up 0.5s ease-out both;
}

@keyframes ns-fade-up {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes ns-pulse-glow {
    0%, 100% { box-shadow: 0 0 24px rgba(0, 229, 255, 0.25); }
    50% { box-shadow: 0 0 40px rgba(0, 229, 255, 0.45); }
}

@keyframes ns-spin {
    to { transform: rotate(360deg); }
}

@keyframes ns-shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stToolbar"] {
    display: none !important;
}

/* ── Navbar ───────────────────────────────────────────── */
.ns-navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.85rem 0;
    margin-bottom: 0.25rem;
    border-bottom: 1px solid var(--ns-border-soft);
    animation: ns-fade-up 0.4s ease-out both;
}

.ns-brand {
    display: flex;
    align-items: center;
    gap: 0.65rem;
}

.ns-brand-icon {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: linear-gradient(135deg, rgba(0, 229, 255, 0.2), rgba(168, 85, 247, 0.15));
    border: 1px solid var(--ns-border);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    box-shadow: var(--ns-glow-cyan);
}

.ns-brand-text {
    display: flex;
    flex-direction: column;
    line-height: 1.15;
}

.ns-brand-name {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--ns-text) !important;
    letter-spacing: -0.02em;
}

.ns-brand-tag {
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    color: var(--ns-cyan-dim) !important;
    text-transform: uppercase;
}

.ns-nav-links {
    display: flex;
    gap: 1.75rem;
    align-items: center;
}

.ns-nav-link {
    font-size: 0.88rem;
    font-weight: 500;
    color: var(--ns-text-muted) !important;
    text-decoration: none;
    transition: color 0.2s ease;
}

.ns-nav-link:hover {
    color: var(--ns-cyan) !important;
}

.ns-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.4rem 0.85rem;
    border-radius: 999px;
    background: rgba(12, 18, 28, 0.9);
    border: 1px solid var(--ns-border-soft);
    font-size: 0.78rem;
    color: var(--ns-text-muted) !important;
}

.ns-status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--ns-green);
    box-shadow: 0 0 8px var(--ns-green);
    animation: ns-pulse-glow 2s ease-in-out infinite;
}

.ns-status-dot.offline {
    background: var(--ns-red);
    box-shadow: 0 0 8px var(--ns-red);
}

/* ── Hero ───────────────────────────────────────────────── */
.ns-hero {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2.5rem;
    align-items: center;
    padding: 2.5rem 0 2rem;
    animation: ns-fade-up 0.55s ease-out 0.05s both;
}

@media (max-width: 900px) {
    .ns-hero { grid-template-columns: 1fr; }
    .ns-hero-visual { min-height: 280px; }
}

.ns-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    background: rgba(0, 229, 255, 0.08);
    border: 1px solid rgba(0, 229, 255, 0.22);
    font-family: var(--ns-mono);
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    color: var(--ns-cyan) !important;
    margin-bottom: 1rem;
}

.ns-badge-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--ns-cyan);
}

.ns-hero-title {
    margin: 0 0 1rem;
    font-size: clamp(2rem, 4.5vw, 2.85rem);
    font-weight: 800;
    line-height: 1.08;
    letter-spacing: -0.03em;
    color: var(--ns-text) !important;
}

.ns-gradient-text {
    background: linear-gradient(90deg, var(--ns-cyan), #7dd3fc 45%, var(--ns-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.ns-hero-desc {
    margin: 0 0 1.5rem;
    font-size: 0.98rem;
    line-height: 1.65;
    color: var(--ns-text-muted) !important;
    max-width: 34rem;
}

.ns-hero-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin-bottom: 2rem;
}

.ns-hero-stats {
    display: flex;
    flex-wrap: wrap;
    gap: 2.25rem;
}

.ns-stat-value {
    font-size: 1.65rem;
    font-weight: 800;
    color: var(--ns-text) !important;
    letter-spacing: -0.02em;
}

.ns-stat-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    color: var(--ns-text-dim) !important;
    text-transform: uppercase;
    margin-top: 0.15rem;
}

/* Hero visual / radar */
.ns-hero-visual {
    position: relative;
    min-height: 340px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.ns-radar {
    position: relative;
    width: min(340px, 90vw);
    height: min(340px, 90vw);
}

.ns-radar-ring {
    position: absolute;
    inset: 0;
    border: 1px solid rgba(0, 229, 255, 0.12);
    border-radius: 50%;
}

.ns-radar-ring:nth-child(2) { inset: 12%; }
.ns-radar-ring:nth-child(3) { inset: 24%; }
.ns-radar-ring:nth-child(4) { inset: 36%; }

.ns-radar-core {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 88px;
    height: 88px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0, 229, 255, 0.35), rgba(0, 229, 255, 0.05));
    border: 1px solid rgba(0, 229, 255, 0.4);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    animation: ns-pulse-glow 3s ease-in-out infinite;
}

.ns-node {
    position: absolute;
    padding: 0.3rem 0.65rem;
    border-radius: 999px;
    font-family: var(--ns-mono);
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    border: 1px solid var(--ns-border-soft);
    background: rgba(8, 12, 20, 0.85);
    color: var(--ns-text-muted) !important;
    white-space: nowrap;
}

.ns-node.cyan { border-color: rgba(0, 229, 255, 0.35); color: var(--ns-cyan) !important; }
.ns-node.purple { border-color: rgba(168, 85, 247, 0.35); color: #c4b5fd !important; }

/* ── Glass & sections ───────────────────────────────────── */
.ns-glass {
    background: var(--ns-bg-card);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--ns-border-soft);
    border-radius: var(--ns-radius);
    box-shadow: var(--ns-shadow);
}

.ns-section {
    margin: 2.25rem 0 1rem;
    animation: ns-fade-up 0.5s ease-out both;
}

.ns-section-head {
    display: flex;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 0.5rem 0.75rem;
    margin-bottom: 1.1rem;
}

.ns-section-num {
    font-family: var(--ns-mono);
    font-size: 0.95rem;
    font-weight: 500;
    color: var(--ns-cyan) !important;
}

.ns-section-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--ns-text) !important;
    letter-spacing: -0.02em;
}

.ns-section-sub {
    font-family: var(--ns-mono);
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    color: var(--ns-text-dim) !important;
    text-transform: uppercase;
}

/* ── Upload zone ────────────────────────────────────────── */
.ns-upload-wrap {
    padding: 2.5rem 1.5rem 1.75rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
}

.ns-upload-wrap::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(0, 229, 255, 0.06), transparent 60%);
    pointer-events: none;
}

.ns-upload-icon {
    width: 52px;
    height: 52px;
    margin: 0 auto 1rem;
    border-radius: 12px;
    background: rgba(0, 229, 255, 0.1);
    border: 1px solid rgba(0, 229, 255, 0.25);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.35rem;
    box-shadow: var(--ns-glow-cyan);
}

.ns-upload-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--ns-text) !important;
    margin-bottom: 0.35rem;
}

.ns-upload-hint {
    font-size: 0.82rem;
    color: var(--ns-text-dim) !important;
    margin-bottom: 1.25rem;
}

.ns-tags {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.5rem;
}

.ns-tag {
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    font-family: var(--ns-mono);
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    color: var(--ns-text-dim) !important;
    border: 1px solid var(--ns-border-soft);
    background: rgba(0, 0, 0, 0.25);
}

/* ── Comparison cards ───────────────────────────────────── */
.ns-compare-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.25rem;
}

@media (max-width: 768px) {
    .ns-compare-grid { grid-template-columns: 1fr; }
}

.ns-compare-card {
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.ns-compare-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.35);
}

.ns-compare-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.85rem 1rem;
    border-bottom: 1px solid var(--ns-border-soft);
    background: rgba(0, 0, 0, 0.2);
}

.ns-compare-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--ns-text) !important;
}

.ns-compare-meta {
    font-family: var(--ns-mono);
    font-size: 0.72rem;
    color: var(--ns-text-dim) !important;
}

.ns-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}

.ns-dot.red { background: var(--ns-red); box-shadow: 0 0 8px rgba(248, 113, 113, 0.6); }
.ns-dot.cyan { background: var(--ns-cyan); box-shadow: 0 0 8px rgba(0, 229, 255, 0.5); }

.ns-compare-sub {
    font-weight: 400;
    color: var(--ns-text-dim) !important;
    font-size: 0.82rem;
}

.ns-compare-body {
    min-height: 220px;
    background: rgba(0, 0, 0, 0.35);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0.5rem;
}

.ns-placeholder {
    text-align: center;
    color: var(--ns-text-dim) !important;
    font-size: 0.85rem;
}

.ns-placeholder-icon {
    font-size: 2rem;
    opacity: 0.35;
    margin-bottom: 0.5rem;
}

/* ── Metric cards ───────────────────────────────────────── */
.ns-metrics-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
}

.ns-metrics-grid--two {
    grid-template-columns: repeat(2, 1fr);
    max-width: 720px;
}

@media (max-width: 768px) {
    .ns-metrics-grid,
    .ns-metrics-grid--two { grid-template-columns: 1fr; }
}

.ns-metric-card {
    padding: 1.15rem 1.2rem 1.25rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease;
}

.ns-metric-card:hover {
    transform: translateY(-2px);
}

.ns-metric-card::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--ns-cyan), var(--ns-purple));
    opacity: 0.85;
}

.ns-metric-icon {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    background: rgba(0, 229, 255, 0.1);
    border: 1px solid rgba(0, 229, 255, 0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 0.75rem;
    font-size: 1rem;
}

.ns-metric-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--ns-text) !important;
    margin-bottom: 0.15rem;
}

.ns-metric-desc {
    font-size: 0.78rem;
    color: var(--ns-text-dim) !important;
    margin-bottom: 0.85rem;
}

.ns-metric-value {
    font-size: 1.75rem;
    font-weight: 800;
    color: var(--ns-text) !important;
    letter-spacing: -0.02em;
}

.ns-metric-bar {
    height: 4px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.06);
    margin-top: 0.65rem;
    overflow: hidden;
}

.ns-metric-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--ns-cyan), var(--ns-purple));
    transition: width 0.6s ease-out;
}

.ns-metric-unit {
    font-family: var(--ns-mono);
    font-size: 0.68rem;
    color: var(--ns-text-dim) !important;
    margin-top: 0.35rem;
}

/* ── Architecture cards ─────────────────────────────────── */
.ns-arch-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
}

@media (max-width: 900px) {
    .ns-arch-grid { grid-template-columns: 1fr; }
}

.ns-arch-card {
    padding: 1.25rem;
    transition: transform 0.2s ease;
}

.ns-arch-card:hover {
    transform: translateY(-2px);
    border-color: rgba(0, 229, 255, 0.2);
}

.ns-arch-icon {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    background: rgba(0, 229, 255, 0.08);
    border: 1px solid rgba(0, 229, 255, 0.18);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 0.85rem;
    font-size: 1.1rem;
}

.ns-arch-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--ns-text) !important;
    margin-bottom: 0.4rem;
}

.ns-arch-desc {
    font-size: 0.8rem;
    line-height: 1.55;
    color: var(--ns-text-dim) !important;
}

/* ── Loading ────────────────────────────────────────────── */
.ns-loading {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    border-radius: var(--ns-radius-sm);
    background: rgba(0, 229, 255, 0.06);
    border: 1px solid rgba(0, 229, 255, 0.2);
    animation: ns-fade-up 0.3s ease-out both;
}

.ns-spinner {
    width: 28px;
    height: 28px;
    border: 2px solid rgba(0, 229, 255, 0.2);
    border-top-color: var(--ns-cyan);
    border-radius: 50%;
    animation: ns-spin 0.8s linear infinite;
    flex-shrink: 0;
}

.ns-loading-title {
    font-weight: 600;
    color: var(--ns-cyan) !important;
    font-size: 0.92rem;
}

.ns-loading-sub {
    font-size: 0.8rem;
    color: var(--ns-text-dim) !important;
    margin-top: 0.15rem;
}

.ns-progress-track {
    height: 3px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.06);
    margin-top: 0.75rem;
    overflow: hidden;
}

.ns-progress-fill {
    height: 100%;
    width: 40%;
    background: linear-gradient(90deg, var(--ns-cyan), var(--ns-purple));
    border-radius: 999px;
    animation: ns-shimmer 1.5s ease-in-out infinite;
    background-size: 200% 100%;
}

/* ── Footer ─────────────────────────────────────────────── */
.ns-footer {
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--ns-border-soft);
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    animation: ns-fade-up 0.5s ease-out 0.1s both;
}

.ns-footer-left {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.82rem;
    color: var(--ns-text-dim) !important;
}

.ns-footer-links {
    display: flex;
    gap: 1.25rem;
}

.ns-footer-link {
    font-size: 0.82rem;
    color: var(--ns-text-dim) !important;
    text-decoration: none;
    transition: color 0.2s ease;
}

.ns-footer-link:hover {
    color: var(--ns-cyan) !important;
}

/* ── Streamlit widget overrides ─────────────────────────── */
section[data-testid="stSidebar"] {
    background: rgba(8, 12, 20, 0.95) !important;
    border-right: 1px solid var(--ns-border-soft) !important;
}

section[data-testid="stSidebar"] * {
    color: var(--ns-text) !important;
}

section[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: rgba(12, 18, 28, 0.9) !important;
    border: 1px solid var(--ns-border-soft) !important;
    border-radius: var(--ns-radius-sm) !important;
}

[data-testid="stFileUploader"] {
    margin-top: 0.5rem !important;
}

[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: 1px dashed rgba(0, 229, 255, 0.28) !important;
    border-radius: 12px !important;
    min-height: 72px !important;
}

.ns-compare-body [data-testid="stImage"] {
    width: 100%;
}

[data-testid="stFileUploader"] section:hover,
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(0, 229, 255, 0.5) !important;
    background: rgba(0, 229, 255, 0.04) !important;
}

[data-testid="stFileUploader"] * {
    color: var(--ns-text-muted) !important;
}

[data-testid="stFileUploader"] button {
    background: rgba(0, 229, 255, 0.1) !important;
    border: 1px solid rgba(0, 229, 255, 0.3) !important;
    color: var(--ns-cyan) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: rgba(8, 12, 20, 0.9) !important;
    color: var(--ns-text) !important;
    border-color: var(--ns-border-soft) !important;
    border-radius: var(--ns-radius-sm) !important;
}

.stButton > button[kind="primary"],
.stDownloadButton > button {
    background: linear-gradient(135deg, #00e5ff, #06b6d4) !important;
    color: #05070a !important;
    border: none !important;
    border-radius: var(--ns-radius-sm) !important;
    font-weight: 700 !important;
    padding: 0.55rem 1.25rem !important;
    box-shadow: 0 4px 24px rgba(0, 229, 255, 0.35) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}

.stButton > button[kind="primary"]:hover,
.stDownloadButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 32px rgba(0, 229, 255, 0.45) !important;
}

.stButton > button[kind="secondary"] {
    background: transparent !important;
    color: var(--ns-text) !important;
    border: 1px solid var(--ns-border-soft) !important;
    border-radius: var(--ns-radius-sm) !important;
}

[data-testid="stImage"] img {
    border-radius: 8px;
    width: 100%;
    object-fit: contain;
}

div[data-testid="stExpander"] {
    background: var(--ns-bg-card) !important;
    border: 1px solid var(--ns-border-soft) !important;
    border-radius: var(--ns-radius-sm) !important;
}

.ns-hidden-label label {
    display: none !important;
}

.ns-spacer-sm { height: 0.75rem; }
.ns-spacer-md { height: 1.25rem; }
"""
