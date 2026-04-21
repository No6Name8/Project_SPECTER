"""
specter/app.py — SPECTER Competition Dashboard
Run from inside the specter/ folder:
    streamlit run app.py
"""

import os, sys, io, time, warnings
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── path setup ────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_HERE, "results")
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data.dataset          import MODULATION_CLASSES
from models.specter_cnn    import SpecterCNN
from models.baseline_cnn   import VT_CNN2
from models.open_set       import EnergyOpenSetDetector
from scoring.threat_scorer import ThreatScorer
from geolocation.tdoa_simulator import TDOASimulator

# ── constants ─────────────────────────────────────────────────────────────────
NUM_CLASSES  = 24
INPUT_LEN    = 1024
CLASS_NAMES  = MODULATION_CLASSES
SIM_OPTIONS  = CLASS_NAMES + ["??? Unknown/Custom"]
LEVEL_COLORS = {"LOW": "#00cc44", "MEDIUM": "#ffaa00", "CRITICAL": "#ff4444"}
LEVEL_BG     = {"LOW": "#0d3b1f", "MEDIUM": "#3b2d0d", "CRITICAL": "#3b0d0d"}

# ── must be FIRST streamlit call ──────────────────────────────────────────────
st.set_page_config(
    page_title="SPECTER — Signal Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* hide streamlit chrome */
  #MainMenu, footer, header {visibility: hidden;}

  /* card containers */
  .specter-card {
    background: #161b2e;
    border: 1px solid #2a3050;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }
  .specter-card h4 {
    color: #8899cc;
    font-size: 0.75em;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 0 0 6px 0;
  }
  .specter-card .val {
    color: #ffffff;
    font-size: 2em;
    font-weight: 700;
    line-height: 1.1;
  }
  .specter-card .sub {
    color: #556688;
    font-size: 0.8em;
    margin-top: 4px;
  }

  /* pipeline boxes (tab 4) */
  .pipe-box {
    background: #1a2040;
    border: 1px solid #3a4570;
    border-radius: 8px;
    padding: 14px 10px;
    text-align: center;
    font-weight: 600;
    font-size: 0.85em;
    color: #c0ccee;
  }
  .pipe-arrow {
    text-align: center;
    font-size: 1.4em;
    color: #4fa3e0;
    padding-top: 14px;
  }

  /* innovation cards */
  .inno-card {
    background: #0f1825;
    border-left: 4px solid #4fa3e0;
    border-radius: 0 10px 10px 0;
    padding: 18px 20px;
    height: 100%;
  }
  .inno-card h3 { color: #4fa3e0; margin: 0 0 10px 0; font-size: 1em; }
  .inno-card p  { color: #8899aa; font-size: 0.87em; line-height: 1.6; margin: 0; }

  /* trace steps */
  .trace-step {
    padding: 6px 0;
    font-family: monospace;
    font-size: 0.9em;
    border-bottom: 1px solid #1e2840;
  }

  /* threat breakdown table */
  .breakdown-row {
    display: flex;
    justify-content: space-between;
    padding: 7px 0;
    border-bottom: 1px solid #1e2840;
    font-size: 0.88em;
  }
  .breakdown-row:last-child { border-bottom: none; }
  .bd-name  { color: #8899cc; }
  .bd-score { color: #ffffff; font-weight: 600; }
  .bd-wt    { color: #445566; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING  (cached — loaded once at startup)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    """Returns (specter, baseline, detector, demo_mode: bool)."""
    specter_path  = os.path.join(_RESULTS, "specter_best.pt")
    baseline_path = os.path.join(_RESULTS, "baseline_best.pt")
    thresh_candidates = [
        os.path.join(_RESULTS, "open_set_threshold.pkl"),
        os.path.join(_RESULTS, "open_set_threshold.json"),
    ]

    demo_mode = False

    # SPECTER model
    specter = SpecterCNN(num_classes=NUM_CLASSES)
    if os.path.isfile(specter_path):
        specter.load_state_dict(torch.load(specter_path, map_location="cpu"))
    else:
        demo_mode = True
    specter.eval()

    # Baseline model
    baseline = VT_CNN2(num_classes=NUM_CLASSES, input_len=INPUT_LEN)
    if os.path.isfile(baseline_path):
        baseline.load_state_dict(torch.load(baseline_path, map_location="cpu"))
    baseline.eval()

    # Open-set detector
    detector = None
    for p in thresh_candidates:
        if os.path.isfile(p):
            detector = EnergyOpenSetDetector()
            detector.load_threshold(p)
            break

    return specter, baseline, detector, demo_mode


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _norm(sig: np.ndarray) -> np.ndarray:
    p = np.mean(np.abs(sig) ** 2)
    return sig / np.sqrt(p + 1e-12)


def generate_signal(modulation: str, n: int = INPUT_LEN) -> np.ndarray:
    """Return complex IQ samples (unit power, no noise) for the given modulation."""
    rng = np.random.default_rng(42 + abs(hash(modulation)) % 9999)
    sps = 8   # samples per symbol

    def _psk(M):
        sym = rng.integers(0, M, n // sps)
        ph  = 2 * np.pi * sym / M
        return np.repeat(np.exp(1j * ph), sps)[:n]

    def _ask(M):
        levels = np.linspace(-1, 1, M)
        sym = levels[rng.integers(0, M, n // sps)]
        return np.repeat(sym.astype(complex), sps)[:n]

    def _qam(M):
        s = int(np.sqrt(M))
        pts = np.arange(s) * 2 - (s - 1)
        i_s = rng.choice(pts, n // sps)
        q_s = rng.choice(pts, n // sps)
        return np.repeat((i_s + 1j * q_s) / (s - 1 + 1e-9), sps)[:n]

    def _apsk(M):
        inner = max(4, M // 4)
        outer = M - inner
        sym   = rng.integers(0, M, n // sps)
        inn_m = sym < inner
        ph_i  = sym[inn_m] * 2 * np.pi / inner
        ph_o  = (sym[~inn_m] - inner) * 2 * np.pi / outer
        base  = np.zeros(len(sym), dtype=complex)
        base[inn_m]  = 0.45 * np.exp(1j * ph_i)
        base[~inn_m] = 1.00 * np.exp(1j * ph_o)
        return np.repeat(base, sps)[:n]

    t = np.arange(n)

    dispatch = {
        "OOK":      lambda: np.repeat(rng.integers(0, 2, n // sps).astype(complex), sps)[:n],
        "4ASK":     lambda: _ask(4),
        "8ASK":     lambda: _ask(8),
        "BPSK":     lambda: _psk(2),
        "QPSK":     lambda: _psk(4),
        "8PSK":     lambda: _psk(8),
        "16PSK":    lambda: _psk(16),
        "32PSK":    lambda: _psk(32),
        "16APSK":   lambda: _apsk(16),
        "32APSK":   lambda: _apsk(32),
        "64APSK":   lambda: _apsk(64),
        "128APSK":  lambda: _apsk(128),
        "16QAM":    lambda: _qam(16),
        "32QAM":    lambda: _qam(32),
        "64QAM":    lambda: _qam(64),
        "128QAM":   lambda: _qam(128),
        "256QAM":   lambda: _qam(256),
        "AM-SSB-WC": lambda: (0.5 + np.sin(2*np.pi*0.05*t)) * np.exp(1j*2*np.pi*0.15*t),
        "AM-SSB-SC": lambda: np.sin(2*np.pi*0.05*t) * np.exp(1j*2*np.pi*0.15*t),
        "AM-DSB-WC": lambda: (1 + 0.8*np.sin(2*np.pi*0.04*t)) * np.exp(1j*2*np.pi*0.10*t),
        "AM-DSB-SC": lambda: np.sin(2*np.pi*0.04*t) * np.exp(1j*2*np.pi*0.10*t),
        "FM": lambda: np.exp(1j*(2*np.pi*0.1*t + 0.5*np.cumsum(np.sin(2*np.pi*0.03*t)))),
        "GMSK": lambda: np.exp(1j*(np.pi/2*np.cumsum(
            np.convolve(
                np.repeat(rng.choice([-1,1], n//sps), sps)[:n].astype(float),
                np.exp(-np.arange(-4,5)**2/3.0) / np.exp(-np.arange(-4,5)**2/3.0).sum(),
                mode="same"
            ) / n * n
        ))),
        "OQPSK": lambda: (
            lambda bi, bq: (
                np.repeat(bi, sps)[:n] +
                1j * np.concatenate([[0]*( sps//2), np.repeat(bq, sps)])[:n]
            ) / np.sqrt(2)
        )(rng.choice([-1,1], n//sps), rng.choice([-1,1], n//sps)),
        "??? Unknown/Custom": lambda: np.exp(1j*(np.pi*0.003*t**2)) * (
            1 + 0.4*np.sin(2*np.pi*0.07*t + np.sin(2*np.pi*0.02*t))
        ),
    }

    fn  = dispatch.get(modulation, dispatch["QPSK"])
    sig = fn()
    return _norm(sig[:n])


def add_awgn(sig: np.ndarray, snr_db: float) -> np.ndarray:
    """Add AWGN at the requested SNR level."""
    noise_pwr = 1.0 / (10 ** (snr_db / 10.0))
    noise = np.sqrt(noise_pwr / 2) * (
        np.random.randn(len(sig)) + 1j * np.random.randn(len(sig))
    )
    return _norm(sig + noise)


def to_tensor(sig: np.ndarray) -> torch.Tensor:
    """(N,) complex → (1, 2, N) float32 tensor."""
    iq = np.stack([sig.real, sig.imag], axis=0).astype(np.float32)
    return torch.from_numpy(iq).unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_specter(model, detector, x: torch.Tensor) -> dict:
    """Full SPECTER inference: classify + open-set check."""
    logits = model(x)                           # (1, C)
    probs  = torch.softmax(logits, dim=1)[0]    # (C,)

    class_id   = int(probs.argmax().item())
    confidence = float(probs.max().item())

    is_unknown = False
    if detector is not None:
        pred       = detector.predict(logits)[0]
        is_unknown = pred.is_unknown
        class_id   = pred.class_id if not is_unknown else -1

    return {
        "logits":     logits,
        "probs":      probs.numpy(),
        "class_id":   class_id,
        "confidence": confidence,
        "is_unknown": is_unknown,
        "class_name": "UNKNOWN" if is_unknown else CLASS_NAMES[class_id],
    }


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#111827",
    font=dict(color="#c0ccee", size=11),
    margin=dict(t=44, b=36, l=48, r=16),
)


def plot_constellation(sig: np.ndarray, title="IQ Constellation") -> go.Figure:
    pts = min(800, len(sig))
    fig = go.Figure(go.Scatter(
        x=sig[:pts].real, y=sig[:pts].imag,
        mode="markers",
        marker=dict(size=3, color="#4fa3e0", opacity=0.55),
    ))
    fig.update_layout(
        **_DARK_LAYOUT, title=dict(text=title, font=dict(size=12)),
        height=310,
        xaxis=dict(title="I", gridcolor="#1e2840", zeroline=True, zerolinecolor="#2a3860"),
        yaxis=dict(title="Q", gridcolor="#1e2840", zeroline=True, zerolinecolor="#2a3860",
                   scaleanchor="x"),
    )
    return fig


def plot_psd(sig: np.ndarray, title="Power Spectral Density") -> go.Figure:
    n    = len(sig)
    freq = np.fft.fftshift(np.fft.fftfreq(n))
    psd  = np.fft.fftshift(np.abs(np.fft.fft(sig)) ** 2) / n
    psd_db = 10 * np.log10(psd + 1e-12)
    fig = go.Figure(go.Scatter(
        x=freq, y=psd_db, mode="lines",
        line=dict(color="#4fa3e0", width=1.4),
        fill="tozeroy", fillcolor="rgba(79,163,224,0.08)",
    ))
    fig.update_layout(
        **_DARK_LAYOUT, title=dict(text=title, font=dict(size=12)),
        height=230,
        xaxis=dict(title="Normalised frequency", gridcolor="#1e2840"),
        yaxis=dict(title="dB",                   gridcolor="#1e2840"),
    )
    return fig


def plot_top5(probs: np.ndarray, is_unknown: bool) -> go.Figure:
    idx   = np.argsort(probs)[-5:][::-1]
    names = [CLASS_NAMES[i] for i in idx]
    vals  = probs[idx]
    colors = ["#ff4444" if is_unknown else "#00cc44"] + ["#4fa3e0"] * 4
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in vals],
        textposition="inside", textfont=dict(color="white", size=11),
    ))
    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text="Top-5 Class Probabilities", font=dict(size=12)),
        height=240,
        xaxis=dict(range=[0, 1], title="Probability", gridcolor="#1e2840"),
        yaxis=dict(gridcolor="#1e2840"),
    )
    return fig


def make_gauge(value: float) -> go.Figure:
    color = LEVEL_COLORS["LOW"] if value < 40 else (
            LEVEL_COLORS["MEDIUM"] if value < 70 else LEVEL_COLORS["CRITICAL"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(font=dict(color=color, size=52)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#445566", tickfont=dict(color="#8899aa")),
            bar=dict(color=color, thickness=0.28),
            bgcolor="#111827",
            steps=[
                dict(range=[0,  40], color="#0d3b1f"),
                dict(range=[40, 70], color="#3b2d0d"),
                dict(range=[70,100], color="#3b0d0d"),
            ],
            borderwidth=0,
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c0ccee"),
        height=220,
        margin=dict(t=20, b=0, l=20, r=20),
    )
    return fig


def plot_tdoa(receivers, true_pos, est_pos, conf_radius, noise_std) -> plt.Figure:
    """Matplotlib figure for TDOA — compatible with st.pyplot()."""
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#161b2e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3860")

    rx = np.array(receivers)
    ax.scatter(rx[:, 0], rx[:, 1], marker="^", s=180, color="#4fa3e0",
               zorder=5, label="Receiver")
    for i, (x, y) in enumerate(rx):
        ax.annotate(f" RX{i}", (x, y), color="#4fa3e0", fontsize=9)
    for i in range(len(rx)):
        for j in range(i + 1, len(rx)):
            ax.plot([rx[i,0], rx[j,0]], [rx[i,1], rx[j,1]],
                    color="#2a3860", linewidth=0.8, linestyle="--")

    tx, ty = true_pos
    ex, ey = est_pos
    ax.scatter(tx, ty, marker="*", s=320, color="#00cc44", zorder=6, label="True")
    ax.scatter(ex, ey, marker="x", s=160, color="#ff4444", linewidths=2.5,
               zorder=6, label=f"Estimated ±{conf_radius:.0f} m")
    circle = mpatches.Circle((ex, ey), conf_radius, fill=False,
                              edgecolor="#ff4444", linewidth=1.5, linestyle=":")
    ax.add_patch(circle)
    err = np.linalg.norm(np.array(true_pos) - np.array(est_pos))
    ax.plot([tx, ex], [ty, ey], color="#888", linewidth=1.2,
            label=f"Error: {err:.1f} m")

    ax.set_title(f"⚠  SIMULATED  —  TDOA Geolocation  |  noise σ={noise_std:.0f} m",
                 color="#c0ccee", fontsize=11)
    ax.set_xlabel("East (m)",  color="#8899aa")
    ax.set_ylabel("North (m)", color="#8899aa")
    ax.tick_params(colors="#8899aa")
    ax.grid(True, color="#1e2840", linewidth=0.5)
    leg = ax.legend(fontsize=9)
    for t in leg.get_texts():
        t.set_color("#c0ccee")
    leg.get_frame().set_facecolor("#111827")
    leg.get_frame().set_edgecolor("#2a3860")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SMALL UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def metric_card(label: str, value: str, sub: str = "") -> str:
    return f"""
    <div class="specter-card">
      <h4>{label}</h4>
      <div class="val">{value}</div>
      <div class="sub">{sub}</div>
    </div>"""


def threat_badge_html(level: str, score: float) -> str:
    c  = LEVEL_COLORS.get(level, "#4fa3e0")
    bg = LEVEL_BG.get(level, "#0d1a2e")
    return f"""
    <div style="background:{bg};border:2px solid {c};border-radius:14px;
                padding:22px;text-align:center;box-shadow:0 0 28px {c}44;">
      <div style="font-size:0.75em;letter-spacing:3px;color:{c};margin-bottom:6px;">
        THREAT LEVEL
      </div>
      <div style="font-size:3em;font-weight:900;color:{c};letter-spacing:6px;
                  text-shadow:0 0 20px {c};">
        {level}
      </div>
      <div style="font-size:1.3em;color:#c0ccee;margin-top:6px;">
        {score:.1f} / 100
      </div>
    </div>"""


def result_banner(class_name: str, confidence: float, is_unknown: bool) -> str:
    icon  = "⚠️" if is_unknown else "✓"
    color = "#ff4444" if is_unknown else "#00cc44"
    label = "UNKNOWN SIGNAL" if is_unknown else class_name
    return f"""
    <div style="background:#161b2e;border:2px solid {color};border-radius:12px;
                padding:18px 24px;text-align:center;">
      <div style="font-size:2.2em;font-weight:800;color:{color};">
        {icon}  {label}
      </div>
      <div style="color:#8899aa;font-size:0.9em;margin-top:6px;">
        confidence: {confidence:.3f}
        {"  •  open-set detector: REJECTED" if is_unknown else "  •  open-set detector: accepted"}
      </div>
    </div>"""


def breakdown_html(breakdown: dict) -> str:
    weights = {
        "uncertainty_score": ("Uncertainty",    "25%"),
        "snr_danger":        ("SNR Danger",     "30%"),
        "unknown_penalty":   ("Unknown Penalty","35%"),
        "burst_suspicion":   ("Burst Pattern",  "10%"),
    }
    rows = ""
    for key, (name, wt) in weights.items():
        val = breakdown.get(key, 0.0)
        bar_w = int(val * 100)
        bar_c = "#ff4444" if val > 0.7 else "#ffaa00" if val > 0.3 else "#00cc44"
        rows += f"""
        <div class="breakdown-row">
          <span class="bd-name">{name}</span>
          <span style="flex:1;margin:0 12px;">
            <div style="background:#1e2840;border-radius:4px;height:6px;margin-top:6px;">
              <div style="width:{bar_w}%;background:{bar_c};height:6px;border-radius:4px;"></div>
            </div>
          </span>
          <span class="bd-score">{val:.3f}</span>
          <span class="bd-wt" style="margin-left:10px;width:32px;text-align:right;">{wt}</span>
        </div>"""
    return f'<div style="margin-top:8px;">{rows}</div>'


def pipeline_trace_html(steps: list[tuple[str, str]]) -> str:
    rows = ""
    for icon, text in steps:
        rows += f'<div class="trace-step">{icon}&nbsp;&nbsp;<span style="color:#c0ccee;">{text}</span></div>'
    return f'<div style="font-family:monospace;">{rows}</div>'


def image_card(path: str, caption: str, placeholder: str) -> None:
    if os.path.isfile(path):
        st.image(path, use_container_width=True)
        st.caption(caption)
    else:
        st.markdown(f"""
        <div style="background:#161b2e;border:1px dashed #2a3860;border-radius:10px;
                    padding:40px;text-align:center;color:#445566;">
          <div style="font-size:1.8em;margin-bottom:10px;">📊</div>
          <div>{placeholder}</div>
          <div style="font-size:0.8em;margin-top:8px;color:#334455;">
            Run <code>python evaluate.py</code> after training to generate this chart
          </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    specter, baseline, detector, demo_mode = load_models()

    # ── header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:28px 0 8px 0;">
      <div style="font-size:3.2em;font-weight:900;letter-spacing:10px;
                  color:#4fa3e0;text-shadow:0 0 30px #4fa3e044;">
        SPECTER
      </div>
      <div style="color:#445566;font-size:0.95em;letter-spacing:3px;margin-top:4px;">
        SIGNAL PROCESSING &amp; EXTRACTION OF COVERT TRANSMISSIONS
        IN ELECTROMAGNETIC REGIMES
      </div>
      <div style="color:#8899aa;font-style:italic;margin-top:8px;font-size:0.9em;">
        "Seeing what every other system calls nothing"
      </div>
    </div>
    """, unsafe_allow_html=True)

    if demo_mode:
        st.warning(
            "⚠️  **Demo mode** — trained checkpoints not found in `results/`.  "
            "Run `python train.py` to train the models.  "
            "All pipeline components are active with random weights — "
            "classification results are illustrative only.",
            icon=None,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📡  Live Signal Detection",
        "📊  Benchmark Results",
        "🗺️   Threat Map",
        "⚙️   How SPECTER Works",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — LIVE SIGNAL DETECTION
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("## Feed a Signal — Watch SPECTER Decide")
        st.caption(
            "Simulate or upload a real IQ signal and see SPECTER classify, "
            "score, and assess the threat in real time."
        )
        st.markdown("---")

        col_in, col_out = st.columns([1, 2], gap="large")

        # ── LEFT: inputs ──────────────────────────────────────────────────────
        with col_in:
            st.markdown("### Signal Input")

            snr_db = st.slider(
                "Signal Strength (lower = more covert)",
                min_value=-20, max_value=30, value=-10, step=2,
                format="%d dB",
            )
            snr_color = "#ff4444" if snr_db < -6 else "#ffaa00" if snr_db < 10 else "#00cc44"
            snr_label = "⚠️ COVERT RANGE" if snr_db < -6 else "MODERATE" if snr_db < 10 else "✓ DETECTABLE"
            st.markdown(
                f'<div style="background:{snr_color}22;border:1px solid {snr_color};'
                f'color:{snr_color};padding:5px 12px;border-radius:6px;'
                f'font-size:0.82em;text-align:center;margin-bottom:12px;">'
                f'{snr_label} — {snr_db:+d} dB</div>',
                unsafe_allow_html=True,
            )

            modulation = st.selectbox(
                "Modulation to simulate",
                SIM_OPTIONS,
                index=3,   # BPSK default
                help="Select a waveform type. '??? Unknown/Custom' generates a "
                     "signal outside the training distribution.",
            )

            burst_ms = st.slider(
                "Burst Duration",
                min_value=10, max_value=500, value=100, step=10,
                format="%d ms",
            )
            if burst_ms < 50:
                st.markdown(
                    '<div style="color:#ffaa00;font-size:0.8em;">⚠️ Short burst — '
                    'increases suspicion score</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            transmit_clicked = st.button(
                "📡  TRANSMIT SIGNAL",
                use_container_width=True,
                type="primary",
            )

            st.markdown(
                '<div style="text-align:center;color:#334455;margin:10px 0;'
                'font-size:0.85em;">— or upload real IQ data —</div>',
                unsafe_allow_html=True,
            )
            uploaded = st.file_uploader(
                "Upload IQ samples",
                type=["npy", "bin", "sigmf"],
                help="Upload raw IQ samples as float32 numpy array. "
                     "Accepted shapes: (N,) complex64 or (N, 2) float32 "
                     "where columns are [I, Q].",
            )

        # ── RIGHT: results ────────────────────────────────────────────────────
        with col_out:
            # Determine signal source
            sig_complex = None

            if uploaded is not None:
                try:
                    raw = np.load(io.BytesIO(uploaded.read()), allow_pickle=False)
                    if raw.dtype in (np.complex64, np.complex128):
                        sig_complex = raw.astype(np.complex64).flatten()
                    elif raw.ndim == 2 and raw.shape[1] == 2:
                        sig_complex = (raw[:, 0] + 1j * raw[:, 1]).astype(np.complex64)
                    else:
                        sig_complex = raw.flatten().astype(np.complex64)
                    sig_complex = add_awgn(_norm(sig_complex[:INPUT_LEN]), snr_db)
                    st.success(f"Loaded uploaded file — {len(sig_complex)} samples")
                except Exception as e:
                    st.error(f"Could not load file: {e}")

            elif transmit_clicked:
                sig_complex = add_awgn(generate_signal(modulation), snr_db)

            # ── display if we have a signal ───────────────────────────────────
            if sig_complex is not None:
                x_tensor = to_tensor(sig_complex[:INPUT_LEN])

                # inference
                t0      = time.perf_counter()
                result  = run_specter(specter, detector, x_tensor)
                lat_ms  = (time.perf_counter() - t0) * 1000

                threat  = ThreatScorer().score(
                    classification_confidence=result["confidence"],
                    snr_db=snr_db,
                    is_unknown=result["is_unknown"],
                    burst_duration_ms=float(burst_ms),
                )

                # ── 1. Signal visualization ───────────────────────────────────
                st.markdown(
                    '<div style="color:#8899cc;font-size:0.75em;letter-spacing:2px;'
                    'text-transform:uppercase;margin-bottom:4px;">'
                    '① What the signal looks like</div>',
                    unsafe_allow_html=True,
                )
                vc1, vc2 = st.columns(2)
                with vc1:
                    st.plotly_chart(
                        plot_constellation(sig_complex),
                        use_container_width=True,
                    )
                with vc2:
                    st.plotly_chart(
                        plot_psd(sig_complex),
                        use_container_width=True,
                    )

                # ── 2. Classification ─────────────────────────────────────────
                st.markdown(
                    '<div style="color:#8899cc;font-size:0.75em;letter-spacing:2px;'
                    'text-transform:uppercase;margin:12px 0 6px 0;">'
                    '② What SPECTER thinks it is</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    result_banner(
                        result["class_name"],
                        result["confidence"],
                        result["is_unknown"],
                    ),
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    plot_top5(result["probs"], result["is_unknown"]),
                    use_container_width=True,
                )

                # ── 3. Threat assessment ──────────────────────────────────────
                st.markdown(
                    '<div style="color:#8899cc;font-size:0.75em;letter-spacing:2px;'
                    'text-transform:uppercase;margin:12px 0 6px 0;">'
                    '③ How dangerous is this signal</div>',
                    unsafe_allow_html=True,
                )
                tc1, tc2 = st.columns([1, 1])
                with tc1:
                    st.markdown(
                        threat_badge_html(threat.level, threat.score),
                        unsafe_allow_html=True,
                    )
                with tc2:
                    st.plotly_chart(
                        make_gauge(threat.score),
                        use_container_width=True,
                    )

                st.markdown(
                    '<div style="color:#8899cc;font-size:0.8em;'
                    'margin:8px 0 4px 0;">Score breakdown</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    breakdown_html(threat.breakdown),
                    unsafe_allow_html=True,
                )

                # ── 4. Pipeline trace ─────────────────────────────────────────
                st.markdown(
                    '<div style="color:#8899cc;font-size:0.75em;letter-spacing:2px;'
                    'text-transform:uppercase;margin:18px 0 6px 0;">'
                    '④ How SPECTER processed it</div>',
                    unsafe_allow_html=True,
                )
                unk_str = "UNKNOWN" if result["is_unknown"] else result["class_name"]
                steps = [
                    ("✅", f"Signal received — SNR: {snr_db:+d} dB"),
                    ("✅", f"Preprocessed — {INPUT_LEN} IQ samples, channels-first"),
                    ("✅", f"Classified — {unk_str} (confidence: {result['confidence']:.3f})"),
                    ("✅" if not result["is_unknown"] else "⚠️",
                     f"Open-set check — {'accepted' if not result['is_unknown'] else 'REJECTED — energy score exceeded threshold'}"),
                    ("✅", f"Threat scored — {threat.level} ({threat.score:.1f}/100)"),
                    ("✅", f"Alert generated — latency: {lat_ms:.1f} ms"),
                ]
                st.markdown(
                    pipeline_trace_html(steps),
                    unsafe_allow_html=True,
                )

            else:
                st.markdown("""
                <div style="background:#111827;border:1px dashed #2a3860;
                            border-radius:14px;padding:60px;text-align:center;
                            color:#334455;margin-top:20px;">
                  <div style="font-size:3em;margin-bottom:16px;">📡</div>
                  <div style="font-size:1.1em;color:#445566;">
                    Configure a signal on the left and press<br>
                    <strong style="color:#4fa3e0;">TRANSMIT SIGNAL</strong>
                    to see SPECTER respond.
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — BENCHMARK RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("## SPECTER vs The State of the Art")
        st.caption(
            "Direct comparison against the DeepSig VT-CNN2 baseline. "
            "All results generated by `python evaluate.py` on RadioML 2018.01A."
        )
        st.markdown("---")

        # Metric cards — load from CSV if available
        csv_path = os.path.join(_RESULTS, "accuracy_vs_snr.csv")
        b_overall = b_low = s_overall = s_low = None

        if os.path.isfile(csv_path):
            try:
                import csv as _csv
                rows = []
                with open(csv_path) as f:
                    for r in _csv.DictReader(f):
                        rows.append(r)
                snrs_all = [int(r["snr_db"]) for r in rows]
                b_accs   = {int(r["snr_db"]): float(r["baseline_acc"]) for r in rows if r["baseline_acc"]}
                s_accs   = {int(r["snr_db"]): float(r["specter_acc"])  for r in rows if r["specter_acc"]}
                low_keys = [s for s in snrs_all if -20 <= s <= -6]
                if b_accs:
                    b_overall = sum(b_accs.values()) / len(b_accs)
                    b_low     = sum(b_accs[k] for k in low_keys if k in b_accs) / len(low_keys) if low_keys else None
                if s_accs:
                    s_overall = sum(s_accs.values()) / len(s_accs)
                    s_low     = sum(s_accs[k] for k in low_keys if k in s_accs) / len(low_keys) if low_keys else None
            except Exception:
                pass

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.markdown(
                metric_card(
                    "Overall Accuracy",
                    f"{s_overall*100:.1f}%" if s_overall else "TBD",
                    "SPECTER — all SNR levels",
                ),
                unsafe_allow_html=True,
            )
        with mc2:
            st.markdown(
                metric_card(
                    "Low-SNR Accuracy",
                    f"{s_low*100:.1f}%" if s_low else "TBD",
                    "−20 to −6 dB  (covert range)",
                ),
                unsafe_allow_html=True,
            )
        with mc3:
            st.markdown(
                metric_card(
                    "Baseline Low-SNR",
                    f"{b_low*100:.1f}%" if b_low else "TBD",
                    "VT-CNN2 in the same range",
                ),
                unsafe_allow_html=True,
            )
        with mc4:
            gap = (s_low - b_low) if (s_low and b_low) else None
            st.markdown(
                metric_card(
                    "Low-SNR Gain",
                    f"+{gap*100:.1f}pp" if gap else "TBD",
                    "SPECTER over baseline",
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts
        st.markdown("#### Per-SNR Accuracy Curve")
        image_card(
            os.path.join(_RESULTS, "accuracy_vs_snr.png"),
            "SPECTER outperforms the baseline precisely where it matters — "
            "below −6 dB where covert transmissions operate.",
            "accuracy_vs_snr.png not yet generated",
        )

        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown("#### SPECTER Confusion Matrix (Low SNR)")
            image_card(
                os.path.join(_RESULTS, "confusion_specter.png"),
                "Confusion matrix at low SNR (−20 to −6 dB) — "
                "what SPECTER gets right and wrong.",
                "confusion_specter.png not yet generated",
            )
        with bc2:
            st.markdown("#### Open-Set AUROC")
            image_card(
                os.path.join(_RESULTS, "open_set_auroc.png"),
                "SPECTER flags unknown signals (FM, GMSK, AM-DSB-SC, OQPSK) "
                "instead of misclassifying them.",
                "open_set_auroc.png not yet generated",
            )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — THREAT MAP
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("## Transmitter Location Estimation")
        st.caption(
            "⚠️  **SIMULATED** — TDOA triangulation turns a detected signal into a located threat. "
            "In real deployment, 3+ synchronized receivers triangulate the transmitter "
            "using signal arrival time differences."
        )
        st.markdown("---")

        mc1, mc2, mc3 = st.columns(3)

        with mc1:
            st.markdown("**Receiver Positions (metres)**")
            rx0x = st.number_input("RX0 East",  value=0.0,    step=50.0, key="rx0x")
            rx0y = st.number_input("RX0 North", value=577.0,  step=50.0, key="rx0y")
            rx1x = st.number_input("RX1 East",  value=-500.0, step=50.0, key="rx1x")
            rx1y = st.number_input("RX1 North", value=-289.0, step=50.0, key="rx1y")
            rx2x = st.number_input("RX2 East",  value=500.0,  step=50.0, key="rx2x")
            rx2y = st.number_input("RX2 North", value=-289.0, step=50.0, key="rx2y")

        with mc2:
            st.markdown("**True Transmitter (metres)**")
            tx_x = st.number_input("Transmitter East",  value=250.0, step=50.0, key="txx")
            tx_y = st.number_input("Transmitter North", value=300.0, step=50.0, key="txy")
            st.markdown("<br>", unsafe_allow_html=True)
            noise_m = st.slider(
                "Location Noise (σ metres)",
                min_value=0, max_value=200, value=50, step=10,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            locate_clicked = st.button(
                "🎯  LOCATE TRANSMITTER",
                use_container_width=True,
                type="primary",
            )

        with mc3:
            st.markdown("**How it works**")
            st.markdown("""
            <div style="background:#111827;border-left:3px solid #4fa3e0;
                        padding:14px 16px;border-radius:0 8px 8px 0;
                        color:#8899aa;font-size:0.87em;line-height:1.7;">
              Each receiver detects the signal at a slightly different time
              proportional to its distance from the transmitter.<br><br>
              The time difference between any two receivers constrains the
              transmitter to lie on a <strong style="color:#4fa3e0;">hyperbola</strong>.<br><br>
              With ≥ 3 receivers we get ≥ 2 independent hyperbolas.
              Their intersection is solved with a
              <strong style="color:#4fa3e0;">least-squares</strong> solver.<br><br>
              The confidence circle shows the 1-σ position uncertainty.
            </div>
            """, unsafe_allow_html=True)

        if locate_clicked:
            receivers = [(rx0x, rx0y), (rx1x, rx1y), (rx2x, rx2y)]
            true_pos  = (tx_x, tx_y)

            with st.spinner("Running TDOA simulation..."):
                tdoa = TDOASimulator(receiver_positions=receivers)
                sim  = tdoa.run_simulation(
                    true_transmitter_pos=true_pos,
                    noise_std_meters=float(noise_m),
                    visualize=False,
                )

            est_pos = sim["estimated_pos"]
            conf_r  = sim["confidence_radius"]
            err_m   = sim["position_error_m"]

            # Result cards
            rc1, rc2, rc3, rc4 = st.columns(4)
            with rc1:
                st.markdown(
                    metric_card("Estimated Position",
                                f"({est_pos[0]:.0f}, {est_pos[1]:.0f})",
                                "East, North (metres)"),
                    unsafe_allow_html=True,
                )
            with rc2:
                st.markdown(
                    metric_card("True Position",
                                f"({true_pos[0]:.0f}, {true_pos[1]:.0f})",
                                "East, North (metres)"),
                    unsafe_allow_html=True,
                )
            with rc3:
                st.markdown(
                    metric_card("Position Error",
                                f"{err_m:.1f} m",
                                "Euclidean distance from truth"),
                    unsafe_allow_html=True,
                )
            with rc4:
                st.markdown(
                    metric_card("Confidence Radius",
                                f"{conf_r:.1f} m",
                                "1-σ uncertainty circle"),
                    unsafe_allow_html=True,
                )

            fig = plot_tdoa(receivers, true_pos, est_pos, conf_r, float(noise_m))
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        else:
            st.markdown("""
            <div style="background:#111827;border:1px dashed #2a3860;
                        border-radius:12px;padding:60px;text-align:center;
                        color:#334455;margin-top:16px;">
              <div style="font-size:2.5em;margin-bottom:12px;">🎯</div>
              <div style="color:#445566;">
                Configure receivers and press
                <strong style="color:#4fa3e0;">LOCATE TRANSMITTER</strong>
                to run the TDOA simulation.
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — HOW SPECTER WORKS
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("## The SPECTER Pipeline")
        st.caption(
            "For judges — a complete explanation of the architecture, "
            "design decisions, and the gap SPECTER fills."
        )
        st.markdown("---")

        # ── pipeline diagram ──────────────────────────────────────────────────
        st.markdown("#### Signal-to-Alert Pipeline")
        stages = [
            ("RAW IQ",       "Raw antenna samples\nI/Q float32"),
            ("PREPROCESS",   "Normalise power\nChannels-first\n(2 × 1024)"),
            ("CLASSIFY",     "SpecterCNN\nResidual CNN\n24 mod. classes"),
            ("OPEN SET",     "Energy Score\n−log Σ exp(fₖ)\nReject unknowns"),
            ("THREAT SCORE", "Weighted formula\nSNR + confidence\n+ burst + unknown"),
            ("LOCATE",       "TDOA solver\nLeast-squares\n3-receiver array"),
            ("ALERT",        "Threat level\nPosition estimate\nOperator report"),
        ]
        cols = st.columns(len(stages) * 2 - 1)
        for i, (name, detail) in enumerate(stages):
            with cols[i * 2]:
                with st.expander(name, expanded=True):
                    st.markdown(
                        f'<div style="font-size:0.78em;color:#8899aa;'
                        f'white-space:pre-line;">{detail}</div>',
                        unsafe_allow_html=True,
                    )
            if i < len(stages) - 1:
                with cols[i * 2 + 1]:
                    st.markdown(
                        '<div class="pipe-arrow">→</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── innovation cards ──────────────────────────────────────────────────
        st.markdown("#### Three Core Innovations")
        ic1, ic2, ic3 = st.columns(3)

        with ic1:
            st.markdown("""
            <div class="inno-card">
              <h3>🔍 Low-SNR Classifier</h3>
              <p>
                A deep 4-stage residual CNN trained on all 24 RadioML 2018.01A
                modulations with <strong>SNR-aware loss weighting</strong> — samples
                in the −20 to −6 dB covert range receive 3× gradient emphasis.<br><br>
                Residual skip connections keep gradients alive across 8 ResBlocks.
                Batch normalisation stabilises training at extreme noise levels.
                Global average pooling removes spatial bias.<br><br>
                <em>Standard systems fail here. SPECTER was designed for it.</em>
              </p>
            </div>
            """, unsafe_allow_html=True)

        with ic2:
            st.markdown("""
            <div class="inno-card">
              <h3>⚠️ Open-Set Detector</h3>
              <p>
                Every closed-set classifier will confidently misclassify a waveform
                it has never seen. SPECTER uses the
                <strong>Energy Score</strong> (Liu et al., NeurIPS 2020):<br><br>
                <code>E(x) = −log Σₖ exp(fₖ(x))</code><br><br>
                Low energy → known class. High energy → reject as unknown.<br><br>
                Threshold fitted at the 95th percentile of known-class energies.
                Evaluated against the softmax MSP baseline via AUROC.
              </p>
            </div>
            """, unsafe_allow_html=True)

        with ic3:
            st.markdown("""
            <div class="inno-card">
              <h3>🎯 Threat Scorer</h3>
              <p>
                Raw classification outputs are not actionable intelligence.
                SPECTER fuses four signals into a single 0–100 score:<br><br>
                • <strong>Uncertainty</strong> (25%) — 1 − confidence<br>
                • <strong>SNR danger</strong> (30%) — −20 dB maps to 1.0<br>
                • <strong>Unknown penalty</strong> (35%) — binary rejection flag<br>
                • <strong>Burst suspicion</strong> (10%) — short bursts score higher<br><br>
                Thresholds: LOW &lt; 40 · MEDIUM 40–69 · CRITICAL ≥ 70
              </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── before / after table ──────────────────────────────────────────────
        st.markdown("#### The Gap SPECTER Fills")
        st.markdown("""
        <table style="width:100%;border-collapse:collapse;font-size:0.9em;">
          <thead>
            <tr style="background:#1a2040;color:#8899cc;">
              <th style="padding:12px 16px;text-align:left;border-bottom:2px solid #2a3860;">
                Scenario
              </th>
              <th style="padding:12px 16px;text-align:left;border-bottom:2px solid #2a3860;">
                ❌ Without SPECTER
              </th>
              <th style="padding:12px 16px;text-align:left;border-bottom:2px solid #2a3860;">
                ✅ With SPECTER
              </th>
            </tr>
          </thead>
          <tbody>
            <tr style="border-bottom:1px solid #1e2840;">
              <td style="padding:11px 16px;color:#c0ccee;">Covert −18 dB signal</td>
              <td style="padding:11px 16px;color:#ff4444;">Looks like noise → ignored</td>
              <td style="padding:11px 16px;color:#00cc44;">Detected → classified → CRITICAL alert</td>
            </tr>
            <tr style="border-bottom:1px solid #1e2840;background:#0d1320;">
              <td style="padding:11px 16px;color:#c0ccee;">Unknown waveform</td>
              <td style="padding:11px 16px;color:#ff4444;">Misclassified as known → missed</td>
              <td style="padding:11px 16px;color:#00cc44;">Rejected → flagged UNKNOWN → escalated</td>
            </tr>
            <tr style="border-bottom:1px solid #1e2840;">
              <td style="padding:11px 16px;color:#c0ccee;">Short-burst transmission</td>
              <td style="padding:11px 16px;color:#ff4444;">Insufficient context → no action</td>
              <td style="padding:11px 16px;color:#00cc44;">Burst pattern raises suspicion score</td>
            </tr>
            <tr style="background:#0d1320;">
              <td style="padding:11px 16px;color:#c0ccee;">Detected threat</td>
              <td style="padding:11px 16px;color:#ff4444;">Classification only → no location</td>
              <td style="padding:11px 16px;color:#00cc44;">TDOA triangulation → estimated position</td>
            </tr>
          </tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── references ────────────────────────────────────────────────────────
        with st.expander("References & Dataset"):
            st.markdown("""
            - O'Shea, T., Corgan, J., Clancy, T.C. (2016). *Convolutional Radio Modulation Recognition Networks.* arXiv:1602.04105
            - Liu, W., Wang, X., Owens, J., Li, Y. (2020). *Energy-based Out-of-distribution Detection.* NeurIPS 2020.
            - Hendrycks, D., Gimpel, K. (2017). *A Baseline for Detecting Misclassified and Out-of-Distribution Examples.* ICLR 2017.
            - **Dataset:** RadioML 2018.01A — DeepSig Inc.
              [Kaggle](https://www.kaggle.com/datasets/pinxau1000/radioml2018) · 4,096,000 samples · 24 classes · −20 to +30 dB
            """)

        st.markdown("""
        <div style="text-align:center;color:#223344;font-size:0.8em;
                    margin-top:30px;padding-top:20px;border-top:1px solid #1a2040;">
          SPECTER — Built for competition. Defensive application only.<br>
          <em>Seeing what every other system calls nothing.</em>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
