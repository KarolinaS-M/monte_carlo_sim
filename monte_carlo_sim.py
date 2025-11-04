# app.py
# Streamlit app: Binomial probability and Monte Carlo estimation
# UI in Polish; code comments in English (for students/teaching)
import math
import numpy as np
import streamlit as st

st.set_page_config(page_title="Rzut monetą: P(HEADS = K w M rzutach)", layout="centered")

# ---- Header ----
st.title("Rzut monetą: prawdopodobieństwo uzyskania K ORŁÓW w M rzutach")
st.caption("Dokładna wartość z rozkładu dwumianowego + estymacja Monte Carlo")

# ---- Sidebar with inputs ----
st.sidebar.header("Parametry")

# p in [0,1]
p = st.sidebar.slider("Prawdopodobieństwo ORŁA p", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# M in [1,100]
M = st.sidebar.number_input("Liczba rzutów M", min_value=1, max_value=100, value=50, step=1)

# K in [0, M]
K = st.sidebar.number_input("Docelowa liczba ORŁÓW K", min_value=0, max_value=int(M), value=min(20, int(M)), step=1)

# N in [10, 100000]
N = st.sidebar.number_input("Liczba prób Monte Carlo N", min_value=10, max_value=100_000, value=10_000, step=10)

# Optional random seed for reproducibility
use_seed = st.sidebar.checkbox("Ustal ziarno generatora (seed)", value=True)
seed_val = st.sidebar.number_input("Seed", min_value=0, max_value=1_000_000, value=42, step=1, disabled=not use_seed)

# Validate inputs
if K > M:
    st.error("Błąd: K nie może być większe niż M.")
    st.stop()
if not (0.0 <= p <= 1.0):
    st.error("Błąd: p musi należeć do przedziału [0,1].")
    st.stop()

# ---- Exact probability (Binomial PMF) ----
# Use math.comb to avoid SciPy dependency
# PMF: C(M,K) * p^K * (1-p)^(M-K)
try:
    comb_mk = math.comb(int(M), int(K))
    exact_prob = comb_mk * (p ** int(K)) * ((1 - p) ** (int(M) - int(K)))
except ValueError:
    st.error("Błąd obliczeń dla podanych parametrów. Sprawdź wartości M i K.")
    st.stop()

st.subheader("Dokładne prawdopodobieństwo")
st.write(
    rf"$P(X=K) = \binom{{M}}{{K}} p^K (1-p)^{{M-K}}$  "
    rf"= $\binom{{{int(M)}}}{{{int(K)}}} \cdot {p:.2f}^{int(K)} \cdot (1-{p:.2f})^{{{int(M)-int(K)}}}$"
)
st.metric("P(X = K)", f"{exact_prob:.6g}")

# ---- Monte Carlo estimation ----
st.subheader("Estymacja Monte Carlo")

col1, col2 = st.columns(2)
with col1:
    run_sim = st.button("Uruchom symulację")
with col2:
    show_hist = st.checkbox("Pokaż histogram wyników", value=False)

if run_sim:
    # Set random seed if requested
    if use_seed:
        np.random.seed(int(seed_val))

    # Efficient generation: counts ~ Binomial(M, p), size=N
    # Then estimate P(X=K) as mean(counts == K)
    with st.spinner("Symuluję..."):
        counts = np.random.binomial(n=int(M), p=float(p), size=int(N))
        hits = (counts == int(K))
        mc_estimate = hits.mean()
        # Standard error of a Bernoulli(mean = mc_estimate): sqrt(p_hat*(1-p_hat)/N)
        mc_se = float(np.sqrt(mc_estimate * (1 - mc_estimate) / int(N)))

    st.write(
        f"Szacowane prawdopodobieństwo (udział trafień K): **{mc_estimate:.6g}** "
        f"± **{1.96*mc_se:.3g}** (95% CI w przybliżeniu)"
    )

    # Compare with exact probability
    abs_err = abs(mc_estimate - exact_prob)
    st.write(f"Bezwzględny błąd względem wartości dokładnej: **{abs_err:.6g}**")

    # Optional histogram
    if show_hist:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.hist(counts, bins=range(0, int(M)+2), edgecolor="black", align="left", density=True)
        ax.set_xlabel("Liczba ORŁÓW w M rzutach")
        ax.set_ylabel("Częstość względna")
        ax.set_title("Histogram wyników w symulacji Monte Carlo")
        # Mark K
        ax.axvline(int(K), linestyle="--")
        st.pyplot(fig)
else:
    st.info("Ustaw parametry po lewej i kliknij **Uruchom symulację**.")

# ---- Footer / didactics ----
with st.expander("Co tu się dzieje? (wyjaśnienie)"):
    st.markdown(
        """
**Rozkład dwumianowy.** Dla `M` rzutów i prawdopodobieństwa orła `p`, zmienna `X` – liczba orłów – ma rozkład dwumianowy:
\\[
P(X=K) = \\binom{M}{K} p^K (1-p)^{M-K}.
\\]

**Monte Carlo.** Generujemy `N` niezależnych powtórzeń eksperymentu (każde to `M` rzutów) i liczymy udział prób,
w których wyszło dokładnie `K` orłów. To jest estymator częstościowy prawdopodobieństwa. Błąd standardowy
oszacowania (dla wskaźnika trafienia) to w przybliżeniu
\\(\\sqrt{\\hat p (1-\\hat p) / N}\\), co pozwala podać 95% CI jako \\(\\hat p \\pm 1.96\\,SE\\).
"""
    )
