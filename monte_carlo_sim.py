# app.py
# Streamlit app: Binomial probability and Monte Carlo estimation
# UI and comments in English
import math
import numpy as np
import streamlit as st

st.set_page_config(page_title="Coin Toss: P(HEADS = K in M tosses)", layout="centered")

# ---- Header ----
st.title("Coin Toss: Probability of Getting K HEADS in M Tosses")
st.caption("Exact value from the Binomial distribution + Monte Carlo estimation")

# ---- Sidebar with inputs ----
st.sidebar.header("Parameters")

# p in [0,1]
p = st.sidebar.slider("Probability of HEADS (p)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# M in [1,100]
M = st.sidebar.number_input("Number of tosses (M)", min_value=1, max_value=100, value=50, step=1)

# K in [0, M]
K = st.sidebar.number_input("Target number of HEADS (K)", min_value=0, max_value=int(M), value=min(20, int(M)), step=1)

# N in [10, 100000]
N = st.sidebar.number_input("Monte Carlo trials (N)", min_value=10, max_value=100_000, value=10_000, step=10)

# Optional random seed for reproducibility
use_seed = st.sidebar.checkbox("Set random seed", value=True)
seed_val = st.sidebar.number_input("Seed", min_value=0, max_value=1_000_000, value=42, step=1, disabled=not use_seed)

# Validate inputs
if K > M:
    st.error("Error: K cannot be greater than M.")
    st.stop()
if not (0.0 <= p <= 1.0):
    st.error("Error: p must be in [0, 1].")
    st.stop()

# ---- Exact probability (Binomial PMF) ----
# PMF: C(M, K) * p^K * (1 - p)^(M-K)
try:
    comb_mk = math.comb(int(M), int(K))
    exact_prob = comb_mk * (p ** int(K)) * ((1 - p) ** (int(M) - int(K)))
except ValueError:
    st.error("Computation error for the given parameters. Please check M and K.")
    st.stop()

st.subheader("Exact Probability")
st.write(
    rf"$P(X=K) = \binom{{M}}{{K}} p^K (1-p)^{{M-K}}$  "
    rf"= $\binom{{{int(M)}}}{{{int(K)}}} \cdot {p:.2f}^{int(K)} \cdot (1-{p:.2f})^{{{int(M)-int(K)}}}$"
)
st.metric("P(X = K)", f"{exact_prob:.6g}")

# ---- Monte Carlo estimation ----
st.subheader("Monte Carlo Estimation")

col1, col2 = st.columns(2)
with col1:
    run_sim = st.button("Run simulation")
with col2:
    show_hist = st.checkbox("Show histogram", value=False)

if run_sim:
    # Set random seed if requested
    if use_seed:
        np.random.seed(int(seed_val))

    # Efficient generation: counts ~ Binomial(M, p), size=N
    # Then estimate P(X=K) as mean(counts == K)
    with st.spinner("Simulating..."):
        counts = np.random.binomial(n=int(M), p=float(p), size=int(N))
        hits = (counts == int(K))
        mc_estimate = hits.mean()
        # Standard error of a Bernoulli with mean mc_estimate: sqrt(p_hat*(1-p_hat)/N)
        mc_se = float(np.sqrt(mc_estimate * (1 - mc_estimate) / int(N)))

    st.write(
        f"Estimated probability (share of exact-K outcomes): **{mc_estimate:.6g}** "
        f"± **{1.96*mc_se:.3g}** (approx. 95% CI)"
    )

    # Compare with exact probability
    abs_err = abs(mc_estimate - exact_prob)
    st.write(f"Absolute error vs exact value: **{abs_err:.6g}**")

    # Optional histogram
    if show_hist:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.hist(counts, bins=range(0, int(M)+2), edgecolor="black", align="left", density=True)
        ax.set_xlabel("Number of HEADS in M tosses")
        ax.set_ylabel("Relative frequency")
        ax.set_title("Histogram of Monte Carlo outcomes")
        # Mark K
        ax.axvline(int(K), linestyle="--")
        st.pyplot(fig)
else:
    st.info("Set the parameters on the left and click **Run simulation**.")

# ---- Footer / didactics ----
with st.expander("What is happening here?"):
    st.markdown(
        """
**Binomial distribution.** For `M` coin tosses with HEADS probability `p`, the random variable `X` — the number of heads —
follows a Binomial distribution:
\\[
P(X=K) = \\binom{M}{K} p^K (1-p)^{M-K}.
\\]

**Monte Carlo.** We generate `N` independent repetitions of the experiment (each with `M` tosses) and compute the share of runs
that produced exactly `K` heads. This is a frequency-based estimator of the desired probability. The standard error of the estimator
(for a Bernoulli indicator) is approximately
\\(\\sqrt{\\hat p (1-\\hat p) / N}\\), which yields an approximate 95% CI of \\(\\hat p \\pm 1.96\\,SE\\).
"""
    )
