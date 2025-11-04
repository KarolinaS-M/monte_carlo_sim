# app_en.py
# Streamlit app: Binomial probability and Monte Carlo estimation
# UI and comments in English
import math
import numpy as np
import streamlit as st

st.set_page_config(page_title="Coin Toss: P(HEADS = K in M tosses)", layout="centered")

# ---- Header ----
st.title("Coin Toss: Probability of Getting K HEADS in M Tosses")
st.caption("Exact value from the Binomial distribution + Monte Carlo estimation (now supports very large N via chunking)")

# ---- Sidebar with inputs ----
st.sidebar.header("Parameters")

# p in [0,1]
p = st.sidebar.slider("Probability of HEADS (p)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# M in [1,100]
M = st.sidebar.number_input("Number of tosses (M)", min_value=1, max_value=100, value=50, step=1)

# K in [0, M]
K = st.sidebar.number_input("Target number of HEADS (K)", min_value=0, max_value=int(M), value=min(20, int(M)), step=1)

# N in [10, 100,000,000]
N = st.sidebar.number_input("Monte Carlo trials (N)", min_value=10, max_value=100_000_000, value=10_000, step=10)

# Optional random seed for reproducibility
use_seed = st.sidebar.checkbox("Set random seed", value=True)
seed_val = st.sidebar.number_input("Seed", min_value=0, max_value=1_000_000, value=42, step=1, disabled=not use_seed)

# Advanced: chunk size for Monte Carlo (auto by default)
st.sidebar.subheader("Advanced")
auto_chunk = st.sidebar.checkbox("Auto chunk size", value=True)
chunk_size = st.sidebar.number_input(
    "Chunk size (only if Auto is off)",
    min_value=10_000, max_value=10_000_000, value=1_000_000, step=10_000,
    disabled=auto_chunk
)

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

col1, col2, col3 = st.columns(3)
with col1:
    run_sim = st.button("Run simulation")
with col2:
    show_hist = st.checkbox("Show histogram", value=False)
with col3:
    sample_for_hist = st.number_input("Histogram sample size (if showing)", min_value=10_000, max_value=2_000_000, value=200_000, step=10_000, disabled=not show_hist)

if run_sim:
    # Set random seed if requested
    if use_seed:
        np.random.seed(int(seed_val))

    # Determine chunk size automatically to control memory usage
    if auto_chunk:
        # Heuristic: aim for ~8-16 MB arrays; each int64 ~8 bytes; float64 ~8 bytes
        # np.random.binomial returns int64 by default; choose up to ~2,000,000 per chunk (≈16 MB)
        chunk_size_eff = min(int(N), 2_000_000)
    else:
        chunk_size_eff = int(chunk_size)

    chunk_size_eff = max(1, chunk_size_eff)

    progress = st.progress(0)
    status = st.empty()

    total_trials = 0
    total_hits = 0

    # For histogram, collect sample without storing all counts
    hist_counts_sample = None
    if show_hist:
        # We'll collect up to sample_for_hist counts, using reservoir-like sampling by concatenating until cap.
        hist_counts_sample = np.empty(0, dtype=np.int32)

    with st.spinner("Simulating in chunks..."):
        remaining = int(N)
        done = 0
        while remaining > 0:
            bs = min(chunk_size_eff, remaining)
            counts = np.random.binomial(n=int(M), p=float(p), size=bs)
            hits = np.count_nonzero(counts == int(K))
            total_hits += int(hits)
            total_trials += int(bs)

            # Accumulate histogram sample if requested
            if show_hist and hist_counts_sample is not None and hist_counts_sample.size < int(sample_for_hist):
                space_left = int(sample_for_hist) - hist_counts_sample.size
                take = min(space_left, bs)
                # take first 'take' items (enough for a good sample)
                hist_counts_sample = np.concatenate([hist_counts_sample, counts[:take].astype(np.int32, copy=False)], axis=0)

            done += bs
            progress.progress(min(1.0, done / int(N)))
            status.text(f"Completed {done:,} / {int(N):,} trials")

    mc_estimate = total_hits / total_trials if total_trials > 0 else float('nan')
    # Standard error of a Bernoulli with mean mc_estimate: sqrt(p_hat*(1-p_hat)/N)
    mc_se = float(np.sqrt(mc_estimate * (1 - mc_estimate) / max(1, total_trials)))

    st.write(
        f"Estimated probability (share of exact-K outcomes): **{mc_estimate:.6g}** "
        f"± **{1.96*mc_se:.3g}** (approx. 95% CI) — from **{total_trials:,}** trials."
    )

    # Compare with exact probability
    abs_err = abs(mc_estimate - exact_prob)
    st.write(f"Absolute error vs exact value: **{abs_err:.6g}**")

    # Optional histogram
    if show_hist and hist_counts_sample is not None and hist_counts_sample.size > 0:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        # Build bins 0..M inclusive
        ax.hist(hist_counts_sample, bins=range(0, int(M)+2), edgecolor="black", align="left", density=True)
        ax.set_xlabel("Number of HEADS in M tosses")
        ax.set_ylabel("Relative frequency (sample)")
        ax.set_title(f"Histogram of Monte Carlo outcomes (sample size {hist_counts_sample.size:,})")
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

**Performance note.** Very large `N` (e.g., tens of millions) are computed in chunks to avoid memory overflow. This makes the simulation feasible on typical machines while keeping the estimator unbiased.
"""
    )
