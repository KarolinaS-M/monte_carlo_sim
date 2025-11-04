# app_en.py
# Streamlit app: Binomial probability and Monte Carlo estimation
# Always-random version: no seed controls; fresh RNG each run
import math
import numpy as np
import streamlit as st

st.set_page_config(page_title="Coin Toss: P(HEADS = K in M tosses)", layout="centered")

# ---- Header ----
st.title("Coin Toss: Probability of Getting K HEADS in M Tosses")
st.caption("Exact Binomial probability + Monte Carlo estimation (always random results)")

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

# Advanced (rarely needed): chunk size and threshold for switching to chunked mode
with st.sidebar.expander("Advanced"):
    auto_chunk = st.checkbox("Auto chunk size", value=True)
    chunk_size = st.number_input("Chunk size (if Auto off)", min_value=100_000, max_value=10_000_000, value=1_000_000, step=100_000, disabled=auto_chunk)
    # Switch threshold: for N <= switch_threshold, use single vectorized call (fastest)
    switch_threshold = st.number_input("Switch to chunked mode above N =", min_value=100_000, max_value=10_000_000, value=2_000_000, step=100_000)

# Validate inputs
if K > M:
    st.error("Error: K cannot be greater than M.")
    st.stop()
if not (0.0 <= p <= 1.0):
    st.error("Error: p must be in [0, 1].")
    st.stop()

# ---- Exact probability (Binomial PMF) ----
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
    # Fresh RNG each run -> always random results (also across students/machines)
    rng = np.random.default_rng()

    # -------- FAST PATH (vectorized one-shot) --------
    if int(N) <= int(switch_threshold):
        with st.spinner("Simulating..."):
            counts = rng.binomial(n=int(M), p=float(p), size=int(N))
        mc_estimate = float(np.mean(counts == int(K)))
        mc_se = float(np.sqrt(mc_estimate * (1 - mc_estimate) / int(N)))
        st.write(
            f"Estimated probability: **{mc_estimate:.6g}** ± **{1.96*mc_se:.3g}** (approx. 95% CI)"
        )
        abs_err = abs(mc_estimate - exact_prob)
        st.write(f"Absolute error vs exact value: **{abs_err:.6g}**")

        if show_hist:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.hist(counts, bins=range(0, int(M)+2), edgecolor="black", align="left", density=True)
            ax.set_xlabel("Number of HEADS in M tosses")
            ax.set_ylabel("Relative frequency")
            ax.set_title("Histogram of Monte Carlo outcomes")
            ax.axvline(int(K), linestyle="--")
            st.pyplot(fig)

    # -------- CHUNKED MODE (for very large N) --------
    else:
        # Determine chunk size
        if auto_chunk:
            chunk_size_eff = min(int(N), 2_000_000)
        else:
            chunk_size_eff = int(chunk_size)
        chunk_size_eff = max(1, chunk_size_eff)

        progress = st.progress(0)
        status = st.empty()

        total_trials = 0
        total_hits = 0

        # If histogram is requested, collect a bounded sample only
        hist_sample_target = 200_000 if show_hist else 0
        hist_counts_sample = np.empty(0, dtype=np.int16) if hist_sample_target > 0 else None

        with st.spinner("Simulating in chunks..."):
            remaining = int(N)
            done = 0
            updates = 0
            update_every = max(1, (int(N) // chunk_size_eff) // 50)  # ~50 UI updates max

            while remaining > 0:
                bs = min(chunk_size_eff, remaining)
                counts = rng.binomial(n=int(M), p=float(p), size=bs)
                total_hits += int(np.count_nonzero(counts == int(K)))
                total_trials += int(bs)

                # Optional histogram sampling
                if hist_counts_sample is not None and hist_counts_sample.size < hist_sample_target:
                    space_left = hist_sample_target - hist_counts_sample.size
                    take = min(space_left, bs)
                    hist_counts_sample = np.concatenate([hist_counts_sample, counts[:take].astype(np.int16, copy=False)], axis=0)

                done += bs
                remaining -= bs

                # Throttle UI updates
                updates += 1
                if updates % update_every == 0 or remaining == 0:
                    progress.progress(min(1.0, done / int(N)))
                    status.text(f"Completed {done:,} / {int(N):,} trials")

        mc_estimate = total_hits / total_trials if total_trials > 0 else float('nan')
        mc_se = float(np.sqrt(mc_estimate * (1 - mc_estimate) / max(1, total_trials)))

        st.write(
            f"Estimated probability: **{mc_estimate:.6g}** "
            f"± **{1.96*mc_se:.3g}** (approx. 95% CI) — from **{total_trials:,}** trials."
        )
        abs_err = abs(mc_estimate - exact_prob)
        st.write(f"Absolute error vs exact value: **{abs_err:.6g}**")

        if show_hist and hist_counts_sample is not None and hist_counts_sample.size > 0:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.hist(hist_counts_sample, bins=range(0, int(M)+2), edgecolor="black", align="left", density=True)
            ax.set_xlabel("Number of HEADS in M tosses (sample)")
            ax.set_ylabel("Relative frequency")
            ax.set_title(f"Histogram (sample size {hist_counts_sample.size:,})")
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

**Monte Carlo.** We generate `N` independent repetitions (each with `M` tosses) and compute the share of runs with exactly `K` heads.
The estimator's standard error is approximately \\(\\sqrt{\\hat p (1-\\hat p)/N}\\), so a 95% CI is \\(\\hat p \\pm 1.96\\,SE\\).

**Always random.** Each simulation uses a fresh random-number generator seeded from system entropy, so different students (or repeated runs) obtain different results even with identical settings.
"""
    )
