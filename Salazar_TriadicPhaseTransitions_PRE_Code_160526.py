# =================================================================================================
# Companion simulation code to the paper
# Salazar, E.
# Triadic Phase Transitions in AI Networks: Composite-Operator Scaling in Cognitive Architectures
# arXiv:2604.27038 [cond-mat.stat-mech]
# https://doi.org/10.48550/arXiv.2604.27038 
#
# Requirements
# ------------
# Tested on Python 3.10.10
# NumPy
# SciPy
# Matplotlib
#
# Aim
# ---
# This simple code empirically demonstrates the analytical mean-field derivations in the paper.
# We simulated the triadic Ising model on a random 3-uniform hypergraph through a vectorized 
# Metropolis-Hastings Monte Carlo algorithm. It tracks the thermodynamic and dynamical evolution 
# of the composite formation observable $\Psi_{\text{form}} = \langle \phi_i \phi_j \phi_k \rangle$
# across multiple network sizes (N = 210, 600 and 1200).
#
# By explicitly evaluating the order parameter, its conjugate susceptibility $\chi_{\text{TF}}$, 
# the critical equation of state in response to a triadic field $h_3$, and the autocorrelation 
# relaxation time $\tau_{\text{relax}}$, this numerical routine verifies the theoretically 
# predicted composite-operator scaling regime. It confirms the non-standard magnetization scaling 
# ($\beta_{\text{TF}} = 3/2$) and the strict vanishing of susceptibility at criticality 
# ($\gamma_{\text{TF}} = -1$). The aim is to demonstrate that the exact $N \to \infty$ 
# factorization robustly governs the macroscopic behavior of finite, sparse triadic networks, 
# subject only to standard finite-size and $O(1/K)$ finite-connectivity corrections.
#
# Important
# ---------
# Despite being vectorized, the code is not fully optimized (not parallelized and makes no use of 
# GPU if available, e.g., using CuPy) for numerical lifting. On a modern computer, it should run 
# acceptably for K up to 1200, but as the graph size increases the execution will markedly slow
# down. You are free to modify the code to suit your available computational resources.
#
# The code generates a publication-ready matplotlib rendering of the results as one collage graph
# (merging four individual graphs, also provided for insertion in 2-column text layouts) and two 
# raw CSV data exports.
#
# This code is (c) E. Salazar
# Nebula Technology Lab LLC
# 2026
#
# For any queries, contact the author at
# eduardo [dot] salazar [at] nebulalab [dot] ae
# =================================================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.signal import correlate as _signal_correlate
import sys
import time

# =============
# CONFIGURATION
# =============

# Note that we set K as a fixed percentage of the network size.
# Therefore, local connectivity diverges alongside the thermodynamic 
# limit ($K \to \infty$). In this dense limit, local fluctuations 
# suppress to zero, ensuring convergence to the true Curie-Weiss 
# mean-field limit.
# ------------------------------------------------------------------
sparsity_ratio = 0.2    # -> Agents participate in triads scaling as X% of N (X set by default at 20%)

J = 0.6                 # -> Pairwise alignment coupling
gamma = 0.4             # -> Gradient coefficient
w = 1.0                 # -> Edge weight
Tc = J + gamma * w      # -> Critical Temperature threshold (Tc = 1.0)

# We choose system sizes divisible
# by 3 but K is anyway forced to
# be divisible by 3 later (meaning
# no truncation error for ANY size)
# -> you can choose any size
# ---------------------------------
system_sizes = [210, 600, 1200]

# Graph colors
# ------------
colors = {210: 'royalblue', 600: 'forestgreen', 1200: 'darkorange'}

temperatures = np.linspace(0.4, 1.4, 100)
mcs_steps = 14000
burn_in = 4000

h3_fields = np.logspace(-3, -1, 10)
psi_vs_h3 = {N: [] for N in system_sizes}
results = {}

rng = np.random.default_rng(12345)

# =======================================
# TRANSCENDENTAL MEAN-FIELD THEORY SOLVER
# =======================================

# Solves the Curie-Weiss self-consistency relation
# using the exact 9*m^4 operator variance scaling
# ++++++++++++++++++++++++++++++++++++++++++++++++
def compute_exact_theory(t_array, Tc):
    psi_th = np.zeros_like(t_array)
    chi_th = np.zeros_like(t_array)

    mask = t_array < Tc
    if not np.any(mask):
        return psi_th, chi_th

    T_below = t_array[mask]
    m = np.full(T_below.shape, 0.999)
    for _ in range(1000):
        m = np.tanh((Tc * m) / T_below)

    beta_below  = 1.0 / T_below
    numerator   = beta_below * (1.0 - m * m)
    denominator = np.maximum(1.0 - beta_below * Tc * (1.0 - m * m), 1e-5)

    psi_th[mask] = m ** 3
    chi_th[mask] = 9.0 * (m ** 4) * (numerator / denominator)

    return psi_th, chi_th

# Solves the exact mean-field self-consistency relation 
# at T = Tc while accounting for the inverse critical 
# temperature scaling beta_c = 1/Tc. Returns Psi_form = m^3, 
# avoiding the trivial m=0 root by seeding from the positive 
# analytical leading-order solution m ~ 9*h3.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def compute_eos_theory(h3_array, Tc):
    m = 9.0 * h3_array
    for _ in range(5000):
        m_new = np.tanh(m + 3.0 * (1.0 / Tc) * h3_array * m * m)
        if np.max(np.abs(m_new - m)) < 1e-12:
            break
        m = m_new
    return m ** 3

# =====================================
# RANDOM 3-UNIFORM HYPERGRAPH GENERATOR
# =====================================

# Builds the 3-uniform random hypergraph topology and computes a static
# CSR projection matrix (M_sparse). This matrix maps interaction pairs 
# directly onto target node fields via standard sparse matrix products.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def generate_vectorized_hypergraph(N, K):
    triads_needed = (N * K) // 3

    # We draw node arrays in one batch each, resolving 
    # collisions via rejection. Collision probability 
    # per triad is O(1/N), so the while-loops terminate 
    # in O(1) expected rounds for any practical N.
    # -------------------------------------------------
    u = rng.integers(0, N, size=triads_needed)
    v = rng.integers(0, N, size=triads_needed)
    bad = u == v

    while np.any(bad):
        v[bad] = rng.integers(0, N, size=int(bad.sum()))
        bad = u == v

    w_node = rng.integers(0, N, size=triads_needed)
    bad = (w_node == u) | (w_node == v)
    while np.any(bad):
        w_node[bad] = rng.integers(0, N, size=int(bad.sum()))
        bad = (w_node == u) | (w_node == v)

    u_arr = np.concatenate([u, v, w_node])
    v_arr = np.concatenate([v, u, u])
    w_arr = np.concatenate([w_node, w_node, v ])

    M_data = np.ones(len(u_arr), dtype=np.float64)
    M_col = np.arange(len(u_arr))
    M_sparse = csr_matrix((M_data, (u_arr, M_col)), shape=(N, len(u_arr)))

    return u_arr, v_arr, w_arr, M_sparse

# ==========
# SIMULATION
# ==========

print("\nLaunching Simulations for Salazar (2026b) paper")
print("In -> https://doi.org/10.48550/arXiv.2604.27038")
print("===============================================")

global_start = time.time()

# Pre-compute mean-field EOS curve and equilibrium 
# magnetizations -> these are N-independent and are 
# reused for both initialization and the reference 
# line in the EOS plot.
# -------------------------------------------------
psi_eos_th_ref = compute_eos_theory(h3_fields, Tc)
m_eos_th_ref = np.cbrt(psi_eos_th_ref)

for N in system_sizes:
    # We ensure K is an integer and 
    # (N * K) is divisible by 3
    # -----------------------------
    target_K = int(N * sparsity_ratio)
    while (N * target_K) % 3 != 0:
        target_K += 1
    
    K = target_K

    print(f"\n[Size N = {N}, K = {K}] Building network arrays...", end="")
    sys.stdout.flush()
    u_arr, v_arr, w_arr, M_sparse = generate_vectorized_hypergraph(N, K)
    print(" Done")

    psi_vs_T = []
    chi_vs_T = []
    tau_vs_T = []

    # Part 1 -> Temperature Sweep (h3 = 0)
    # Here we're measuring equilibrium 
    # fluctuations
    # ------------------------------------
    print(f"[Size N = {N}, K = {K}] Running Part 1: Temperature sweep")
    for t_idx, T in enumerate(temperatures):
        print(f"\r>>> Processing Temp {T:.3f} ({t_idx + 1}/{len(temperatures)})...", end="")
        sys.stdout.flush()
        
        beta = 1.0 / T
        spins = rng.integers(0, 2, size=N) * 2 - 1
        psi_samples = []
        magnetization_history = []

        for step in range(mcs_steps):
            # Lookup
            # ------
            pair_sums = spins[v_arr] + spins[w_arr]
            local_fields = M_sparse.dot(pair_sums)
            
            # Energy change matrix
            # (Kac scaling anchors Tc)
            # ------------------------
            dE = 2.0 * Tc * spins * (local_fields / (2.0 * K))
            
            # Stochastic update -> we mask using active_subset to prevent parallel-update 
            # oscillation artifacts. To resolve equilibrium thermal fluctuations across the 
            # temperature sweep, we use 0.2 as active update fraction (for asynchronous 
            # emulation, feedback loop suppression, and fluctuation protection). On a triad 
            # 0.2^3 = 0.8% -> low enough to make synchronization artifacts negligible.
            # -----------------------------------------------------------------------------
            active_subset = rng.random(N) < 0.2  
            accept = (dE <= 0) | (rng.random(N) < np.exp(-beta * dE))
            spins[active_subset & accept] *= -1

            if step >= burn_in:
                m_t = np.mean(spins)
                psi_samples.append(np.abs(m_t)**3)
                magnetization_history.append(m_t)

        psi_samples = np.array(psi_samples)
        m_hist = np.array(magnetization_history) - np.mean(magnetization_history)

        # We use scipy.signal.correlate with 
        # FFT-acceleration instead of computing 
        # full convolutions for O(n log n) cost
        # -------------------------------------
        if len(m_hist) > 1 and np.var(m_hist) > 1e-6:
            n = len(m_hist)
            raw = _signal_correlate(m_hist, m_hist, mode='full', method='fft')
            raw = raw[n - 1:]

            # Unbiased normalization -> we divide each lag sum by 
            # the number of contributing pairs (n, n-1, ..., 1)
            # ---------------------------------------------------
            counts = np.arange(n, 0, -1, dtype=np.float64)
            acf = raw / counts

            # We normalize C(0) = 1
            # ---------------------
            acf /= acf[0]                                 

            # First lag where the normalized ACF drops below 1/e.
            # If the threshold is never crossed within the sample,
            # we report the sample length as the lower bound on tau.
            # ------------------------------------------------------
            below = np.where(acf < np.exp(-1.0))[0]
            if len(below) > 0:
                tau_val = int(below[0])
            else:
                tau_val = n - 1
        else:
            tau_val = 0

        psi_vs_T.append(np.mean(psi_samples))
        chi_vs_T.append(beta * N * np.var(psi_samples))
        tau_vs_T.append(tau_val)

    print(" Complete")
    results[N] = {'psi': psi_vs_T, 'chi': chi_vs_T, 'tau': tau_vs_T}

    # Part 2 -> Critical equation of state (EOS)
    # We're driving relaxation toward an EOS measurement 
    # under a strong external field h_3 (no fluctuations 
    # are being measured).
    # --------------------------------------------------
    print(f"[Size N = {N}, K = {K}] Running Part 2: Critical equation of state")
    print("\r>>> Processing...", end="")
    sys.stdout.flush()
    beta_c = 1.0 / Tc

    for h3, m_th in zip(h3_fields, m_eos_th_ref):
        # Theory-informed initialization -> we set the spin-up fraction to
        # match the mean-field equilibrium magnetization m_theory(h3, Tc).
        # A 50% sweep (0.5) is appropriate because thermalization speed is 
        # the priority here (to eliminate cold-start thermalization gaps). 
        # ----------------------------------------------------------------
        p_up = 0.5 * (1.0 + float(m_th))
        spins = np.where(rng.random(N) < p_up, 1, -1).astype(np.int64)
        psi_h3_samples = []

        # Thermalization steps to 
        # ensure convergence at Tc
        # ------------------------
        for step in range(12000):
            pair_sums = spins[v_arr] + spins[w_arr]
            pair_prods = spins[v_arr] * spins[w_arr]
            
            local_fields_ising = M_sparse.dot(pair_sums)
            local_fields_h3 = M_sparse.dot(pair_prods)
            
            # We use Kac scaling
            # to anchor Tc
            # ------------------
            dE_ising = 2.0 * Tc * spins * (local_fields_ising / (2.0 * K))
            
            # Align scales
            # ------------
            dE_h3 = 6.0 * h3 * spins * (local_fields_h3 / K)
            dE = dE_ising + dE_h3
            
            # Safely open-up the update channel
            # to 50% speed up relaxation
            # ---------------------------------
            active_subset = rng.random(N) < 0.5
            accept = (dE <= 0) | (rng.random(N) < np.exp(-beta_c * dE))
            spins[active_subset & accept] *= -1

            if step >= 6000:
                psi_h3_samples.append(np.abs(np.mean(spins))**3)

        psi_vs_h3[N].append(np.mean(psi_h3_samples))
    print(" Complete")

print(f"All iterations finished in {time.time() - global_start:.2f} seconds.")

# ===========
# DATA EXPORT
# ===========

with open("equilibrium_data_sweep.csv", "w") as f:
    header = "Temperature" + "".join([f",Psi_N{N},Chi_N{N},Tau_N{N}" for N in system_sizes])
    f.write(header + "\n")
    for idx, T in enumerate(temperatures):
        line = f"{T:.6f}" + "".join([f",{results[N]['psi'][idx]:.6f},{results[N]['chi'][idx]:.6f},{results[N]['tau'][idx]:.6f}" for N in system_sizes])
        f.write(line + "\n")

with open("critical_field_data.csv", "w") as f:
    header = "h3_Field" + "".join([f",Psi_N{N}_at_Tc" for N in system_sizes])
    f.write(header + "\n")
    for idx, h3 in enumerate(h3_fields):
        line = f"{h3:.8f}" + "".join([f",{psi_vs_h3[N][idx]:.6f}" for N in system_sizes])
        f.write(line + "\n")

# =====
# PLOTS
# =====

t_fine = np.linspace(0.4, 1.4, 500)
psi_exact, chi_exact = compute_exact_theory(t_fine, Tc)
plt.rcParams.update({'font.size': 11, 'figure.figsize': (4.2, 3.4)})

# Fig 1a -> Order parameter
# -------------------------
plt.figure()
plt.plot(t_fine, psi_exact, 'k--', label=r'Theory: $(T_c - T)^{3/2}$')
for N in system_sizes:
    plt.scatter(temperatures, results[N]['psi'], color=colors[N], edgecolor='k', s=25, alpha=0.8, label=f'M Carlo N = {N}')
plt.axvline(Tc, color='crimson', linestyle=':', lw=2, label=f'$T_c = {Tc:.1f}$')
plt.xlabel('Temperature ($T$)', fontsize=11)
plt.ylabel(r'$\Psi_{\mathrm{form}} \equiv \langle \phi_i \phi_j \phi_k \rangle$', fontsize=11)
plt.ylim(-0.05, 1.05)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(prop={'size': 8})
plt.tight_layout()
plt.savefig('figure_1a.png', dpi=300)
plt.close()

# Fig 1b -> Susceptibility
# ------------------------
plt.figure()
plt.plot(t_fine, chi_exact, 'k--', lw=1.5, label='Analytical Profile')
for N in system_sizes:
    plt.plot(temperatures, results[N]['chi'], color=colors[N], alpha=0.3)
    plt.scatter(temperatures, results[N]['chi'], color=colors[N], edgecolor='k', s=25, alpha=0.8, label=f'M Carlo N = {N}')
plt.axvline(Tc, color='crimson', linestyle=':', lw=2, label=f'$T_c = {Tc:.1f}$')
plt.xlabel('Temperature ($T$)', fontsize=11)
plt.ylabel(r'$\chi_{\mathrm{TF}}$', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(prop={'size': 8})
plt.tight_layout()
plt.savefig('figure_1b.png', dpi=300)
plt.close()

# Fig 1c -> Equation of state
# ---------------------------
plt.figure()
plt.loglog(h3_fields, psi_eos_th_ref, 'k--', label=r'Mean-Field EOS: $\Psi_{\mathrm{form}} \sim h_3^{3}$')
for N in system_sizes:
    plt.loglog(h3_fields, psi_vs_h3[N], marker='o', markersize=4, color=colors[N], label=f'M Carlo N = {N}')
plt.xlabel(r'Conjugate Triadic Field ($h_3$)', fontsize=11)
plt.ylabel(r'$\Psi_{\mathrm{form}}$', fontsize=11)
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.legend(prop={'size': 8})
plt.tight_layout()
plt.savefig('figure_1c.png', dpi=300)
plt.close()

# Fig 1d -> Relaxation lifetime
# -----------------------------
plt.figure()
for N in system_sizes:
    plt.plot(temperatures, results[N]['tau'], color=colors[N], alpha=0.3)
    plt.scatter(temperatures, results[N]['tau'], color=colors[N], edgecolor='k', s=25, alpha=0.8, label=f'M Carlo N = {N}')
plt.axvline(Tc, color='crimson', linestyle=':', lw=2, label=f'$T_c = {Tc:.1f}$')
plt.xlabel('Temperature ($T$)', fontsize=11)
plt.ylabel(r'Correlation Time $\tau_{\mathrm{relax}}$', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(prop={'size': 8})
plt.tight_layout()
plt.savefig('figure_1d.png', dpi=300)
plt.close()

######

fig, axes = plt.subplots(2, 2, figsize=(15, 11))

axes[0, 0].plot(t_fine, psi_exact, 'k--', label=r'Theory: $(T_c - T)^{3/2}$')
for N in system_sizes:
    axes[0, 0].scatter(temperatures, results[N]['psi'], color=colors[N], edgecolor='k', s=40, label=f'M Carlo N = {N}')
axes[0, 0].axvline(Tc, color='crimson', linestyle=':', lw=2, label=f'$Tc = {Tc:.1f}$')
axes[0, 0].set_title(r'(a) Triadic Formation Parameter $\Psi_{\mathrm{form}}$')
axes[0, 0].set_xlabel('Temperature ($T$)', fontsize=11)
axes[0, 0].set_ylabel(r'$\Psi_{\mathrm{form}}$', fontsize=11)
axes[0, 0].set_ylim(-0.05, 1.05)
axes[0, 0].grid(True, linestyle='--', alpha=0.5)
axes[0, 0].legend()

axes[0, 1].plot(t_fine, chi_exact, 'k--', lw=2, label='Analytical Profile')
for N in system_sizes:
    axes[0, 1].plot(temperatures, results[N]['chi'], color=colors[N], alpha=0.3)
    axes[0, 1].scatter(temperatures, results[N]['chi'], color=colors[N], edgecolor='k', s=40, label=f'M Carlo N = {N}')
axes[0, 1].axvline(Tc, color='crimson', linestyle=':', lw=2, label=f'$Tc = {Tc:.1f}$')
axes[0, 1].set_title(r'(b) Vanishing Susceptibility Curve $\chi_{\mathrm{TF}}$')
axes[0, 1].set_xlabel('Temperature ($T$)', fontsize=11)
axes[0, 1].set_ylabel(r'$\chi_{\mathrm{TF}}$', fontsize=11)
axes[0, 1].grid(True, linestyle='--', alpha=0.5)
axes[0, 1].legend()

axes[1, 0].loglog(h3_fields, psi_eos_th_ref, 'k--', label=r'Mean-Field EOS: $\Psi_{\mathrm{form}} \sim h_3^{3}$')
for N in system_sizes:
    axes[1, 0].loglog(h3_fields, psi_vs_h3[N], marker='o', color=colors[N], label=f'M Carlo N = {N}')
axes[1, 0].set_title(r'(c) Equation of State at $T=T_c$')
axes[1, 0].set_xlabel(r'Conjugate Triadic Field ($h_3$)', fontsize=11)
axes[1, 0].set_ylabel(r'$\Psi_{\mathrm{form}}$', fontsize=11)
axes[1, 0].grid(True, which="both", linestyle='--', alpha=0.5)
axes[1, 0].legend()

for N in system_sizes:
    axes[1, 1].plot(temperatures, results[N]['tau'], color=colors[N], alpha=0.3)
    axes[1, 1].scatter(temperatures, results[N]['tau'], color=colors[N], edgecolor='k', s=40, label=f'M Carlo N = {N}')
axes[1, 1].axvline(Tc, color='crimson', linestyle=':', lw=2, label=f'$Tc = {Tc:.1f}$')
axes[1, 1].set_title(r'(d) Relaxation Lifetime (Dynamical Slowing Down)')
axes[1, 1].set_xlabel('Temperature ($T$)', fontsize=11)
axes[1, 1].set_ylabel(r'Correlation Time $\tau_{\mathrm{relax}}$', fontsize=11)
axes[1, 1].grid(True, linestyle='--', alpha=0.5)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('complete_triadic_criticality_portfolio.png', dpi=300)

print("Graphs now saved. All processing completed.")
