"""
Black-Scholes PDE: Finite Difference Analysis

Implements and compares three finite difference schemes for pricing
European call options directly on the Black-Scholes PDE:
  - FTCS  (explicit, conditionally stable)
  - BTCS  (implicit, unconditionally stable)
  - Crank-Nicolson (implicit, unconditionally stable, 2nd-order in time)

Results are validated against the closed-form Black-Scholes formula.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.special import erf


def black_scholes_call(S, K, r, sigma, T):
    """Closed-form Black-Scholes price for a European call."""
    V = np.zeros_like(S, dtype=float)
    idx = S > 0
    d1 = (np.log(S[idx] / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    N_d1 = 0.5 * (1 + erf(d1 / np.sqrt(2)))
    N_d2 = 0.5 * (1 + erf(d2 / np.sqrt(2)))
    V[idx] = S[idx] * N_d1 - K * np.exp(-r * T) * N_d2
    return V


def bs_fd_solver(S, K, r, sigma, T, N, method):
    """
    Solve the Black-Scholes PDE using a finite difference scheme.

    Parameters
    ----------
    S      : spatial grid (M+1 points, including boundaries)
    K      : strike price
    r      : risk-free interest rate
    sigma  : volatility
    T      : time to maturity
    N      : number of time steps
    method : 'FTCS' | 'BTCS' | 'CN'

    Returns
    -------
    V : option prices at t=0 on the grid S
    """
    M  = len(S) - 1
    dS = S[1] - S[0]
    dt = T / N

    V  = np.maximum(S - K, 0).astype(float)
    Si = S[1:M]

    # PDE coefficients: L[V] = a*V_{j-1} + b*V_j + c*V_{j+1}
    a =  0.5 * sigma**2 * Si**2 / dS**2 - 0.5 * r * Si / dS
    b = -sigma**2 * Si**2 / dS**2 - r
    c =  0.5 * sigma**2 * Si**2 / dS**2 + 0.5 * r * Si / dS

    A = diags([a[1:], b, c[:-1]], offsets=[-1, 0, 1],
              shape=(M - 1, M - 1), format='csr')
    I = diags([np.ones(M - 1)], offsets=[0],
              shape=(M - 1, M - 1), format='csr')

    if method.upper() == 'BTCS':
        LHS_b = I - dt * A
    elif method.upper() == 'CN':
        LHS_cn = I - 0.5 * dt * A
        RHS_cn = I + 0.5 * dt * A

    for n in range(N):
        # Right boundary tracks the deep in-the-money asymptote S - Ke^(-r*tau)
        V_right_old = S[-1] - K * np.exp(-r * n * dt)
        V_right_new = S[-1] - K * np.exp(-r * (n + 1) * dt)

        bc_old = np.zeros(M - 1)
        bc_new = np.zeros(M - 1)
        bc_old[-1] = c[-1] * V_right_old
        bc_new[-1] = c[-1] * V_right_new

        V_in = V[1:M].copy()

        if method.upper() == 'FTCS':
            V_in_new = V_in + dt * (A @ V_in + bc_old)
        elif method.upper() == 'BTCS':
            V_in_new = spsolve(LHS_b, V_in + dt * bc_new)
        elif method.upper() == 'CN':
            rhs = RHS_cn @ V_in + 0.5 * dt * (bc_old + bc_new)
            V_in_new = spsolve(LHS_cn, rhs)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'FTCS', 'BTCS', or 'CN'.")

        V = np.concatenate([[0.0], V_in_new, [V_right_new]])

    return V


COLORS = {
    'exact': ('black',   '-',   2,   'Analytical'),
    'ftcs':  ('#e74c3c', '--',  2,   'FTCS'),
    'btcs':  ('#2980b9', '-.',  2,   'BTCS'),
    'cn':    ('#8e44ad', ':',   2.5, 'Crank-Nicolson'),
}

MARKERS = {
    'ftcs': ('o', slice(0, None, 12)),
    'btcs': ('s', slice(2, None, 12)),
    'cn':   ('d', slice(5, None, 12)),
}


def _plot_price(ax, S, solutions, title, xlim=(0, 180)):
    color, ls, lw, label = COLORS['exact']
    ax.plot(S, solutions['exact'], color=color, ls=ls, lw=lw, label=label, zorder=5)
    for key in ('ftcs', 'btcs', 'cn'):
        color, ls, lw, label = COLORS[key]
        mk, sl = MARKERS[key]
        ax.plot(S, solutions[key], color=color, ls=ls, lw=lw, label=label,
                marker=mk, markevery=sl, markersize=4)
    ax.set_xlabel('Asset Price  S', fontsize=13)
    ax.set_ylabel('Option Value  V(S,0)', fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left', fontsize=11)
    ax.set_xlim(xlim)
    ax.grid(True, alpha=0.4)


def _plot_error(ax, S, solutions, title, xlim=(0, 180)):
    exact = solutions['exact']
    for key in ('ftcs', 'btcs', 'cn'):
        color, ls, lw, label = COLORS[key]
        mk, sl = MARKERS[key]
        ax.semilogy(S, np.abs(solutions[key] - exact) + 1e-14,
                    color=color, ls=ls, lw=lw, label=label,
                    marker=mk, markevery=sl, markersize=4)
    ax.set_xlabel('Asset Price  S', fontsize=13)
    ax.set_ylabel('|Error|  (log scale)', fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(xlim)
    ax.grid(True, alpha=0.4)


def make_figure(stable_solutions, unstable_solutions,
                N_stable, N_unstable, lambda_stable, lambda_unstable, S):

    fig1, axes1 = plt.subplots(2, 1, figsize=(11, 8), facecolor='white')
    fig1.suptitle('Finite Difference Schemes vs Analytical Black-Scholes',
                  fontsize=15, fontweight='bold', y=0.98)
    _plot_price(axes1[0], S, stable_solutions,
                title=f'Stable Schemes vs Analytical  (N={N_stable},  λ_max={lambda_stable:.3f})')
    _plot_error(axes1[1], S, stable_solutions,
                title='Absolute Error Relative to Analytical Solution')
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    fig1.savefig('bs_stable_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved: bs_stable_comparison.png")

    fig2, axes2 = plt.subplots(2, 1, figsize=(11, 8), facecolor='white')
    fig2.suptitle('FTCS Instability Demonstration', fontsize=15, fontweight='bold', y=0.98)
    _plot_price(axes2[0], S, unstable_solutions,
                title=f'FTCS Instability Demo  (N={N_unstable},  λ_max={lambda_unstable:.2f})')
    axes2[0].set_ylim(-50, 250)
    _plot_error(axes2[1], S, unstable_solutions,
                title='Absolute Error: Unstable vs Stable Comparison')
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig('bs_instability_demo.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved: bs_instability_demo.png")

    plt.show()


def main():
    K     = 100.0
    r     = 0.05
    sigma = 0.2
    T     = 1.0
    Smax  = 300.0

    M          = 200
    N_stable   = 2000
    N_unstable = 200

    S  = np.linspace(0, Smax, M + 1)
    dS = S[1] - S[0]

    lam_stable   = (T / N_stable)   / dS**2 * sigma**2 * Smax**2 / 2
    lam_unstable = (T / N_unstable) / dS**2 * sigma**2 * Smax**2 / 2

    print("=== Stability Info ===")
    print(f"dS = {dS:.4f}")
    print(f"Stable case   — dt={T/N_stable:.6f},  λ_max={lam_stable:.4f}  "
          f"→ {'STABLE' if lam_stable <= 0.5 else 'UNSTABLE'}")
    print(f"Unstable case — dt={T/N_unstable:.6f},  λ_max={lam_unstable:.4f}  "
          f"→ {'STABLE' if lam_unstable <= 0.5 else 'UNSTABLE'}")
    print()

    V_exact = black_scholes_call(S, K, r, sigma, T)

    print(f"Running stable solvers (N={N_stable})...")
    V_ftcs = bs_fd_solver(S, K, r, sigma, T, N_stable, 'FTCS')
    V_btcs = bs_fd_solver(S, K, r, sigma, T, N_stable, 'BTCS')
    V_cn   = bs_fd_solver(S, K, r, sigma, T, N_stable, 'CN')

    print(f"\n=== Max Absolute Errors (N={N_stable}) ===")
    print(f"  FTCS:           {np.max(np.abs(V_ftcs - V_exact)):.6e}")
    print(f"  BTCS:           {np.max(np.abs(V_btcs - V_exact)):.6e}")
    print(f"  Crank-Nicolson: {np.max(np.abs(V_cn   - V_exact)):.6e}")
    print()

    print(f"Running instability demo (N={N_unstable})...")
    V_ftcs_un = bs_fd_solver(S, K, r, sigma, T, N_unstable, 'FTCS')
    V_btcs_un = bs_fd_solver(S, K, r, sigma, T, N_unstable, 'BTCS')
    V_cn_un   = bs_fd_solver(S, K, r, sigma, T, N_unstable, 'CN')
    print()

    stable_solutions   = dict(exact=V_exact, ftcs=V_ftcs,    btcs=V_btcs,    cn=V_cn)
    unstable_solutions = dict(exact=V_exact, ftcs=V_ftcs_un, btcs=V_btcs_un, cn=V_cn_un)

    make_figure(stable_solutions, unstable_solutions,
                N_stable, N_unstable, lam_stable, lam_unstable, S)


if __name__ == '__main__':
    main()