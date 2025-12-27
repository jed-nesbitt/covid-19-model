import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from scipy.integrate import solve_ivp


# -----------------------------
# OUTPUT HELPERS
# -----------------------------
def ensure_outdir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_fig(fig, out_path: Path, dpi: int = 300) -> None:
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# SEIRS MODEL (MATLAB -> PYTHON)
# -----------------------------
@dataclass(frozen=True)
class SEIRSParams:
    beta: float = 0.17
    gamma: float = 1 / 14
    b: float = 12 / 365000
    d: float = 6.8 / 365000
    mu: float = 3.2 / 365000
    alpha: float = 1 / 7
    omega: float = 1 / 180


def seirs_rhs(t, y, p: SEIRSParams):
    S, E, I, R = y
    N = S + E + I + R
    if N <= 0:
        return [0.0, 0.0, 0.0, 0.0]

    dS = p.b * N + p.omega * R - p.beta * S * I / N - p.d * S
    dE = p.beta * S * I / N - p.alpha * E - p.d * E
    dI = p.alpha * E - p.gamma * I - I * (p.d + p.mu)
    dR = p.gamma * I - R * (p.omega + p.d)

    return [dS, dE, dI, dR]


# -----------------------------
# SOLVER (ode45 -> solve_ivp RK45)
# -----------------------------
def run_seirs(y0, tspan=(0, 1500), params=SEIRSParams(), n_points=1501):
    t_eval = np.linspace(tspan[0], tspan[1], n_points)

    sol = solve_ivp(
        fun=lambda t, y: seirs_rhs(t, y, params),
        t_span=tspan,
        y0=y0,
        method="RK45",
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solve failed: {sol.message}")

    return sol


# -----------------------------
# PLOTTING (SAVES TO output/)
# -----------------------------
def plot_timeseries(sol, out_dir: Path, title: str = "SEIRS Model - R0=2.2"):
    fig = plt.figure()
    plt.plot(sol.t, sol.y[0], label="Susceptible")
    plt.plot(sol.t, sol.y[1], label="Exposed")
    plt.plot(sol.t, sol.y[2], label="Infected")
    plt.plot(sol.t, sol.y[3], label="Recovered")
    plt.title(title)
    plt.xlabel("days")
    plt.ylabel("Population (millions)")
    plt.legend()

    save_fig(fig, out_dir / "seirs_timeseries.png")


def plot_phase_plane_SI(sol, out_dir: Path, title: str = "Phase plane with R0=2.2"):
    fig = plt.figure()
    plt.plot(sol.y[0], sol.y[2])
    plt.title(title)
    plt.xlabel("Susceptible People (millions)")
    plt.ylabel("Infected People (millions)")

    save_fig(fig, out_dir / "seirs_phase_plane_SI.png")


def plot_phase_volume_SIR(sol, out_dir: Path, title: str = "Phase Volume of COVID-19 with R0=2.2"):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(sol.y[0], sol.y[2], sol.y[3])
    ax.set_title(title)
    ax.set_xlabel("Susceptible (millions)")
    ax.set_ylabel("Infected (millions)")
    ax.set_zlabel("Recovered (millions)")

    save_fig(fig, out_dir / "seirs_phase_volume_SIR.png")


def plot_beta_sweep_infected(y0, tspan, betas, out_dir: Path):
    fig = plt.figure()
    for beta in betas:
        sol = run_seirs(y0, tspan=tspan, params=SEIRSParams(beta=beta))
        plt.plot(sol.t, sol.y[2], label=f"I (beta={beta})")

    plt.title("Infected over time for different beta")
    plt.xlabel("days")
    plt.ylabel("Infected (millions)")
    plt.legend()

    save_fig(fig, out_dir / "seirs_beta_sweep_infected.png")


if __name__ == "__main__":
    out_dir = ensure_outdir(Path("output"))

    # initial conditions (same as MATLAB)
    y0 = [25, 1e-5, 1e-5, 0]
    tspan = (0, 1500)

    # Base run
    params = SEIRSParams(beta=0.17)
    sol = run_seirs(y0, tspan=tspan, params=params)

    # Save all plots
    plot_timeseries(sol, out_dir, title="SEIRS Model - R0=2.2")
    plot_phase_plane_SI(sol, out_dir, title="Phase plane with R0 = 2.2")
    plot_phase_volume_SIR(sol, out_dir, title="Phase Volume of COVID-19 with R0=2.2")

    # Optional: replicate the idea of “looping” by doing a beta sweep
    betas = [0.07, 0.10, 0.13, 0.17]
    plot_beta_sweep_infected(y0, tspan, betas, out_dir)

    print(f"✅ Saved images to: {out_dir.resolve()}")
