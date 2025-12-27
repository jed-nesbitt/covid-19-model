from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# -----------------------------
# OUTPUT HELPERS
# -----------------------------
def ensure_outdir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_fig(fig, out_path: Path, dpi: int = 300) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _json_default(obj: Any):
    """Best-effort JSON serializer for odd types (Path, numpy scalars, timestamps)."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (datetime, np.datetime64)):
        return str(obj)
    if hasattr(obj, "item"):  # numpy scalars
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def write_run_metadata(path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=_json_default)


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
def run_seirs(
    y0,
    tspan=(0, 1500),
    params=SEIRSParams(),
    n_points=1501,
    method="RK45",
    rtol=1e-6,
    atol=1e-9,
):
    t_eval = np.linspace(tspan[0], tspan[1], n_points)
    sol = solve_ivp(
        fun=lambda t, y: seirs_rhs(t, y, params),
        t_span=tspan,
        y0=y0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solve failed: {sol.message}")
    return sol


# -----------------------------
# PLOTTING (SAVES TO output/)
# -----------------------------
def plot_timeseries(sol, out_dir: Path, title: str = "SEIRS Model - R0=2.2") -> Path:
    fig = plt.figure()
    plt.plot(sol.t, sol.y[0], label="Susceptible")
    plt.plot(sol.t, sol.y[1], label="Exposed")
    plt.plot(sol.t, sol.y[2], label="Infected")
    plt.plot(sol.t, sol.y[3], label="Recovered")
    plt.title(title)
    plt.xlabel("days")
    plt.ylabel("Population (millions)")
    plt.legend()
    out_path = out_dir / "seirs_timeseries.png"
    save_fig(fig, out_path)
    return out_path


def plot_phase_plane_SI(sol, out_dir: Path, title: str = "Phase plane with R0=2.2") -> Path:
    fig = plt.figure()
    plt.plot(sol.y[0], sol.y[2])
    plt.title(title)
    plt.xlabel("Susceptible People (millions)")
    plt.ylabel("Infected People (millions)")
    out_path = out_dir / "seirs_phase_plane_SI.png"
    save_fig(fig, out_path)
    return out_path


def plot_phase_volume_SIR(sol, out_dir: Path, title: str = "Phase Volume of COVID-19 with R0=2.2") -> Path:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(sol.y[0], sol.y[2], sol.y[3])
    ax.set_title(title)
    ax.set_xlabel("Susceptible (millions)")
    ax.set_ylabel("Infected (millions)")
    ax.set_zlabel("Recovered (millions)")
    out_path = out_dir / "seirs_phase_volume_SIR.png"
    save_fig(fig, out_path)
    return out_path


def plot_beta_sweep_infected(y0, tspan, betas, out_dir: Path) -> tuple[Path, list[dict]]:
    fig = plt.figure()
    sweep_summary: list[dict] = []

    for beta in betas:
        sol = run_seirs(y0, tspan=tspan, params=SEIRSParams(beta=beta))
        I = sol.y[2]
        peak_I = float(np.max(I))
        peak_t = float(sol.t[int(np.argmax(I))])
        sweep_summary.append({"beta": beta, "peak_I": peak_I, "peak_day": peak_t})

        plt.plot(sol.t, I, label=f"I (beta={beta})")

    plt.title("Infected over time for different beta")
    plt.xlabel("days")
    plt.ylabel("Infected (millions)")
    plt.legend()

    out_path = out_dir / "seirs_beta_sweep_infected.png"
    save_fig(fig, out_path)
    return out_path, sweep_summary


def get_versions() -> dict:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import numpy as _np
        versions["numpy"] = _np.__version__
    except Exception:
        versions["numpy"] = "unknown"
    try:
        import matplotlib as _mpl
        versions["matplotlib"] = _mpl.__version__
    except Exception:
        versions["matplotlib"] = "unknown"
    try:
        import scipy as _sp
        versions["scipy"] = _sp.__version__
    except Exception:
        versions["scipy"] = "unknown"
    return versions


if __name__ == "__main__":
    run_start = datetime.now().astimezone()
    t0 = time.perf_counter()

    out_dir = ensure_outdir(Path("output"))

    # initial conditions (same as MATLAB)
    y0 = [25, 1e-5, 1e-5, 0]
    tspan = (0, 1500)

    # Base run
    params = SEIRSParams(beta=0.17)

    # Keep solver config explicit so it ends up in metadata
    solver_cfg = {
        "method": "RK45",
        "n_points": 1501,
        "rtol": 1e-6,
        "atol": 1e-9,
    }

    sol = run_seirs(
        y0,
        tspan=tspan,
        params=params,
        n_points=solver_cfg["n_points"],
        method=solver_cfg["method"],
        rtol=solver_cfg["rtol"],
        atol=solver_cfg["atol"],
    )

    # Save all plots
    saved_files: list[str] = []
    saved_files.append(str(plot_timeseries(sol, out_dir, title="SEIRS Model - R0=2.2").as_posix()))
    saved_files.append(str(plot_phase_plane_SI(sol, out_dir, title="Phase plane with R0 = 2.2").as_posix()))
    saved_files.append(str(plot_phase_volume_SIR(sol, out_dir, title="Phase Volume of COVID-19 with R0=2.2").as_posix()))

    # Optional: beta sweep
    betas = [0.07, 0.10, 0.13, 0.17]
    sweep_plot_path, sweep_summary = plot_beta_sweep_infected(y0, tspan, betas, out_dir)
    saved_files.append(str(sweep_plot_path.as_posix()))

    # Build metadata
    run_end = datetime.now().astimezone()
    duration_sec = float(time.perf_counter() - t0)

    run_metadata = {
        "run_started": run_start.isoformat(),
        "run_ended": run_end.isoformat(),
        "duration_sec": round(duration_sec, 6),
        "output_dir": str(out_dir.resolve().as_posix()),
        "model": "SEIRS",
        "initial_conditions": {"S0": y0[0], "E0": y0[1], "I0": y0[2], "R0": y0[3]},
        "tspan": {"t0": tspan[0], "t1": tspan[1]},
        "params": asdict(params),
        "solver": solver_cfg,
        "solution_summary": {
            "n_time_points": int(sol.t.size),
            "peak_infected": float(np.max(sol.y[2])),
            "peak_day": float(sol.t[int(np.argmax(sol.y[2]))]),
            "final_values": {
                "S": float(sol.y[0, -1]),
                "E": float(sol.y[1, -1]),
                "I": float(sol.y[2, -1]),
                "R": float(sol.y[3, -1]),
            },
        },
        "beta_sweep": {
            "betas": betas,
            "summary": sweep_summary,
        },
        "generated_files": saved_files,
        "versions": get_versions(),
    }

    meta_path = out_dir / "run_metadata.json"
    write_run_metadata(meta_path, run_metadata)

    print(f"✅ Saved images to: {out_dir.resolve()}")
    print(f"✅ Saved run metadata to: {meta_path.resolve()}")
