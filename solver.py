"""
Steady-state top-speed solver for a high-speed drone.

Loads a YAML config + APC *_analysed.json prop artifact, then solves the
12-equation system from Governing_Physics/main.tex (A1–A3, P1–P4, E1–E3, M1–M2).

Unknowns (SI; n in rev/s):
    Re, v, phi, D, T, v_ax, J, n, V_term, I_bat, V_m, I_m
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import root

from prop_model import PropModel, load_prop

from datetime import datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
G = 9.80665          # m/s^2
R_S = 287.05         # J/(kg·K), dry air
MPS_TO_MPH = 2.2369362920544

UNKNOWN_NAMES = (
    "Re", "v", "phi", "D", "T", "v_ax", "J", "n", "V_term", "I_bat", "V_m", "I_m",
)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass
class Params:
    """Fixed inputs for one solve (prop geometry/fit come from PropModel)."""

    DC: float
    mass_kg: float
    W: float
    CdA: float
    N: int
    rho: float
    mu: float
    T_K: float | None
    Kv: float
    Rm: float
    I0: float
    Kt: float
    Ke: float
    Voc: float
    Rbat: float
    eta_bus: float
    Ah_rating: float | None
    prop: PropModel
    yaml_path: Path


def _resolve_prop_path(yaml_path: Path, prop_rel: str) -> Path:
    candidate = Path(prop_rel).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    from_yaml = (yaml_path.parent / candidate).resolve()
    if from_yaml.is_file():
        return from_yaml
    from_cwd = (Path.cwd() / candidate).resolve()
    if from_cwd.is_file():
        return from_cwd
    return from_yaml  # prefer YAML-relative for the eventual FileNotFoundError


def load_config(yaml_path: str | Path) -> Params:
    yaml_path = Path(yaml_path).expanduser().resolve()
    with yaml_path.open() as f:
        cfg = yaml.safe_load(f)

    control = cfg["control"]
    vehicle = cfg["vehicle"]
    env = cfg["environment"]
    motor = cfg["motor"]
    battery = cfg["battery"]
    propeller = cfg["propeller"]

    mass_kg = float(vehicle["mass_kg"])
    W = mass_kg * G

    # Check on provided method of atmospheric conditions
    if "rho" in env and env["rho"] is not None:
        rho = float(env["rho"])
    elif "p_Pa" in env and "T_K" in env:
        rho = float(env["p_Pa"]) / (R_S * float(env["T_K"]))
    else:
        raise ValueError("environment must provide rho, or both p_Pa and T_K")

    T_K = float(env["T_K"]) if env.get("T_K") is not None else None
    mu = float(env.get("mu", 1.8e-5))

    Kv = float(motor["Kv"]) # RPM / Volt
    Kt = Ke = 60.0 / (2.0 * math.pi * Kv)

    prop_path = _resolve_prop_path(yaml_path, str(propeller["json"]))
    prop = load_prop(prop_path)

    return Params(
        DC=float(control["DC"]),
        mass_kg=mass_kg,
        W=W,
        CdA=float(vehicle["CdA"]),
        N=int(vehicle["N"]),
        rho=rho,
        mu=mu,
        T_K=T_K,
        Kv=Kv,
        Rm=float(motor["Rm"]),
        I0=float(motor["I0"]),
        Kt=Kt,
        Ke=Ke,
        Voc=float(battery["Voc"]),
        Rbat=float(battery["Rbat"]),
        eta_bus=float(battery["eta_bus"]),
        Ah_rating=float(battery["Ah_rating"]) if battery.get("Ah_rating") is not None else None,
        prop=prop,
        yaml_path=yaml_path,
    )


# ---------------------------------------------------------------------------
# Residuals  (match Governing_Physics / The Final System)
# ---------------------------------------------------------------------------
def unpack(x: np.ndarray) -> dict[str, float]:
    return {name: float(x[i]) for i, name in enumerate(UNKNOWN_NAMES)}


def residuals(x: np.ndarray, p: Params) -> np.ndarray:
    """Twelve residuals A1…M2. Order matches UNKNOWN_NAMES."""
    Re, v, phi, D, T, v_ax, J, n, V_term, I_bat, V_m, I_m = [float(v_) for v_ in x]

    d = p.prop.diameter_m
    c75 = p.prop.chord_75R_m
    CT = p.prop.ct(J, Re)
    CP = p.prop.cp(J, Re)

    # Keep denominators defined for the solver (soft; real solutions stay positive)
    n_safe = n if abs(n) > 1e-12 else 1e-12
    D_safe = D if abs(D) > 1e-12 else 1e-12

    # Vehicle dynamics
    r_A1 = D - 0.5 * p.rho * v**2 * p.CdA
    r_A2 = T - math.hypot(p.W, D)
    r_A3 = phi - math.atan(p.W / D_safe)

    # Propeller
    r_P1 = v_ax - v * math.cos(phi)
    r_P2 = J - v_ax / (n_safe * d)
    r_P3 = p.N * CT * p.rho * n**2 * d**4 - T
    r_P4 = Re - (p.rho * c75 * v) / p.mu

    # Battery and ESC
    r_E1 = V_term - (p.Voc - I_bat * p.Rbat)
    r_E2 = V_m - p.DC * V_term
    r_E3 = I_bat - (p.N * p.DC * I_m) / p.eta_bus

    # Motor
    r_M1 = V_m - p.Ke * (2.0 * math.pi * n) - I_m * p.Rm
    r_M2 = p.Kt * (I_m - p.I0) - (CP * p.rho * n**2 * d**5) / (2.0 * math.pi)

    return np.array(
        [r_A1, r_A2, r_A3, r_P1, r_P2, r_P3, r_P4, r_E1, r_E2, r_E3, r_M1, r_M2],
        dtype=float,
    )


def initial_guess(p: Params) -> np.ndarray:
    """Positive, physics-motivated starting point for the nonlinear solve."""
    d = p.prop.diameter_m
    c75 = p.prop.chord_75R_m

    v = 60.0  # m/s ~ 134 mph starting guess
    D = 0.5 * p.rho * v**2 * p.CdA
    T = math.hypot(p.W, D)
    phi = math.atan2(p.W, D)
    v_ax = v * math.cos(phi)

    n = (p.Kv * p.Voc * p.DC) / 60.0  # rev/s from unloaded motor scaling
    n = max(n, 50.0)
    J = v_ax / (n * d)
    Re = (p.rho * c75 * v) / p.mu

    V_term = p.Voc * 0.95
    V_m = p.DC * V_term
    I_m = max(p.I0 + 5.0, 10.0)
    I_bat = (p.N * p.DC * I_m) / p.eta_bus

    return np.array(
        [Re, v, phi, D, T, v_ax, J, n, V_term, I_bat, V_m, I_m],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Validity warnings
# ---------------------------------------------------------------------------
def _percent_outside(value: float, lo: float, hi: float) -> tuple[str | None, float]:
    """Return (side, percent of envelope width) if outside [lo, hi], else (None, 0)."""
    width = hi - lo
    if width <= 0:
        width = abs(hi) if hi != 0 else 1.0
    if value < lo:
        return "below", 100.0 * (lo - value) / width
    if value > hi:
        return "above", 100.0 * (value - hi) / width
    return None, 0.0


def validity_warnings(J: float, Re: float, prop: PropModel) -> list[str]:
    v = prop.validity
    warnings: list[str] = []

    side, pct = _percent_outside(J, v["J_min"], v["J_max"])
    if side == "above":
        warnings.append(
            f"WARNING: J={J:.4g} is {pct:.1f}% above fit max (J_max={v['J_max']:.4g})"
        )
    elif side == "below":
        warnings.append(
            f"WARNING: J={J:.4g} is {pct:.1f}% below fit min (J_min={v['J_min']:.4g})"
        )

    side, pct = _percent_outside(Re, v["Re_min"], v["Re_max"])
    if side == "above":
        warnings.append(
            f"WARNING: Re={Re:.4g} is {pct:.1f}% above fit max (Re_max={v['Re_max']:.4g})"
        )
    elif side == "below":
        warnings.append(
            f"WARNING: Re={Re:.4g} is {pct:.1f}% below fit min (Re_min={v['Re_min']:.4g})"
        )

    return warnings


# ---------------------------------------------------------------------------
# Solve + derived quantities
# ---------------------------------------------------------------------------
def solve(p: Params) -> dict[str, Any]:
    x0 = initial_guess(p)
    sol = root(lambda x: residuals(x, p), x0, method="hybr")

    state = unpack(sol.x)
    res = residuals(sol.x, p)
    res_norm = float(np.linalg.norm(res))

    result: dict[str, Any] = {
        "success": bool(sol.success),
        "message": str(sol.message),
        "residual_norm": res_norm,
        "state": state,
        "params": p,
        "warnings": validity_warnings(state["J"], state["Re"], p.prop),
        "derived": {},
    }

    if not sol.success:
        return result

    result["derived"] = _derived(state, p)
    return result


def _derived(s: dict[str, float], p: Params) -> dict[str, Any]:
    d = p.prop.diameter_m
    pitch = p.prop.pitch_m
    J, Re, n = s["J"], s["Re"], s["n"]
    CT = p.prop.ct(J, Re)
    CP = p.prop.cp(J, Re)

    eta_p = (J * CT / CP) if abs(CP) > 1e-16 else float("nan")
    shaft_power = p.Kt * (s["I_m"] - p.I0) * (2.0 * math.pi * n)
    elec_power = s["V_m"] * s["I_m"]
    eta_m = (shaft_power / elec_power) if abs(elec_power) > 1e-16 else float("nan")
    eta_bat = s["V_term"] / p.Voc if abs(p.Voc) > 1e-16 else float("nan")

    P_battery = p.Voc * s["I_bat"]
    P_useful = s["D"] * s["v"]
    eta_chain = eta_bat * p.eta_bus * eta_m * eta_p
    P_battery_from_chain = P_useful / eta_chain if abs(eta_chain) > 1e-16 else float("nan")

    out: dict[str, Any] = {
        "CT": CT,
        "CP": CP,
        "eta_p": eta_p,
        "eta_m": eta_m,
        "eta_bat": eta_bat,
        "eta_bus": p.eta_bus,
        "P_battery_W": P_battery,
        "P_useful_W": P_useful,
        "P_battery_from_chain_W": P_battery_from_chain,
        "power_check_rel_err": ( abs(P_battery - P_battery_from_chain) / abs(P_battery) if abs(P_battery) > 1e-16 else float("nan") ),
        "v_mps": s["v"],
        "v_mph": s["v"] * MPS_TO_MPH,
        "n_rps": n,
        "n_rpm": n * 60.0,
        "v_ax_max": pitch * n,
        "phi_deg": math.degrees(s["phi"]),
    }

    if p.Ah_rating is not None and p.Ah_rating > 0:
        out["C_rate"] = s["I_bat"] / p.Ah_rating

    if p.T_K is not None and p.T_K > 0:
        gamma = 1.4
        a = math.sqrt(gamma * R_S * p.T_K)
        tip_speed = math.hypot(math.pi * n * d, s["v_ax"])
        out["a_mps"] = a
        out["M_tip"] = tip_speed / a

    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def format_report(result: dict[str, Any]) -> str:
    p: Params = result["params"]
    s = result["state"]
    d = result["derived"]
    lines: list[str] = []

    lines.append("=" * 60)
    lines.append("Airspeed Solver — top-speed result")
    lines.append("=" * 60)
    lines.append(f"Config:   {p.yaml_path}")
    lines.append(f"Prop:     {p.prop.prop_id}  ({p.prop.path.name})")
    lines.append(f"  d = {p.prop.diameter_m:.4g} m,  pitch = {p.prop.pitch_m:.4g} m,  "
                 f"c_0.75r = {p.prop.chord_75R_m:.4g} m")
    lines.append(
        f"Inputs:   mass={p.mass_kg:.4g} kg (W={p.W:.4g} N), CdA={p.CdA:.4g} m^2, "
        f"N={p.N}, DC={p.DC:.3g}"
    )
    lines.append(
        f"Motor:    Kv={p.Kv:.0f} RPM/V, Rm={p.Rm:.4g} ohm, I0={p.I0:.4g} A, "
        f"Kt=Ke={p.Kt:.6g}"
    )
    lines.append(
        f"Battery:  Voc={p.Voc:.4g} V, Rbat={p.Rbat:.4g} ohm, eta_bus={p.eta_bus:.4g}"
    )
    lines.append("")

    if not result["success"]:
        lines.append("SOLVE FAILED")
        lines.append(f"  message: {result['message']}")
        lines.append(f"  residual_norm: {result['residual_norm']:.4e}")
        lines.append("")
        lines.append("Last iterate (may be invalid):")
        for name in UNKNOWN_NAMES:
            lines.append(f"  {name:8s} = {s[name]:.6g}")
        for w in result["warnings"]:
            lines.append(w)
        lines.append("=" * 60)
        return "\n".join(lines) + "\n"

    lines.append(f"Top speed:  {d['v_mps']:.3f} m/s   ({d['v_mph']:.1f} mph)")
    lines.append(f"RPM:        {d['n_rpm']:.0f}   ({d['n_rps']:.2f} rev/s)")
    lines.append(f"phi:        {d['phi_deg']:.2f} deg")
    lines.append(f"residual:   {result['residual_norm']:.3e}  ({result['message']})")
    lines.append("")
    lines.append("State")
    lines.append(f"  Re      = {s['Re']:.6g}")
    lines.append(f"  v       = {s['v']:.6g} m/s")
    lines.append(f"  phi     = {s['phi']:.6g} rad")
    lines.append(f"  D       = {s['D']:.6g} N")
    lines.append(f"  T       = {s['T']:.6g} N")
    lines.append(f"  v_ax    = {s['v_ax']:.6g} m/s")
    lines.append(f"  J       = {s['J']:.6g}")
    lines.append(f"  n       = {s['n']:.6g} rev/s")
    lines.append(f"  V_term  = {s['V_term']:.6g} V")
    lines.append(f"  I_bat   = {s['I_bat']:.6g} A")
    lines.append(f"  V_m     = {s['V_m']:.6g} V")
    lines.append(f"  I_m     = {s['I_m']:.6g} A")
    lines.append("")
    lines.append("Derived")
    lines.append(f"  CT={d['CT']:.5g}  CP={d['CP']:.5g}")
    lines.append(
        f"  eta_p={d['eta_p']:.4f}  eta_m={d['eta_m']:.4f}  "
        f"eta_bat={d['eta_bat']:.4f}  eta_bus={d['eta_bus']:.4f}"
    )
    lines.append(
        f"  P_battery={d['P_battery_W']:.2f} W  P_useful={d['P_useful_W']:.2f} W  "
        f"power_check_rel_err={d['power_check_rel_err']:.3e}"
    )
    lines.append(f"  v_ax_max (pitch*n) = {d['v_ax_max']:.3f} m/s  "
                 f"(v_ax={s['v_ax']:.3f} m/s)")
    if "C_rate" in d:
        lines.append(f"  C_rate = {d['C_rate']:.2f}")
    if "M_tip" in d:
        lines.append(f"  M_tip = {d['M_tip']:.3f}  (a={d['a_mps']:.1f} m/s)")

    if result["warnings"]:
        lines.append("")
        lines.append("Validity")
        for w in result["warnings"]:
            lines.append(f"  {w}")
    else:
        v = p.prop.validity
        lines.append("")
        lines.append(
            f"Validity: J and Re within fit envelope "
            f"[J {v['J_min']:.4g}–{v['J_max']:.4g}, "
            f"Re {v['Re_min']:.4g}–{v['Re_max']:.4g}]"
        )

    lines.append("=" * 60)
    return "\n".join(lines) + "\n"


def default_results_path(yaml_path: Path) -> Path:
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / f"{yaml_path.stem}_result.txt"


def report(result: dict[str, Any], results_path: Path | None = None) -> str:
    text = format_report(result)
    print(text, end="")

    p: Params = result["params"]
    base_path = results_path or default_results_path(p.yaml_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = base_path.with_name(f"{base_path.stem}_{timestamp}{base_path.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(f"Wrote results to {out_path}")
    return text


def solve_from_files(yaml_path: str | Path) -> dict[str, Any]:
    """Load config, solve, return result dict (no printing)."""
    params = load_config(yaml_path)
    return solve(params)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Solve theoretical top speed from a drone YAML + prop JSON."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=str(Path(__file__).resolve().parent / "configs" / "example_drone.yaml"),
        help="Path to drone YAML config (default: configs/example_drone.yaml)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Results text file path (default: results/<config_stem>_result.txt)",
    )
    args = parser.parse_args(argv)

    try:
        result = solve_from_files(args.config)
    except Exception as exc:  # BLE001 — CLI should show a clean error
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    out = Path(args.output) if args.output else None
    report(result, results_path=out)
    return 0 if result["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
