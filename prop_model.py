"""Load APC *_analysed.json artifacts and evaluate CT/CP(J, Re)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PropModel:
    """Propeller response surface and geometry from an analysed JSON file."""

    path: Path
    data: dict[str, Any]

    @property
    def prop_id(self) -> str:
        return str(self.data.get("prop_id", self.data.get("model", "unknown")))

    @property
    def model(self) -> str:
        return str(self.data.get("model", "unknown"))

    @property
    def diameter_m(self) -> float:
        return float(self.data["diameter_m"])

    @property
    def pitch_m(self) -> float:
        return float(self.data["pitch_m"])

    @property
    def chord_75R_m(self) -> float:
        geom = self.data.get("geometry") or {}
        value = geom.get("chord_75R_m")
        if value is None:
            raise ValueError(
                f"Prop JSON '{self.path}' is missing geometry.chord_75R_m "
                "(needed for Reynolds number P4). Re-run analyze_apc_prop with a geometry.dat file."
            )
        return float(value)

    @property
    def validity(self) -> dict[str, float]:
        v = self.data["validity"]
        return {
            "J_min": float(v["J_min"]),
            "J_max": float(v["J_max"]),
            "Re_min": float(v["Re_min"]),
            "Re_max": float(v["Re_max"]),
        }

    def ct(self, J: float, Re: float) -> float:
        return _eval_poly(J, Re, self.data["fit_model"], "CT_coefficients")

    def cp(self, J: float, Re: float) -> float:
        return _eval_poly(J, Re, self.data["fit_model"], "CP_coefficients")


def load_prop(path: str | Path) -> PropModel:
    path = Path(path).expanduser().resolve()
    with path.open() as f:
        data = json.load(f)
    if "fit_model" not in data:
        raise ValueError(f"Not an analysed prop JSON (missing fit_model): {path}")
    return PropModel(path=path, data=data)


def _eval_poly(J: float, Re: float, fit_model: dict[str, Any], coeff_key: str) -> float:
    scale = float(fit_model["Re_scaling"]["scale"])
    re_hat = Re / scale
    coeffs = fit_model[coeff_key]
    basis = fit_model["basis"]
    total = 0.0
    for a, term in zip(coeffs, basis):
        total += float(a) * (J ** int(term["i"])) * (re_hat ** int(term["j"]))
    return total
