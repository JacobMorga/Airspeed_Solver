# Airspeed Solver

Steady-state theoretical top-speed estimate for a high-speed drone.

Plug in part characteristics (YAML) and a fitted APC propeller JSON, then solve the 12-equation system from `Governing_Physics/`. This tool is for **manual part-selection studies** — it does not recommend or optimize parts.

```text
Manufacturer data → MATLAB fitting → prop JSON ─┐
                                                ├→ solver.py → top-speed estimate
Drone configuration YAML ───────────────────────┘
```

## Setup

```bash
cd Airspeed_Solver
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python solver.py configs/example_drone.yaml
```

Prints a report to the terminal and writes the same text to `results/<config_stem>_result.txt`.

```bash
python solver.py path/to/my_drone.yaml -o path/to/out.txt
```

## Inputs

### Drone YAML

Categories match the white paper input table:

| Section | Keys | Notes |
|---------|------|--------|
| `control` | `DC` | Throttle 0–1; use `1.0` for top speed |
| `vehicle` | `mass_kg`, `CdA`, `N` | Weight is `W = mass_kg * g` inside the solver |
| `environment` | `rho`, `mu` (or `p_Pa` + `T_K`) | Prefer `rho` |
| `motor` | `Kv`, `Rm`, `I0` | `Kt = Ke = 60/(2π Kv)` |
| `battery` | `Voc`, `Rbat`, `eta_bus`, optional `Ah_rating` | |
| `propeller` | `json` | Path to `*_analysed.json` (relative to the YAML or cwd) |

Prop diameter, pitch, 0.75R chord, and \(C_T(J,Re)\) / \(C_P(J,Re)\) **always** come from the JSON — not the YAML.

See `configs/example_drone.yaml` for a baseline and typical min/max ballparks on hard-to-know values (`Rm`, `I0`, `Rbat`, `eta_bus`, …).

### Prop JSON

Produced by `APC_Propeller_Tools/analyze_apc_prop.m` into `prop_data/<name>_analysed.json`.

## Units

| Quantity | Internal | Report |
|----------|----------|--------|
| Speed \(v\) | m/s | m/s and mph |
| Rotation \(n\) | **rev/s** | rev/s and RPM |
| Angles | rad | deg in report |
| Forces | N | N |

## Validity warnings

If the solved advance ratio \(J\) or Reynolds number \(Re\) leaves the propeller fit envelope, the solver still returns a speed and emits a soft warning with how far outside the bound you are, as a percent of the envelope width. The same warning appears in the terminal and the results file.

## API hook

```python
from solver import solve_from_files

result = solve_from_files("configs/example_drone.yaml")
print(result["derived"]["v_mph"], result["success"], result["warnings"])
```

## Out of scope (for now)

Batch sweeps, plots, presets, and optimization belong in a future analysis script — not in `solver.py`.
