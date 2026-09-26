"""
Propeller / motor / battery matching for a sized design point.

Propellers come from the database in propdata.py (UIUC wind-tunnel measurements
where available, APC computed data otherwise). A "generic" propeller is kept as
a fallback; it uses the sizing's own assumptions (figure of merit, propeller
efficiency) with a simple linear thrust line C_T(J) = C_T0 (1 - J/J0).

Motor Kv is chosen so the highest required RPM is reached at ~90 % throttle,
with the loaded motor turning ~80 % of Kv x V:  Kv = RPM_max / (0.72 V_pack).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from propdata import INCH, load_database
from uav_model import DesignPoint, Inputs, _level_flight, aero_model, isa_density, min_power_flight

KV_FACTOR = 0.72        # RPM_max = KV_FACTOR * Kv * V_pack
LOADED_CELL_SAG = 0.93  # loaded / nominal pack voltage
AUTO_D_TOL = 0.10       # auto choice: props within ±10 % of the sizing diameter


def speed_of_sound(h):
    return math.sqrt(1.4 * 287.05 * (288.15 - 0.0065 * h))


class GenericProp:
    """Linear thrust line + the sizing's FM / eta_prop (no data)."""
    sources = "generic model"
    key = "generic"

    def __init__(self, diameter, pitch_ratio, inp: Inputs):
        self.diameter, self.pitch_ratio, self.inp = diameter, pitch_ratio, inp
        self.d_in = diameter / INCH

    @property
    def label(self):
        return f'{self.d_in:.0f}" × {self.d_in * self.pitch_ratio:.1f}" (generic)'

    @property
    def max_rpm(self):
        return 150000.0 / self.d_in

    def _ct0(self):
        return 0.07 + 0.08 * self.pitch_ratio

    def _j0(self):
        return 1.1 * self.pitch_ratio

    def max_thrust(self, V, rho, rpm):
        n, D = rpm / 60, self.diameter
        return max(self._ct0() * (1 - V / (n * D) / self._j0()), 0.0) * rho * n * n * D**4

    def solve(self, T, V, rho):
        D, c0, jz = self.diameter, self._ct0(), self._j0()
        a, b = rho * D**4 * c0, -rho * D**3 * c0 * V / jz
        n = (-b + math.sqrt(b * b + 4 * a * T)) / (2 * a)
        if V > 0:
            return dict(rps=n, rpm=n * 60, J=V / (n * D), P_shaft=T * V / self.inp.eta_prop,
                        eta=self.inp.eta_prop, source="generic")
        A = math.pi * D * D / 4
        P = T**1.5 / (self.inp.figure_of_merit * math.sqrt(2 * rho * A))
        return dict(rps=n, rpm=n * 60, J=0.0, P_shaft=P, eta=math.nan,
                    FM=self.inp.figure_of_merit, source="generic")


@dataclass
class Condition:
    name: str
    V: float
    thrust: float          # N per motor
    rpm: float
    J: float
    eff: float             # propeller efficiency (forward) or figure of merit (hover)
    p_elec: float          # W per motor
    tip_mach: float
    source: str
    ok: bool = True        # reachable within the prop's RPM limit
    torque: float = math.nan   # N·m at the shaft


@dataclass
class MotorGroup:
    name: str
    count: int
    prop: object
    conditions: list[Condition]
    hover: bool
    forward: bool
    target_diameter: float = math.nan
    kv: float = math.nan
    rpm_max: float = math.nan
    p_max: float = math.nan
    p_rating: float = math.nan
    i_max: float = math.nan
    thrust_static_max: float = math.nan
    torque_max: float = math.nan           # N·m at the shaft, highest condition
    energy_wh: float = math.nan            # per motor over the mission
    eta_fwd: float = math.nan              # energy-weighted forward-flight efficiency
    fm_hover: float = math.nan

    @property
    def feasible(self):
        return all(c.ok for c in self.conditions) and math.isfinite(self.energy_wh)


@dataclass
class Propulsion:
    groups: list[MotorGroup]
    v_pack: float
    capacity_ah: float
    i_peak: float
    peak_case: str
    c_required: float
    warnings: list[str] = field(default_factory=list)
    chart: dict = field(default_factory=dict)


# ───────────────────────────────────────────────────────── conditions ──

def _forward_cases(inp: Inputs, dp: DesignPoint, count: int):
    """Forward-flight cases per motor: (name, V, thrust N, rho, altitude, seconds in mission)."""
    ae = aero_model(inp)
    ws, W = dp.ws, dp.derived["W"]
    rho_f, rho_c, rho_h = (isa_density(inp.field_alt), isa_density(inp.cruise_alt),
                           isa_density(inp.ceiling_alt))
    m = inp.constraint_margin
    cases = []
    c = min_power_flight(np.array([ws]), rho_f, ae, inp.speed_margin)
    Vc, twc = float(c["V"][0]), float(c["TW"][0])
    dh = max(inp.cruise_alt - inp.field_alt, 0.0)
    cases.append((f"Climb {inp.climb_rate:g} m/s", Vc, m * (twc + inp.climb_rate / Vc), rho_f,
                  inp.field_alt, dh / inp.climb_rate if inp.climb_rate > 0 else 0.0))
    c = min_power_flight(np.array([ws]), rho_h, ae, inp.speed_margin)
    Vh = float(c["V"][0])
    cases.append((f"Ceiling climb {inp.ceiling_climb_rate:g} m/s", Vh,
                  m * (float(c["TW"][0]) + inp.ceiling_climb_rate / Vh), rho_h, inp.ceiling_alt, 0.0))
    tw = float(_level_flight(ws, inp.v_cruise, rho_c, ae)[0])
    cases.append(("Cruise", inp.v_cruise, tw, rho_c, inp.cruise_alt,
                  inp.cruise_distance_km * 1000 / inp.v_cruise))
    tw = float(_level_flight(ws, inp.v_max, rho_c, ae)[0])
    cases.append(("Dash (with margin)", inp.v_max, m * tw, rho_c, inp.cruise_alt, 0.0))
    lo = dp.sizing.loiter
    if lo:
        cases.append(("Loiter", lo["V"], lo["TW"], rho_c, inp.cruise_alt, inp.loiter_time_h * 3600))
    return [(n, V, tw * W / count, rho, h, t) for n, V, tw, rho, h, t in cases]


def _evaluate_group(inp, dp, name, count, prop, hover, forward):
    W = dp.derived["W"]
    eta_me = inp.eta_motor * inp.eta_esc
    rho_f = isa_density(inp.field_alt)
    cases = []
    if hover:   # equal thrust on every hover rotor
        for label, tw, t in ((f"Hover, max (T/W {inp.hover_tw:.2f})", inp.hover_tw, 0.0),
                             ("Hover, 1 g", 1.0, inp.hover_time_s)):
            cases.append((label, 0.0, tw * W / inp.n_rotors, rho_f, inp.field_alt, t))
    if forward:
        cases += _forward_cases(inp, dp, count)

    conds, energy, work, shaft = [], 0.0, 0.0, 0.0
    fm = math.nan
    for label, V, T, rho, h, t in cases:
        r = prop.solve(T, V, rho)
        if r is None:
            conds.append(Condition(label, V, T, math.nan, math.nan, math.nan, math.nan, math.nan,
                                   "out of range", ok=False))
            if t > 0:
                energy = math.inf
            continue
        p_el = r["P_shaft"] / eta_me
        eff = r.get("FM", math.nan) if V == 0 else r["eta"]
        conds.append(Condition(label, V, T, r["rpm"], r["J"], eff, p_el,
                               math.pi * prop.diameter * r["rps"] / speed_of_sound(h), r["source"],
                               ok=r["rpm"] <= prop.max_rpm,
                               torque=r["P_shaft"] / (2 * math.pi * r["rps"])))
        energy += p_el * t / 3600.0
        if V > 0 and t > 0:
            work += T * V * t
            shaft += r["P_shaft"] * t
        if label == "Hover, 1 g":
            fm = eff
    g = MotorGroup(name, count, prop, conds, hover, forward)
    g.energy_wh = energy
    g.eta_fwd = work / shaft if shaft > 0 else math.nan
    g.fm_hover = fm
    return g


# ────────────────────────────────────────────────────────── group layout ──

def group_layout(inp: Inputs):
    """[(name, count, hover, forward, sizing diameter, pitch ratio for the generic prop)]."""
    out = []
    if inp.vtol_mode in ("tiltrotor", "tilt_quadplane"):
        out.append(("Tilting motor", inp.n_tilt, True, True, inp.d_tilt, inp.fwd_pitch_ratio))
        if inp.n_rotors > inp.n_tilt:
            out.append(("Lift-only motor", inp.n_rotors - inp.n_tilt, True, False,
                        inp.rotor_diameter, inp.lift_pitch_ratio))
    else:
        out.append(("Cruise motor", 1, False, True, inp.cruise_prop_diameter, inp.fwd_pitch_ratio))
        if inp.vtol_mode == "quadplane":
            out.append(("Lift-only motor", inp.n_rotors, True, False, inp.rotor_diameter, inp.lift_pitch_ratio))
    return out


def rank_props(inp: Inputs, dp: DesignPoint, group_name: str, max_d_tol: float | None = None):
    """Every database prop evaluated for one group, best mission energy first (feasible first)."""
    name, count, hover, forward, target, _ = {g[0]: g for g in group_layout(inp)}[group_name]
    res = []
    for e in load_database().values():
        if max_d_tol is not None and abs(e.diameter - target) > max_d_tol * target:
            continue
        g = _evaluate_group(inp, dp, name, count, e, hover, forward)
        g.target_diameter = target
        res.append(g)
    res.sort(key=lambda g: (not g.feasible, g.energy_wh if math.isfinite(g.energy_wh) else 1e12))
    return res


def _pick_prop(inp, dp, lay, choice):
    name, _, _, _, target, pd = lay
    db = load_database()
    if choice in db:
        return db[choice]
    if choice == "generic":
        return GenericProp(target, pd, inp)
    ranked = rank_props(inp, dp, name, AUTO_D_TOL)
    if ranked and ranked[0].feasible:
        return ranked[0].prop
    return GenericProp(target, pd, inp)


# ─────────────────────────────────────────────────────────────── matching ──

def match_propulsion(inp: Inputs, dp: DesignPoint, choices: dict | None = None) -> Propulsion | None:
    """choices: {group name: "auto" | "generic" | database key}."""
    if not dp.sizing.converged:
        return None
    choices = choices or {}
    rho_f = isa_density(inp.field_alt)
    v_pack = inp.battery_cells_s * inp.cell_voltage
    v_loaded = v_pack * LOADED_CELL_SAG
    rating = {name: p for name, _, p in dp.sizing.motor_groups}

    groups = []
    for lay in group_layout(inp):
        prop = _pick_prop(inp, dp, lay, choices.get(lay[0], "auto"))
        g = _evaluate_group(inp, dp, lay[0], lay[1], prop, lay[2], lay[3])
        g.target_diameter = lay[4]
        ok = [c for c in g.conditions if math.isfinite(c.rpm)]
        g.rpm_max = max(c.rpm for c in ok) if ok else math.nan
        g.p_max = max(c.p_elec for c in ok) if ok else math.nan
        g.torque_max = max(c.torque for c in ok) if ok else math.nan
        g.p_rating = max(g.p_max, rating.get(g.name, 0.0))
        g.kv = g.rpm_max / (KV_FACTOR * v_pack)
        g.i_max = g.p_rating / v_loaded
        rpm_full = KV_FACTOR / 0.9 * g.kv * v_pack
        # static thrust at full throttle: limited by RPM and by motor power (momentum theory)
        A1 = math.pi * prop.diameter**2 / 4
        fm = g.fm_hover if math.isfinite(g.fm_hover) else inp.figure_of_merit
        t_power = (g.p_rating * inp.eta_motor * inp.eta_esc * fm * math.sqrt(2 * rho_f * A1)) ** (2 / 3)
        g.thrust_static_max = min(prop.max_thrust(0.0, rho_f, rpm_full), t_power)
        groups.append(g)

    def case_current(key):
        tot = 0.0
        for g in groups:
            cs = [c for c in g.conditions if c.name.startswith(key) and math.isfinite(c.p_elec)]
            if cs:
                tot += g.count * max(c.p_elec for c in cs)
        return tot / v_loaded
    cases = {"hover at max thrust": case_current("Hover, max"),
             "dash": case_current("Dash"), "climb": case_current("Climb")}
    peak_case = max(cases, key=cases.get)
    i_peak = cases[peak_case]
    cap_ah = dp.sizing.battery_energy_wh / v_pack
    c_req = i_peak / cap_ah

    warns = []
    if c_req > inp.battery_max_c:
        warns.append(f"Pack must deliver {c_req:.0f} C in {peak_case}, above its {inp.battery_max_c:.0f} C "
                     "rating: use a higher-C pack, a bigger pack, or more cells in series.")
    for g in groups:
        p = g.prop
        bad = [c.name for c in g.conditions if not c.ok]
        if bad:
            warns.append(f"{g.name} ({p.label}): {', '.join(bad)} needs more than the prop's RPM limit "
                         f"({p.max_rpm:.0f} rpm = 150 000 / D[in]).")
        if isinstance(p, GenericProp):
            warns.append(f"{g.name}: no database prop within ±{AUTO_D_TOL:.0%} of {g.target_diameter:.2f} m "
                         "meets every condition; using the generic model. Pick one from the ranking.")
        elif abs(p.diameter - g.target_diameter) > 0.03 * g.target_diameter:
            warns.append(f"{g.name}: prop diameter {p.diameter:.3f} m differs from the {g.target_diameter:.3f} m "
                         "used in the sizing for hover power. Set the sizing diameter to match.")
        if g.forward and math.isfinite(g.eta_fwd) and g.eta_fwd < inp.eta_prop - 0.05:
            warns.append(f"{g.name}: this prop averages η = {g.eta_fwd:.2f} in forward flight, below the "
                         f"sizing's η_prop = {inp.eta_prop:.2f}, so the battery is undersized. "
                         "Use 'Apply prop efficiencies to sizing' or pick a better-matched prop.")
        if g.hover and math.isfinite(g.fm_hover) and g.fm_hover < inp.figure_of_merit - 0.05:
            warns.append(f"{g.name}: hover figure of merit {g.fm_hover:.2f} is below the sizing's "
                         f"{inp.figure_of_merit:.2f}.")
        mt = max((c.tip_mach for c in g.conditions if math.isfinite(c.tip_mach)), default=0)
        if mt > 0.6:
            warns.append(f"{g.name}: tip Mach {mt:.2f}; above ~0.6 noise and losses rise quickly.")
        if g.i_max > 60:
            warns.append(f"{g.name}: {g.i_max:.0f} A per motor is high for {inp.battery_cells_s}S; "
                         "consider more cells in series.")
        hov = [c.rpm for c in g.conditions if c.name == "Hover, 1 g"]
        if g.forward and hov and g.rpm_max > 2.2 * hov[0]:
            warns.append(f"{g.name}: forward flight needs {g.rpm_max / hov[0]:.1f}× the hover RPM, so the motor "
                         "hovers at low throttle and low efficiency.")

    fwd = next(g for g in groups if g.forward)
    ae = aero_model(inp)
    rho_c = isa_density(inp.cruise_alt)
    W = dp.derived["W"]
    Vs = np.linspace(max(0.8 * dp.derived["V_min"], 1.0), 1.6 * inp.v_max, 120)
    drag = np.array([float(_level_flight(dp.ws, v, rho_c, ae)[0]) for v in Vs]) * W
    rpm_full = KV_FACTOR / 0.9 * fwd.kv * v_pack
    t_rpm = np.array([fwd.prop.max_thrust(v, rho_c, rpm_full) for v in Vs]) * fwd.count
    eta = fwd.eta_fwd if math.isfinite(fwd.eta_fwd) else inp.eta_prop
    t_pow = fwd.count * fwd.p_rating * inp.eta_motor * inp.eta_esc * eta / Vs
    t_av = np.minimum(t_rpm, t_pow)
    okv = t_av >= drag
    v_top = float(Vs[okv][-1]) if okv.any() else math.nan
    chart = dict(V=Vs, drag=drag, t_avail=t_av, t_rpm=t_rpm, t_power=t_pow, v_top=v_top,
                 v_min=dp.derived["V_min"], v_cruise=inp.v_cruise, v_max=inp.v_max, group=fwd.name)
    return Propulsion(groups, v_pack, cap_ah, i_peak, peak_case, c_req, warns, chart)


def effective_efficiencies(p: Propulsion):
    """(eta_prop, figure_of_merit) implied by the chosen props, for feeding back into the sizing."""
    fwd = [g for g in p.groups if g.forward and math.isfinite(g.eta_fwd)]
    hov = [g for g in p.groups if g.hover and math.isfinite(g.fm_hover)]
    eta = fwd[0].eta_fwd if fwd else math.nan
    fm = (sum(g.fm_hover * g.count for g in hov) / sum(g.count for g in hov)) if hov else math.nan
    return eta, fm
