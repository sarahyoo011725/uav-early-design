"""
Early-design model for a small electric fixed-wing / VTOL UAV.

Pure computation, no GUI: the aerodynamic model, the constraint analysis
(T/W and P/W versus wing loading W/S) and the weight-sizing loop. Used by
app.py (interactive) and can be imported from scripts.

Units: SI throughout. W/S in N/m^2, P/W in W/N (electrical power drawn
from the battery per newton of weight), masses in kg.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict, fields

import numpy as np

G = 9.80665  # m/s^2


def isa_density(h):
    """ISA density [kg/m^3] at geometric altitude h [m] (troposphere)."""
    return 1.225 * (1.0 - 2.25577e-5 * h) ** 4.2559


def air_viscosity(h):
    """Dynamic viscosity [Pa s] at altitude h (ISA temperature, Sutherland's law)."""
    T = 288.15 - 0.0065 * h
    return 1.458e-6 * T**1.5 / (T + 110.4)


# ─────────────────────────────────────────────────────────────── inputs ──

@dataclass
class Inputs:
    # Weights
    payload_mass: float = 0.9          # kg
    mtow_guess: float = 5.0            # kg, starting point of the sizing loop
    empty_model: str = "components"    # "components" | "regression"
    # regression model: everything except battery and payload
    empty_A: float = 0.699             # We/W0 = A * W0^C  (W0 in kg)
    empty_C: float = -0.051
    wing_areal_mass: float = 0.0       # kg/m^2 of wing area added on top of the regression (0 = off)
    # component model
    comp_wing_areal: float = 1.5       # kg/m^2: wing structure, servos, winglets
    comp_fuselage_frac: float = 0.20   # fuselage, tail, landing gear as a fraction of MTOW
    comp_motor_kw_per_kg: float = 3.0  # motor + ESC specific power, kW/kg
    comp_per_rotor_mass: float = 0.10  # kg per prop: propeller, mount, share of boom
    comp_tilt_mech_mass: float = 0.06  # kg per tilting rotor: tilt servo and hinge
    comp_avionics_mass: float = 0.40   # kg fixed: autopilot, GPS, radios, wiring

    # Aerodynamics
    aspect_ratio: float = 10.0
    winglet_ratio: float = 0.075       # winglet height / span  (0.15 m / 2 m)
    sweep_deg: float = 15.0            # leading-edge sweep
    cd_min: float = 0.04               # minimum drag coefficient of the whole aircraft
    cl_min_drag: float = 0.2           # CL at which CD = CD_min (cambered polar)
    cl_max: float = 1.6

    # Propulsion & battery
    eta_prop: float = 0.75
    eta_motor: float = 0.85
    eta_esc: float = 0.98
    battery_wh_per_kg: float = 150.0
    battery_dod: float = 0.80          # usable fraction of capacity
    energy_reserve: float = 0.0        # extra energy fraction kept in reserve
    systems_power_w: float = 15.0      # W drawn all flight by avionics + payload (camera, gimbal, radios)

    # VTOL
    vtol_mode: str = "tilt_quadplane"  # "tilt_quadplane" | "tiltrotor" | "quadplane" | "none"
    n_rotors: int = 4                  # all rotors used in hover
    n_tilt_rotors: int = 2             # tilt_quadplane: rotors that tilt forward for cruise
    rotor_diameter: float = 0.6        # m, lift rotors (all rotors unless tilt_rotor_diameter is set)
    tilt_rotor_diameter: float = 0.0   # m, tilting rotors; 0 = same as rotor_diameter
    figure_of_merit: float = 0.65      # hover efficiency of the rotors
    hover_tw: float = 1.2              # hover thrust / weight (control margin)
    hover_time_s: float = 120.0        # total hover time (take-off + landing)

    # Mission & performance requirements
    field_alt: float = 0.0             # m, take-off site
    cruise_alt: float = 100.0          # m, cruise / loiter altitude
    ceiling_alt: float = 3000.0        # m, service ceiling
    v_stall_max: float = 12.0          # m/s, highest acceptable stall speed
    speed_margin: float = 1.2          # min flying speed = margin * stall speed
    v_cruise: float = 18.0             # m/s
    v_max: float = 25.0                # m/s, dash speed
    cruise_distance_km: float = 0.0    # km flown at v_cruise (energy budget)
    climb_rate: float = 3.0            # m/s at field altitude
    ceiling_climb_rate: float = 0.5    # m/s at the ceiling
    loiter_time_h: float = 1.0         # h
    loiter_mode: str = "orbit"         # "orbit" (circle of turn_radius) | "straight"
    turn_radius: float = 60.0          # m
    constraint_margin: float = 1.2     # multiplies every fixed-wing T/W and P/W

    # Propulsion matching (propulsion.py; does not change the sizing)
    battery_cells_s: int = 4           # cells in series
    cell_voltage: float = 3.7          # nominal V per cell (LiPo 3.7, Li-ion 3.6)
    battery_max_c: float = 20.0        # continuous discharge rating of the pack [C]
    fwd_pitch_ratio: float = 0.55      # pitch/diameter of the tilting (or cruise) props
    lift_pitch_ratio: float = 0.45     # pitch/diameter of lift-only props
    cruise_prop_diameter: float = 0.30 # m, separate cruise prop (quadplane / conventional)

    # Plot range
    ws_min: float = 5.0
    ws_max: float = 300.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    @property
    def eta_total(self):
        return self.eta_prop * self.eta_motor * self.eta_esc

    @property
    def n_tilt(self):
        """Rotors that also give cruise thrust."""
        if self.vtol_mode == "tiltrotor":
            return self.n_rotors
        if self.vtol_mode == "tilt_quadplane":
            return min(max(int(self.n_tilt_rotors), 1), self.n_rotors)
        return 0

    @property
    def hover_share(self):
        """Fraction of hover thrust carried by the motors that also cruise."""
        return self.n_tilt / self.n_rotors if self.vtol_mode != "none" else 0.0

    @property
    def d_tilt(self):
        return self.tilt_rotor_diameter if self.tilt_rotor_diameter > 0 else self.rotor_diameter

    def rotor_diameters(self):
        """Diameter of every hover rotor (tilting ones first)."""
        if self.vtol_mode == "none":
            return []
        if self.vtol_mode == "quadplane":
            return [self.rotor_diameter] * self.n_rotors
        return [self.d_tilt] * self.n_tilt + [self.rotor_diameter] * (self.n_rotors - self.n_tilt)

    @property
    def disk_area(self):
        return sum(math.pi * d * d / 4 for d in self.rotor_diameters())

    @property
    def hover_power_share(self):
        """Fraction of hover POWER on the tilting rotors (equal thrust per rotor, P_i ∝ 1/D_i)."""
        ds = self.rotor_diameters()
        if not ds or self.n_tilt == 0:
            return 0.0
        return sum(1 / d for d in ds[: self.n_tilt]) / sum(1 / d for d in ds)


# ─────────────────────────────────────────────────────────── aerodynamics ──

@dataclass
class Aero:
    e: float            # Oswald efficiency
    ar_eff: float       # effective aspect ratio (with winglets)
    k1: float           # CD = CD0 + k1*CL^2 + k2*CL
    k2: float
    cd0: float
    cl_ld_max: float    # CL for best L/D
    ld_max: float
    cl_min_power: float # CL for minimum power (max endurance)
    ld_min_power: float
    cl_max: float

    def cd(self, cl):
        return self.cd0 + self.k1 * cl**2 + self.k2 * cl


def aero_model(inp: Inputs) -> Aero:
    A = inp.aspect_ratio
    # Winglets: effective AR increase (Raymer)  A_eff = A (1 + 1.9 h/b)
    ar_eff = A * (1.0 + 1.9 * inp.winglet_ratio)

    # Oswald factor (whole aircraft), Raymer eq. 12.48 for straight wings and
    # eq. 12.49 for sweep > 30 deg, geometric AR. The swept-wing fit comes from
    # high-sweep, low-AR aircraft and gives far lower e at moderate sweep, so it
    # is only used above 30 deg (blended over 25-35 deg to avoid a jump).
    base = 1.0 - 0.045 * A**0.68
    e_straight = 1.78 * base - 0.64
    e_swept = 4.61 * base * math.cos(math.radians(inp.sweep_deg))**0.15 - 3.1
    t = min(max((inp.sweep_deg - 25.0) / 10.0, 0.0), 1.0)
    e = (1.0 - t) * e_straight + t * e_swept

    k1 = 1.0 / (math.pi * e * ar_eff)
    # Cambered polar CD = CDmin + k1 (CL - CLmd)^2 expanded to CD0 + k1 CL^2 + k2 CL
    k2 = -2.0 * k1 * inp.cl_min_drag
    cd0 = inp.cd_min + k1 * inp.cl_min_drag**2

    # max L/D:  d(CL/CD)/dCL = 0  ->  CL = sqrt(CD0/k1)  (still true with k2)
    cl_ld = math.sqrt(cd0 / k1)
    ld_max = 1.0 / (2.0 * math.sqrt(cd0 * k1) + k2)
    # min power: d(CD/CL^1.5)/dCL = 0  ->  k1 CL^2 - k2 CL - 3 CD0 = 0
    cl_mp = (k2 + math.sqrt(k2**2 + 12.0 * k1 * cd0)) / (2.0 * k1)
    ld_mp = cl_mp / (cd0 + k1 * cl_mp**2 + k2 * cl_mp)
    return Aero(e, ar_eff, k1, k2, cd0, cl_ld, ld_max, cl_mp, ld_mp, inp.cl_max)


# ─────────────────────────────────────────────── level-flight building blocks ──

def _level_flight(ws, V, rho, ae: Aero, n=1.0):
    """T/W and CL for steady flight at speed V with load factor n."""
    q = 0.5 * rho * V**2
    cl = n * ws / q
    tw = q * ae.cd(cl) / ws
    return tw, cl


def stall_speed(ws, rho, cl_max):
    return np.sqrt(2.0 * ws / (rho * cl_max))


def min_orbit_speed(ws, rho, ae: Aero, margin, radius):
    """
    Slowest speed for a level circle of given radius while staying `margin`
    above the stall speed *in the turn*:
        V^2 = m^2 Vs^2 n,   n = sqrt(1 + (V^2/(gR))^2)
    closed form  V^2 = m^2 Vs^2 / sqrt(1 - (m^2 Vs^2 / gR)^2).
    Returns NaN where the circle cannot be flown (wing loading too high).
    """
    a = margin**2 * stall_speed(ws, rho, ae.cl_max) ** 2
    x = a / (G * radius)
    with np.errstate(invalid="ignore", divide="ignore"):
        v2 = np.where(x < 1.0, a / np.sqrt(1.0 - x**2), np.nan)
    return np.sqrt(v2)


def min_power_flight(ws, rho, ae: Aero, margin, radius=None, n_grid=400):
    """
    Minimum-power steady flight (straight, or a level circle of `radius`)
    at speeds >= margin * stall speed. Found numerically so that the stall
    margin, the linear polar term and the turn load factor are all respected.
    Returns dict of arrays: V, TW, PW_aero (= P_req / W  [W/N]), CL, n.
    """
    ws = np.atleast_1d(np.asarray(ws, float))
    if radius is None:
        v_lo = margin * stall_speed(ws, rho, ae.cl_max)
    else:
        v_lo = min_orbit_speed(ws, rho, ae, margin, radius)
    V = v_lo[:, None] * np.linspace(1.0, 3.0, n_grid)[None, :]
    if radius is None:
        n = np.ones_like(V)
    else:
        n = np.sqrt(1.0 + (V**2 / (G * radius)) ** 2)
    tw, cl = _level_flight(ws[:, None], V, rho, ae, n)
    pw = tw * V
    pw_filled = np.where(np.isfinite(pw), pw, np.inf)
    i = np.argmin(pw_filled, axis=1)
    r = np.arange(len(ws))
    ok = np.isfinite(v_lo)
    pick = lambda a: np.where(ok, a[r, i], np.nan)
    return dict(V=pick(V), TW=pick(tw), PW_aero=pick(pw), CL=pick(cl), n=pick(n))


# ───────────────────────────────────────────────────────────── constraints ──

@dataclass
class Curve:
    key: str
    label: str
    TW: np.ndarray
    PW: np.ndarray          # electrical W/N, margin included
    color: str
    in_envelope: bool = True
    dashed: bool = False


@dataclass
class Limit:
    key: str
    label: str
    ws_max: float
    color: str


@dataclass
class Constraints:
    ws: np.ndarray
    curves: list[Curve]
    limits: list[Limit]
    TW_env: np.ndarray
    PW_env: np.ndarray
    ws_limit: float         # max feasible W/S (min over limits, clipped to range)
    hover_PW: float         # NaN if no VTOL

    def env_at(self, ws):
        return (float(np.interp(ws, self.ws, self.TW_env)),
                float(np.interp(ws, self.ws, self.PW_env)))


PALETTE = {
    "hover": "#E07B39", "cruise": "#6B8CBA", "vmax": "#3B4FA0",
    "climb": "#2BA57A", "ceiling": "#B07D2A", "loiter": "#C2477E",
    "stall": "#D62828", "orbit": "#7E57C2",
}


def hover_rotor_power(inp: Inputs, W, thrust_ratio, diameter):
    """Electrical hover power [W] of one rotor carrying an equal share of thrust_ratio * W.
    Momentum theory with figure of merit:  P = T^1.5 / (FM sqrt(2 rho A))."""
    rho = isa_density(inp.field_alt)
    T = thrust_ratio * W / inp.n_rotors
    A = math.pi * diameter**2 / 4
    return T**1.5 / (inp.figure_of_merit * math.sqrt(2.0 * rho * A) * inp.eta_motor * inp.eta_esc)


def hover_pw(inp: Inputs, W, thrust_ratio):
    """Electrical hover P/W [W/N], summed over all rotors (equal thrust per rotor).
    With equal rotors this is (T/W)^1.5 sqrt(W / (2 rho A_total)) / FM."""
    return sum(hover_rotor_power(inp, W, thrust_ratio, d) for d in inp.rotor_diameters()) / W


def speed_pw(inp, ae, ws, V, rho, margin, n=1.0):
    """T/W, P/W at a required fixed speed. Masked where V < margin*Vs(n)."""
    tw, cl = _level_flight(ws, V, rho, ae, n)
    tw = margin * tw
    tw = np.where(cl <= ae.cl_max / inp.speed_margin**2, tw, np.nan)
    return tw, tw * V / inp.eta_total


def fw_segments(inp: Inputs, ae: Aero, ws, margin):
    """All fixed-wing constraint curves at wing loadings ws. margin=1 for energy use."""
    rho_f = isa_density(inp.field_alt)
    rho_c = isa_density(inp.cruise_alt)
    rho_h = isa_density(inp.ceiling_alt)
    eta = inp.eta_total
    out = {}

    tw, pw = speed_pw(inp, ae, ws, inp.v_cruise, rho_c, margin)
    out["cruise"] = (tw, pw)
    tw, pw = speed_pw(inp, ae, ws, inp.v_max, rho_c, margin)
    out["vmax"] = (tw, pw)

    # Best rate of climb for a propeller aircraft ~ minimum-power speed
    c = min_power_flight(ws, rho_f, ae, inp.speed_margin)
    tw = margin * (c["TW"] + inp.climb_rate / c["V"])
    out["climb"] = (tw, tw * c["V"] / eta)

    c = min_power_flight(ws, rho_h, ae, inp.speed_margin)
    tw = margin * (c["TW"] + inp.ceiling_climb_rate / c["V"])
    out["ceiling"] = (tw, tw * c["V"] / eta)

    radius = inp.turn_radius if inp.loiter_mode == "orbit" else None
    lo = min_power_flight(ws, rho_c, ae, inp.speed_margin, radius)
    tw = margin * lo["TW"]
    out["loiter"] = (tw, tw * lo["V"] / eta)
    out["_loiter_state"] = lo
    return out


def constraint_analysis(inp: Inputs, W: float, n_points=600) -> Constraints:
    ae = aero_model(inp)
    ws = np.linspace(inp.ws_min, inp.ws_max, n_points)
    seg = fw_segments(inp, ae, ws, inp.constraint_margin)
    orbit = inp.loiter_mode == "orbit"

    curves = [
        Curve("cruise", f"Cruise {inp.v_cruise:g} m/s", *seg["cruise"], PALETTE["cruise"]),
        Curve("vmax", f"Dash {inp.v_max:g} m/s", *seg["vmax"], PALETTE["vmax"]),
        Curve("climb", f"Climb {inp.climb_rate:g} m/s", *seg["climb"], PALETTE["climb"]),
        Curve("ceiling", f"Ceiling {inp.ceiling_alt:g} m", *seg["ceiling"], PALETTE["ceiling"]),
        Curve("loiter", f"Loiter ({'orbit R=%g m' % inp.turn_radius if orbit else 'straight'})",
              *seg["loiter"], PALETTE["loiter"]),
    ]

    hpw = math.nan
    if inp.vtol_mode != "none":
        hpw = hover_pw(inp, W, inp.hover_tw)
        share = inp.hover_share
        # The envelope is the requirement on the motors that give cruise thrust.
        # In hover they carry `share` of the weight (equal thrust per rotor).
        if share > 0:
            label = "Hover" if share == 1 else f"Hover share of {inp.n_tilt} tilt rotors"
            curves.insert(0, Curve("hover_tilt", label,
                                   np.full_like(ws, inp.hover_tw * share),
                                   np.full_like(ws, hpw * inp.hover_power_share),
                                   PALETTE["hover"]))
        if share < 1:
            label = "Hover, lift rotors (separate)" if share == 0 else "Hover, all rotors"
            curves.insert(0, Curve("hover", label,
                                   np.full_like(ws, inp.hover_tw), np.full_like(ws, hpw),
                                   PALETTE["hover"], in_envelope=False, dashed=True))

    rho_f = isa_density(inp.field_alt)
    limits = [Limit("stall", f"Stall ≤ {inp.v_stall_max:g} m/s",
                    0.5 * rho_f * inp.v_stall_max**2 * inp.cl_max, PALETTE["stall"])]
    if orbit:
        rho_c = isa_density(inp.cruise_alt)
        # circle of radius R flyable only while m^2 Vs^2 < gR
        ws_turn = G * inp.turn_radius * rho_c * inp.cl_max / (2.0 * inp.speed_margin**2)
        limits.append(Limit("orbit", f"Orbit R = {inp.turn_radius:g} m", ws_turn, PALETTE["orbit"]))

    # cruise speed must stay >= margin * stall speed at cruise altitude
    rho_c = isa_density(inp.cruise_alt)
    limits.append(Limit("cruise_speed", f"Cruise ≥ {inp.speed_margin:g}·Vs",
                        0.5 * rho_c * inp.v_cruise**2 * inp.cl_max / inp.speed_margin**2,
                        PALETTE["cruise"]))

    TW_env = np.zeros_like(ws)
    PW_env = np.zeros_like(ws)
    for c in curves:
        if c.in_envelope:
            TW_env = np.fmax(TW_env, c.TW)
            PW_env = np.fmax(PW_env, c.PW)

    ws_limit = min([inp.ws_max] + [l.ws_max for l in limits])
    return Constraints(ws, curves, limits, TW_env, PW_env, ws_limit, hpw)


# ────────────────────────────────────────────────────────────────── sizing ──

@dataclass
class Sizing:
    converged: bool
    message: str
    mtow: float = math.nan
    empty_mass: float = math.nan
    battery_mass: float = math.nan
    payload_mass: float = math.nan
    empty_fraction: float = math.nan
    battery_fraction: float = math.nan
    battery_energy_wh: float = math.nan     # installed (nameplate) energy
    energy_breakdown_wh: dict = field(default_factory=dict)
    empty_breakdown_kg: dict = field(default_factory=dict)
    motor_groups: list = field(default_factory=list)   # (name, count, W per motor)
    P_installed: float = math.nan
    loiter: dict = field(default_factory=dict)


def mission_energy_per_weight(inp: Inputs, ae: Aero, ws: float, W: float):
    """Battery energy drawn per newton of weight [J/N] for each mission leg."""
    wsa = np.array([ws], float)
    seg = fw_segments(inp, ae, wsa, margin=1.0)
    legs = {}
    if inp.vtol_mode != "none":
        legs["Hover"] = hover_pw(inp, W, 1.0) * inp.hover_time_s
    dh = max(inp.cruise_alt - inp.field_alt, 0.0)
    if dh > 0:
        # wing-borne climb to cruise altitude (after transition for VTOL)
        legs["Climb"] = float(seg["climb"][1][0]) * dh / inp.climb_rate
    if inp.cruise_distance_km > 0:
        legs["Cruise"] = float(seg["cruise"][1][0]) * inp.cruise_distance_km * 1000 / inp.v_cruise
    legs["Loiter"] = float(seg["loiter"][1][0]) * inp.loiter_time_h * 3600.0
    if inp.systems_power_w > 0:
        t_flight = (inp.loiter_time_h * 3600.0 + inp.cruise_distance_km * 1000 / inp.v_cruise
                    + (dh / inp.climb_rate if dh > 0 and inp.climb_rate > 0 else 0.0)
                    + (inp.hover_time_s if inp.vtol_mode != "none" else 0.0))
        legs["Avionics & payload"] = inp.systems_power_w * t_flight / W
    lo = {k: float(v[0]) for k, v in seg["_loiter_state"].items()}
    return legs, lo


def motor_sizing(inp: Inputs, ae: Aero, ws: float, W: float, pw: float | None = None):
    """
    Electrical power of each motor group at wing loading ws and weight W.
    pw is the design P/W of the motors that give cruise thrust; None = the
    minimum required (the envelope). Returns (groups, pw_req, pw_fixed_wing, pw_hover)
    with groups = [(name, count, W per motor)].
    """
    seg = fw_segments(inp, ae, np.array([ws], float), inp.constraint_margin)
    fw_vals = [float(seg[k][1][0]) for k in ("cruise", "vmax", "climb", "ceiling", "loiter")]
    fw_vals = [v for v in fw_vals if math.isfinite(v)]
    pw_fw = max(fw_vals) if fw_vals else math.nan
    pw_hov = hover_pw(inp, W, inp.hover_tw) if inp.vtol_mode != "none" else 0.0
    pw_req = max(pw_fw, pw_hov * inp.hover_power_share)
    if pw is None:
        pw = pw_req
    groups = []
    if inp.vtol_mode in ("tiltrotor", "tilt_quadplane"):
        p_tilt = hover_rotor_power(inp, W, inp.hover_tw, inp.d_tilt)
        groups.append(("Tilting motor", inp.n_tilt, max(pw * W / inp.n_tilt, p_tilt)))
        if inp.n_rotors > inp.n_tilt:
            groups.append(("Lift-only motor", inp.n_rotors - inp.n_tilt,
                           hover_rotor_power(inp, W, inp.hover_tw, inp.rotor_diameter)))
    else:
        groups.append(("Cruise motor", 1, pw * W))
        if inp.vtol_mode == "quadplane":
            groups.append(("Lift-only motor", inp.n_rotors,
                           hover_rotor_power(inp, W, inp.hover_tw, inp.rotor_diameter)))
    return groups, pw_req, pw_fw, pw_hov


def empty_mass(inp: Inputs, m0: float, ws: float, groups) -> dict:
    """Empty-mass breakdown [kg] (everything except battery and payload)."""
    S = m0 * G / ws
    if inp.empty_model == "regression":
        out = {"Empty (A·W0^C regression)": inp.empty_A * m0 ** (1.0 + inp.empty_C)}
        if inp.wing_areal_mass > 0:
            out["Wing add-on"] = inp.wing_areal_mass * S
        return out
    p_total = sum(n * p for _, n, p in groups)
    n_props = sum(n for _, n, _ in groups)
    out = {
        "Wing": inp.comp_wing_areal * S,
        "Fuselage, tail, gear": inp.comp_fuselage_frac * m0,
        "Motors + ESCs": p_total / (inp.comp_motor_kw_per_kg * 1000.0),
        "Props, mounts, booms": inp.comp_per_rotor_mass * n_props,
    }
    if inp.n_tilt > 0:
        out["Tilt mechanisms"] = inp.comp_tilt_mech_mass * inp.n_tilt
    out["Avionics & wiring"] = inp.comp_avionics_mass
    return out


def size_aircraft(inp: Inputs, ws: float, pw: float | None = None,
                  tol=1e-6, max_iter=300) -> Sizing:
    """
    Converge take-off mass for a chosen wing loading (and optionally P/W):
        m0 = m_payload / (1 - We/W0 - Wb/W0)
    Empty mass comes from the regression or the component model (which needs
    the installed motor power); the battery fraction is recomputed every
    iteration from the mission energy (depends on W/S and, for hover, on W).
    """
    ae = aero_model(inp)
    m0 = inp.mtow_guess
    e_spec = inp.battery_wh_per_kg * 3600.0 * inp.battery_dod  # usable J/kg
    for _ in range(max_iter):
        W = m0 * G
        legs, lo = mission_energy_per_weight(inp, ae, ws, W)
        groups = motor_sizing(inp, ae, ws, W, pw)[0]
        if not all(math.isfinite(v) for v in legs.values()) or not all(
                math.isfinite(p) for _, _, p in groups):
            return Sizing(False, "Mission cannot be flown at this wing loading "
                                 "(a required speed or orbit is below the stall margin).")
        e_per_w = sum(legs.values()) * (1.0 + inp.energy_reserve)
        f_b = G * e_per_w / e_spec
        f_e = sum(empty_mass(inp, m0, ws, groups).values()) / m0
        denom = 1.0 - f_e - f_b
        if denom <= 0.02:
            return Sizing(False, f"Sizing does not close: empty ({f_e:.2f}) + battery "
                                 f"({f_b:.2f}) fractions leave no room for payload.",
                          empty_fraction=f_e, battery_fraction=f_b)
        m_new = inp.payload_mass / denom
        if abs(m_new - m0) < tol * m0:
            m0 = m_new
            break
        m0 = 0.5 * m0 + 0.5 * m_new
    else:
        return Sizing(False, "Sizing loop did not converge.")

    W = m0 * G
    legs, lo = mission_energy_per_weight(inp, ae, ws, W)
    groups = motor_sizing(inp, ae, ws, W, pw)[0]
    eb = empty_mass(inp, m0, ws, groups)
    f_e = sum(eb.values()) / m0
    e_per_w = sum(legs.values()) * (1.0 + inp.energy_reserve)
    f_b = G * e_per_w / e_spec
    breakdown = {k: v * W / 3600.0 for k, v in legs.items()}  # Wh drawn
    if inp.energy_reserve > 0:
        breakdown["Reserve"] = sum(breakdown.values()) * inp.energy_reserve
    return Sizing(True, "Converged", mtow=m0, empty_mass=f_e * m0,
                  battery_mass=f_b * m0, payload_mass=inp.payload_mass,
                  empty_fraction=f_e, battery_fraction=f_b,
                  battery_energy_wh=f_b * m0 * inp.battery_wh_per_kg,
                  energy_breakdown_wh=breakdown, empty_breakdown_kg=eb,
                  motor_groups=groups, P_installed=sum(n * p for _, n, p in groups),
                  loiter=lo)


# ──────────────────────────────────────────────────────────── design point ──

@dataclass
class DesignPoint:
    ws: float
    pw: float
    feasible: bool
    reasons: list[str]
    sizing: Sizing
    constraints: Constraints
    aero: Aero
    derived: dict


def suggested_points(c: Constraints):
    """(ws, pw) of the max-wing-loading corner and of the minimum-power point."""
    ok = (c.ws <= c.ws_limit) & np.isfinite(c.PW_env)
    if not ok.any():
        return None, None
    ws_ok, pw_ok = c.ws[ok], c.PW_env[ok]
    corner = (float(ws_ok[-1]), float(pw_ok[-1]))
    # on a flat envelope (e.g. hover-dominated) take the highest W/S that is
    # within 0.5 % of the minimum: same power, smaller wing
    i = int(np.nonzero(pw_ok <= pw_ok.min() * 1.005)[0][-1])
    minp = (float(ws_ok[i]), float(pw_ok[i]))
    return corner, minp


def evaluate(inp: Inputs, ws: float, pw: float | None = None) -> DesignPoint:
    """Size the aircraft at W/S = ws and evaluate the design point.
    pw=None means 'exactly on the envelope' (the minimum required P/W)."""
    ae = aero_model(inp)
    sz = size_aircraft(inp, ws, pw)
    W = (sz.mtow if sz.converged else inp.mtow_guess) * G
    c = constraint_analysis(inp, W)
    tw_req, pw_req = c.env_at(ws)
    if pw is None:
        pw = pw_req

    reasons = []
    for l in c.limits:
        if ws > l.ws_max:
            reasons.append(f"W/S above {l.label} limit ({l.ws_max:.0f} N/m²)")
    if not math.isfinite(pw_req):
        reasons.append("A required speed is below the stall margin at this W/S")
    elif pw < pw_req * (1 - 1e-6):
        reasons.append(f"P/W below requirement ({pw_req:.2f} W/N)")
    if not sz.converged:
        reasons.append(sz.message)

    d: dict = {"TW_req": tw_req, "PW_req": pw_req}
    d["V_stall"] = float(stall_speed(ws, isa_density(inp.field_alt), inp.cl_max))
    d["V_min"] = inp.speed_margin * d["V_stall"]
    d["q_cruise"] = 0.5 * isa_density(inp.cruise_alt) * inp.v_cruise**2
    d["CL_cruise"] = ws / d["q_cruise"]
    d["LD_cruise"] = d["CL_cruise"] / ae.cd(d["CL_cruise"])
    rho_c = isa_density(inp.cruise_alt)
    d["V_best_LD"] = math.sqrt(2 * ws / (rho_c * ae.cl_ld_max))
    d["V_min_power"] = math.sqrt(2 * ws / (rho_c * ae.cl_min_power))
    if sz.converged:
        W = sz.mtow * G
        S = W / ws
        b = math.sqrt(inp.aspect_ratio * S)
        d.update(W=W, S=S, span=b, chord=S / b, winglet_h=inp.winglet_ratio * b)
        _, _, pw_fw, pw_hov = motor_sizing(inp, ae, ws, W, pw)
        d["P_fixed_wing"] = pw_fw * W
        if inp.vtol_mode != "none":
            d["P_hover"] = pw_hov * W
            d["P_hover_per_rotor"] = d["P_hover"] / inp.n_rotors
            d["disk_loading"] = W / inp.disk_area
            d["rotor_span_ratio"] = sum(inp.rotor_diameters()) / 2 / b
        d["motor_groups"] = sz.motor_groups
        d["P_installed"] = sz.P_installed
        mu = air_viscosity(inp.cruise_alt)
        d["Re_min_speed"] = rho_c * d["V_min"] * d["chord"] / mu
        d["Re_cruise"] = rho_c * inp.v_cruise * d["chord"] / mu
    feasible = not reasons
    return DesignPoint(ws, pw, feasible, reasons, sz, c, ae, d)


def sizing_sweep(inp: Inputs, ws_values):
    """Rows of (ws, MTOW, battery, wing area, span, installed power), NaN where sizing fails."""
    rows = []
    for ws in ws_values:
        s = size_aircraft(inp, float(ws))
        if s.converged:
            S = s.mtow * G / ws
            rows.append((ws, s.mtow, s.battery_mass, S, math.sqrt(inp.aspect_ratio * S), s.P_installed))
        else:
            rows.append((ws, math.nan, math.nan, math.nan, math.nan, math.nan))
    return np.array(rows)


def feasible_ws_range(inp: Inputs):
    """(lowest, highest) wing loading that meets every limit, from a quick constraint pass."""
    c = constraint_analysis(inp, inp.mtow_guess * G, n_points=300)
    ok = (c.ws <= c.ws_limit) & np.isfinite(c.PW_env)
    if not ok.any():
        return None
    return float(c.ws[ok][0]), float(c.ws[ok][-1])


def min_mtow_ws(inp: Inputs, n=40):
    """Wing loading with the lowest converged MTOW inside the feasible range."""
    rng = feasible_ws_range(inp)
    if rng is None:
        return None
    lo, hi = rng
    ws = np.linspace(lo, hi, n)
    m = sizing_sweep(inp, ws)[:, 1]
    if not np.isfinite(m).any():
        return None
    i = int(np.nanargmin(m))
    # refine between the neighbours
    a, b = ws[max(i - 1, 0)], ws[min(i + 1, n - 1)]
    ws2 = np.linspace(a, b, 15)
    m2 = sizing_sweep(inp, ws2)[:, 1]
    return float(ws2[int(np.nanargmin(m2))]) if np.isfinite(m2).any() else float(ws[i])


TRADE_COLUMNS = ("value", "ws", "mtow", "battery", "empty", "P_installed", "span", "feasible")


def trade_study(inp: Inputs, param: str, values, ws_rule: str = "corner", ws_fixed: float | None = None):
    """
    Vary one input and re-size the aircraft for each value.
    ws_rule: "fixed" (keep ws_fixed), "corner" (max feasible W/S), "min_mtow".
    Returns an array with columns TRADE_COLUMNS.
    """
    rows = []
    for v in values:
        d = inp.to_dict()
        d[param] = type(getattr(inp, param))(v)
        p = Inputs.from_dict(d)
        if ws_rule == "fixed" and ws_fixed is not None:
            ws = ws_fixed
        elif ws_rule == "min_mtow":
            ws = min_mtow_ws(p)
        else:
            rng = feasible_ws_range(p)
            ws = rng[1] if rng else None
        if ws is None:
            rows.append((v,) + (math.nan,) * 6 + (0.0,))
            continue
        dp = evaluate(p, ws)
        s = dp.sizing
        if s.converged:
            rows.append((v, ws, s.mtow, s.battery_mass, s.empty_mass, s.P_installed,
                         dp.derived["span"], float(dp.feasible)))
        else:
            rows.append((v, ws) + (math.nan,) * 5 + (0.0,))
    return np.array(rows, float)
