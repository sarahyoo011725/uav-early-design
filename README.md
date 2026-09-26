# UAV early design

A desktop tool for the **conceptual (early) design of small battery-electric
fixed-wing UAVs, including VTOL layouts**. From mission requirements it draws a
constraint diagram, lets you pick a design point, sizes the aircraft (mass,
wing, battery, motor power) and suggests real propellers, motor Kv and battery
configuration.

## What kind of UAV it is for

| Covered | Not covered |
|---|---|
| Battery-electric, propeller-driven | Fuel or hybrid power, jets |
| Fixed wing with one main wing, ≈1–25 kg MTOW, low subsonic (< ≈40 m/s), altitudes up to a few km | Compressibility, high-altitude or long-range fuel aircraft |
| **Tilt-rotor quadplane**: some rotors tilt forward for cruise, the rest only lift | Pure multicopters (no wing), helicopters |
| **Tiltrotor**: all rotors tilt | Tail-sitters, lift-plus-cruise with ducted fans |
| **Quadplane**: separate lift rotors + a cruise propeller | |
| **Conventional** (no VTOL) | |

The propeller database covers APC Thin Electric props from 12 to 27 in, the
usual range for this class.

## Install and run

```
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\python.exe app.py
```

## How to use it

1. **Enter the requirements** in the left panel: payload, endurance, hover
   time, speeds (max stall, cruise, dash), climb rates, ceiling, orbit radius,
   configuration and rotor layout. Hover over any field for a hint. Results
   update as you type.
2. **Constraint diagram tab**: every requirement is drawn as a curve of power-to-weight (P/W)
   and thrust-to-weight (T/W) against wing loading W/S. The green area meets all
   of them; the hatched area breaks a wing-loading limit (stall, orbit, cruise
   speed). **Click** to choose the design point, or use a button:
   - *Max-W/S corner*: the smallest wing that meets every requirement.
   - *Min-power point*: the lowest required P/W.
   - *Min-MTOW point*: the lightest aircraft (searches the feasible range).

   *Snap design point to envelope* picks the minimum P/W for the clicked W/S;
   untick it to choose P/W freely (extra power costs motor mass).
3. **Right panel**: the sized aircraft at that point. It shows mass and empty-mass
   breakdown, wing geometry, motor power by group, energy per mission leg, speeds,
   aerodynamics, Reynolds number and design notes.
4. **Sizing vs wing loading tab**: MTOW, battery, span and area across all W/S.
5. **Trade study tab**: pick any input and a range, then **Run**. Each value
   is re-sized and plotted as MTOW, masses, installed power and span. Use it to find the
   requirements that cost the most. Export to CSV.
6. **Propulsion tab**: choose a propeller per motor group (auto, a specific
   prop, or *Rank all props…* and double-click). It shows RPM, efficiency,
   power, torque and tip Mach per flight condition, the suggested motor Kv, current,
   battery mAh and C-rating, and thrust available vs drag. Then click
   **Apply prop efficiencies to sizing** to feed the real propeller efficiency,
   figure of merit and diameters back into the sizing; apply once or twice more
   until the numbers settle.
7. **File menu**: save/load inputs (JSON), export the diagram (PNG) and the
   report (HTML).

## Files

| File | Contents |
|---|---|
| `app.py` | The window (PySide6 + matplotlib) |
| `uav_model.py` | Atmosphere, aerodynamics, constraint analysis, sizing, trade study (no GUI, importable) |
| `propulsion.py` | Propeller / motor / battery matching for a design point |
| `propdata.py` | Propeller database loader and lookup |
| `data/uiuc_vol4/` | Measured propeller data (UIUC) |
| `data/apc/` | Computed propeller data (APC) |

Scripting example:

```python
from uav_model import Inputs, evaluate, min_mtow_ws
from propulsion import match_propulsion

inp = Inputs(payload_mass=1.2, loiter_time_h=1.5)
dp = evaluate(inp, min_mtow_ws(inp))
print(dp.sizing.mtow, dp.derived["span"])
print([(g.name, g.prop.label, round(g.kv)) for g in match_propulsion(inp, dp).groups])
```

---

## How it calculates

SI units throughout. Symbols: $W = m_0 g$ weight, $S$ wing area, $b$ span,
$A$ aspect ratio, $q=\tfrac12\rho V^2$ dynamic pressure, $\eta$ efficiencies.
Numbers in [brackets] refer to the sources at the bottom.

### 1. Atmosphere [1]

International Standard Atmosphere (troposphere):

$$T = 288.15 - 0.0065\,h,\qquad \rho = 1.225\,(1 - 2.25577\times10^{-5}\,h)^{4.2559}$$

$$\mu = \frac{1.458\times10^{-6}\,T^{1.5}}{T + 110.4}\ \ \text{(Sutherland)},\qquad a = \sqrt{1.4\cdot 287.05\,T}$$

Take-off/hover uses the field altitude, cruise and loiter the cruise altitude,
the ceiling constraint the ceiling altitude.

### 2. Aerodynamics [2, 3]

Winglets are counted as an effective aspect-ratio increase. This is Raymer's
relation for end plates, a conservative estimate for winglets [2, Ch. 4]:

$$A_\text{eff} = A\,(1 + 1.9\,h_w/b)$$

Oswald efficiency of the whole aircraft from Raymer's empirical fits
[2, Ch. 12], using the geometric $A$:

$$e_\text{straight} = 1.78\,(1 - 0.045A^{0.68}) - 0.64,\qquad
e_\text{swept} = 4.61\,(1 - 0.045A^{0.68})(\cos\Lambda_{LE})^{0.15} - 3.1$$

Raymer gives the swept-wing fit for $\Lambda_{LE} > 30°$ only. The tool uses
$e_\text{straight}$ up to 25°, $e_\text{swept}$ above 35°, and blends linearly
in between.

Cambered drag polar, with minimum drag $C_{D,min}$ at $C_{L,md}$ [2, Ch. 12; 3]:

$$C_D = C_{D,min} + K_1 (C_L - C_{L,md})^2 = C_{D0} + K_1 C_L^2 + K_2 C_L$$

$$K_1 = \frac{1}{\pi e A_\text{eff}},\quad K_2 = -2K_1 C_{L,md},\quad C_{D0} = C_{D,min} + K_1 C_{L,md}^2$$

Best lift-to-drag ratio and minimum power follow from $d(C_L/C_D)/dC_L = 0$ and
$d(C_D/C_L^{3/2})/dC_L = 0$:

$$C_{L,(L/D)max} = \sqrt{C_{D0}/K_1},\qquad (L/D)_{max} = \frac{1}{2\sqrt{C_{D0}K_1} + K_2}$$

$$C_{L,P_{min}} = \frac{K_2 + \sqrt{K_2^2 + 12 K_1 C_{D0}}}{2K_1}$$

Reynolds number on the mean chord $c = S/b$: $\;Re = \rho V c / \mu$.

### 3. Steady flight and constraints [2, Ch. 5 & 17; 3]

Steady flight at speed $V$ and load factor $n$ (thrust equals drag):

$$C_L = \frac{n\,(W/S)}{q},\qquad \frac{T}{W} = \frac{q\,C_D(C_L)}{W/S}$$

Level turn of radius $R$: $\;\tan\phi = V^2/(gR)$, $\;n = \sqrt{1 + (V^2/gR)^2}$.

Stall speed and minimum operating speed (margin $m$, default 1.2):

$$V_s = \sqrt{\frac{2\,(W/S)}{\rho\,C_{L,max}}},\qquad V \ge m\,V_s\sqrt{n}$$

Electrical power drawn from the battery per unit weight:

$$\frac{P}{W} = \frac{(T/W)\,V}{\eta_p\,\eta_m\,\eta_{esc}}$$

Each fixed-wing constraint is multiplied by the constraint margin $k$
(default 1.2). The requirements:

| Constraint | Model |
|---|---|
| Stall | $W/S \le \tfrac12\rho V_{s,max}^2 C_{L,max}$ (vertical line) |
| Cruise-speed limit | cruise speed $\ge m V_s$: $\;W/S \le \tfrac12\rho V_{cr}^2 C_{L,max}/m^2$ |
| Cruise, dash | $T/W$ at the required speed, $n = 1$ |
| Climb, ceiling | at the minimum-power speed ($\ge mV_s$): $\;T/W = D/W + \text{ROC}/V$ (small climb angle). For a propeller aircraft, best rate of climb is near the minimum-power speed [2, Ch. 17] |
| Loiter | minimum $P_{req}$ over $V \ge mV_s\sqrt n$, flown straight or as an orbit of radius $R$ |
| Orbit limit | slowest possible circle, from $V^2 = m^2V_s^2 n$: $\;V^2 = \dfrac{m^2V_s^2}{\sqrt{1-(m^2V_s^2/gR)^2}}$, possible only while $W/S < \dfrac{gR\rho C_{L,max}}{2m^2}$ |
| Hover | see §4 |

Minimum-power speeds are found numerically (a speed grid from the minimum
speed up), so the stall margin, the linear drag term and the turn load factor
are all respected. The envelope is the maximum over all curves. For a
tilt-rotor quadplane it is the requirement on the tilting motors, which also
carry their share of hover.

### 4. Hover [4]

Momentum (actuator-disk) theory with a figure of merit $FM$. Each of the $N$
rotors carries an equal share of the thrust:

$$T_i = \frac{(T/W)_h\,W}{N},\qquad P_i = \frac{T_i^{3/2}}{FM\,\sqrt{2\rho A_i}\;\eta_m\eta_{esc}},\qquad A_i = \frac{\pi D_i^2}{4}$$

$(T/W)_h$ is the hover thrust margin for motor sizing. Hover energy uses $(T/W)_h = 1$.
With unequal rotor sizes, the tilting rotors carry $n_t/N$ of the thrust and
$\sum_\text{tilt} D_i^{-1} / \sum_\text{all} D_i^{-1}$ of the power.

### 5. Motor power

| Group | Electrical power per motor |
|---|---|
| Tilting motor | $\max\!\left((P/W)_{design}\,W/n_t,\ P_{i,hover}\right)$ |
| Lift-only motor | $P_{i,hover}$ at $(T/W)_h$ |
| Cruise motor | $(P/W)_{design}\,W$ |

### 6. Mission energy and battery [2, electric-aircraft chapter; 5]

Energy is summed over the legs, each at its own electrical power:

$$E = (1 + r)\Big[P_{hover}t_{hover} + P_{climb}\tfrac{\Delta h}{\text{ROC}} + P_{cruise}\tfrac{d}{V_{cr}} + P_{loiter}t_{loiter} + P_{sys}\,t_{flight}\Big]$$

$r$ is the energy reserve, and $P_{sys}$ is the avionics and payload power over the whole
flight. The battery mass uses the pack specific energy $e_b$ and usable depth
of discharge:

$$m_b = \frac{E}{e_b \cdot DoD}$$

### 7. Mass sizing [2, Ch. 3]

Take-off mass closes the mass balance:

$$m_0 = \frac{m_{payload}}{1 - m_e/m_0 - m_b/m_0}$$

This is solved by fixed-point iteration with 50 % relaxation. The empty mass
$m_e$ (everything except battery and payload) comes from one of two models:

- **Regression** [2, Ch. 3]: $\;m_e/m_0 = A\,m_0^{C}$ (plus an optional wing mass $\rho_A S$).
- **Component build-up**:
  $$m_e = \rho_A S + f_{fus}\,m_0 + \frac{P_{installed}}{p_{motor}} + m_{prop}N_{props} + m_{tilt}N_{tilt} + m_{avionics}$$
  $\rho_A$ is wing mass per area, $f_{fus}$ the fuselage/tail/gear fraction,
  and $p_{motor}$ the motor + ESC specific power (kW/kg). Every term is an
  input; the defaults are engineering estimates for this class and should be
  calibrated against a real aircraft.

Geometry follows from the design point: $S = W/(W/S)$, $b = \sqrt{A S}$, $c = S/b$.
The *Min-MTOW point* searches the feasible W/S range on a grid and refines around the minimum.

### 8. Propellers [6, 7, 8]

Standard propeller coefficients ($n$ in rev/s, $D$ in m):

$$J = \frac{V}{nD},\quad C_T = \frac{T}{\rho n^2 D^4},\quad C_P = \frac{P}{\rho n^3 D^5},\quad \eta_p = \frac{C_T J}{C_P},\quad FM = \sqrt{\tfrac{2}{\pi}}\,\frac{C_T^{3/2}}{C_P}\ \ (J=0)$$

For each flight condition, the RPM that gives the required thrust is found by
bisection on $C_T(J,\text{RPM})\,\rho n^2 D^4 = T$. Then:

$$P_{shaft} = C_P\rho n^3D^5,\quad P_{elec} = \frac{P_{shaft}}{\eta_m\eta_{esc}},\quad Q = \frac{P_{shaft}}{2\pi n},\quad M_{tip} = \frac{\pi D n}{a}$$

**Data**: $C_T$ and $C_P$ come from the UIUC wind-tunnel measurements [7, 8]
wherever they cover the point. Otherwise they come from APC's computed data [9]. Past the
end of the measured range, the computed data is scaled to meet the measured
curve and the correction fades out over $\Delta J = 0.15$. UIUC runs at nearly
the same RPM are merged. Every result shows the source it used.

**Limits**: APC's maximum RPM for Thin Electric props is $150\,000 / D_{[in]}$ [10].

**Mission averages** fed back to the sizing: the energy-weighted forward
efficiency $\eta_p = \sum T V t / \sum P_{shaft} t$, and the hover figure of merit.

**Generic prop** (only when no data prop is chosen): linear thrust line
$C_T = C_{T0}(1 - J/J_0)$ with $C_{T0} = 0.07 + 0.08\,P/D$ and $J_0 = 1.1\,P/D$.
These are rough fits; power uses the sizing's $FM$ and $\eta_p$.

### 9. Motor and battery matching

These are rules of thumb, not published models:

- Kv puts the highest required RPM at about 90 % throttle, with a loaded motor
  turning about 80 % of Kv × V: $\;K_v = \text{RPM}_{max} / (0.72\,V_{pack})$.
- Pack voltage $V_{pack} = N_S \cdot V_{cell}$. Current is computed at 93 % of nominal voltage
  (sag): $\;I = P_{elec}/(0.93\,V_{pack})$.
- Capacity $= E_{battery}/V_{pack}$. Required C-rating $= I_{peak}/\text{capacity}$,
  where $I_{peak}$ is the worst of all hover motors at maximum thrust, dash, and climb.
- Static thrust at full throttle is the smaller of the RPM-limited thrust
  and the power-limited thrust $\big(P\,\eta_m\eta_{esc}\,FM\sqrt{2\rho A}\big)^{2/3}$.
- Top speed is where thrust available ($\min$ of the RPM limit and $\eta_p P / V$)
  meets drag.

Confirm motor choices with the manufacturer's data or a tool such as eCalc,
and compare the torque column with the motor's rated torque.

## Not modelled

Transition aerodynamics and transition energy (add it to hover time); trim
drag; the drag of stopped lift rotors and booms (include them in $C_{D,min}$);
wing Reynolds-number effects on $C_{L,max}$ and $C_{D,min}$ (use low-Re airfoil
data); structures and loads; stability, control and CG; wind; battery
temperature and ageing beyond the usable-DoD factor.

## Sources

1. NOAA, NASA, USAF, *U.S. Standard Atmosphere, 1976*, NASA-TM-X-74335. https://ntrs.nasa.gov/citations/19770009539
2. D. P. Raymer, *Aircraft Design: A Conceptual Approach*, 6th ed., AIAA Education Series, 2018 (Ch. 3 sizing, Ch. 4 wing tips, Ch. 5 T/W and W/S, Ch. 12 aerodynamics, Ch. 17 performance, electric-aircraft chapter). Equation numbers differ between editions. https://arc.aiaa.org/doi/book/10.2514/4.104909
3. S. Gudmundsson, *General Aviation Aircraft Design: Applied Methods and Procedures*, 2nd ed., Butterworth-Heinemann, 2022 (constraint analysis, cambered drag polar, performance, electric powertrains). https://www.sciencedirect.com/book/monograph/9780128184653/general-aviation-aircraft-design
4. J. G. Leishman, *Principles of Helicopter Aerodynamics*, 2nd ed., Cambridge University Press, 2016 (momentum theory, figure of merit). https://www.cambridge.org/9781107013353
5. M. Hepperle, *Electric Flight – Potential and Limitations*, NATO STO-MP-AVT-209, DLR, 2012. https://elib.dlr.de/78726/1/MP-AVT-209-09.pdf
6. J. B. Brandt and M. S. Selig, *Propeller Performance Data at Low Reynolds Numbers*, AIAA 2011-1255 (coefficient definitions, test method). https://m-selig.ae.illinois.edu/pubs/BrandtSelig-2011-AIAA-2011-1255-LRN-Propellers.pdf
7. O. D. Dantsker, M. Caccamo, R. W. Deters and M. S. Selig, *Performance Testing of APC Electric Fixed-Blade UAV Propellers*, AIAA 2022-4020. https://m-selig.ae.illinois.edu/pubs/DantskerCaccamoDetersSelig-2022-AIAA-Paper-2022-4020-APCEPropTestsVol4.pdf
8. UIUC Propeller Data Site, Volume 4 (data in `data/uiuc_vol4/`). https://m-selig.ae.illinois.edu/props/volume-4/propDB-volume-4.html
9. APC Propellers, performance data (computed with APC's vortex-theory software). The copies in `data/apc/` are APC's 2014 versions, taken from the MIT-licensed mirror https://github.com/JARC99/apc-prop-analyzer. Newer files from https://www.apcprop.com/technical-information/performance-data/ can be dropped into `data/apc/` with the same names.
10. APC Propellers, *RPM Limits*. https://www.apcprop.com/technical-information/rpm-limits/
