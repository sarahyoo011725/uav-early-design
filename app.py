"""
UAV early-design studio — interactive constraint diagram and sizing.

Run:  venv\\Scripts\\python.exe app.py

Edit requirements on the left, click the P/W or T/W chart to choose a
design point (wing loading), and read the sized aircraft on the right.
"""
from __future__ import annotations

import json
import math
import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser,
    QVBoxLayout, QWidget,
)

import matplotlib
matplotlib.use("QtAgg")
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from propdata import load_database
from propulsion import GenericProp, effective_efficiencies, group_layout, match_propulsion, rank_props
from uav_model import (G, TRADE_COLUMNS, Inputs, evaluate, min_mtow_ws, sizing_sweep,
                       suggested_points, trade_study)

# ───────────────────────────────────────────────────────── input panel spec ──
# (key, label, unit, min, max, step, decimals, tooltip)
FIELDS = [
    ("Mission", [
        ("payload_mass", "Payload", "kg", 0.01, 200, 0.1, 2, "Mass carried (camera, sensors…)."),
        ("loiter_time_h", "Endurance / loiter", "h", 0, 24, 0.1, 2, "Time spent loitering."),
        ("cruise_distance_km", "Cruise distance", "km", 0, 2000, 1, 1, "Distance flown at cruise speed (energy only)."),
        ("hover_time_s", "Hover time (total)", "s", 0, 3600, 10, 0, "Take-off + landing hover time."),
        ("field_alt", "Take-off altitude", "m", 0, 5000, 50, 0, ""),
        ("cruise_alt", "Cruise / loiter altitude", "m", 0, 6000, 50, 0, ""),
        ("energy_reserve", "Energy reserve", "–", 0, 1, 0.05, 2, "Extra energy fraction on top of the mission."),
    ]),
    ("Performance requirements", [
        ("v_stall_max", "Max stall speed", "m/s", 3, 60, 0.5, 1, "Sets the maximum wing loading."),
        ("speed_margin", "Min speed / stall speed", "–", 1.0, 2.0, 0.05, 2, "Operating margin above stall (1.2 typical)."),
        ("v_cruise", "Cruise speed", "m/s", 3, 100, 0.5, 1, ""),
        ("v_max", "Dash (max) speed", "m/s", 3, 150, 0.5, 1, ""),
        ("climb_rate", "Rate of climb", "m/s", 0, 30, 0.1, 1, "At take-off altitude, best-climb speed."),
        ("ceiling_alt", "Service ceiling", "m", 0, 10000, 100, 0, ""),
        ("ceiling_climb_rate", "Climb rate at ceiling", "m/s", 0, 10, 0.1, 1, "0.5 m/s = classic service-ceiling definition."),
        ("turn_radius", "Loiter orbit radius", "m", 5, 2000, 5, 0, "Used when loiter mode is 'orbit'."),
        ("constraint_margin", "Constraint margin", "–", 1.0, 2.0, 0.05, 2, "Multiplies every fixed-wing T/W and P/W."),
    ]),
    ("Aerodynamics", [
        ("aspect_ratio", "Aspect ratio", "–", 2, 40, 0.5, 1, ""),
        ("winglet_ratio", "Winglet height / span", "–", 0, 0.3, 0.005, 3, "A_eff = A (1 + 1.9 h/b)"),
        ("sweep_deg", "Sweep (LE)", "deg", 0, 60, 1, 0, ""),
        ("cd_min", "CD min", "–", 0.005, 0.2, 0.002, 4, "Minimum drag coefficient of the whole aircraft."),
        ("cl_min_drag", "CL at CD min", "–", -0.5, 1.0, 0.05, 2, "Cambered polar offset."),
        ("cl_max", "CL max", "–", 0.5, 3.0, 0.05, 2, ""),
    ]),
    ("Propulsion & battery", [
        ("eta_prop", "Propeller efficiency", "–", 0.3, 0.95, 0.01, 2, ""),
        ("eta_motor", "Motor efficiency", "–", 0.3, 0.99, 0.01, 2, ""),
        ("eta_esc", "ESC / electronics eff.", "–", 0.5, 1.0, 0.01, 2, ""),
        ("battery_wh_per_kg", "Battery specific energy", "Wh/kg", 50, 500, 5, 0, "Pack level (≈150 LiPo, ≈200–250 Li-ion)."),
        ("battery_dod", "Usable depth of discharge", "–", 0.3, 1.0, 0.05, 2, ""),
        ("systems_power_w", "Avionics + payload power", "W", 0, 1000, 1, 0,
         "Electrical power drawn for the whole flight by autopilot, GPS, radios, camera, gimbal."),
    ]),
    ("VTOL", [
        ("n_rotors", "Rotors used in hover", "", 1, 16, 1, 0, "All rotors that lift in hover."),
        ("n_tilt_rotors", "…of which tilt for cruise", "", 1, 16, 1, 0,
         "Tilt-rotor quadplane only: rotors that tilt forward and give cruise thrust."),
        ("rotor_diameter", "Rotor diameter (lift)", "m", 0.05, 3, 0.01, 3,
         "Lift rotors; also the tilting rotors unless set separately below."),
        ("tilt_rotor_diameter", "Tilting rotor diameter", "m", 0.0, 3, 0.01, 3,
         "0 = same as the lift rotors. Hover assumes equal thrust on every rotor."),
        ("figure_of_merit", "Figure of merit", "–", 0.3, 0.85, 0.01, 2, "Hover efficiency (0.6–0.75 small rotors)."),
        ("hover_tw", "Hover T/W", "–", 1.0, 3.0, 0.05, 2, "Thrust margin for control in hover."),
    ]),
    ("Propulsion matching", [
        ("battery_cells_s", "Battery cells in series", "S", 1, 24, 1, 0, "Pack voltage = S × cell voltage."),
        ("cell_voltage", "Cell nominal voltage", "V", 2.0, 4.2, 0.05, 2, "LiPo 3.7 V, Li-ion 3.6 V."),
        ("battery_max_c", "Pack continuous rating", "C", 1, 150, 1, 0,
         "Max continuous discharge. LiPo 20–50 C, Li-ion packs 3–10 C."),
        ("fwd_pitch_ratio", "Generic prop pitch ÷ diam. (fwd)", "", 0.2, 1.5, 0.01, 2,
         "Pitch-to-diameter ratio of the props that give cruise thrust."),
        ("lift_pitch_ratio", "Generic prop pitch ÷ diam. (lift)", "", 0.2, 1.5, 0.01, 2,
         "Only used when the Propulsion tab uses the generic prop model."),
        ("cruise_prop_diameter", "Cruise prop diameter", "m", 0.05, 2, 0.01, 2,
         "Separate cruise prop (quadplane / conventional only)."),
    ]),
    ("Weights", [
        ("mtow_guess", "MTOW rough guess", "kg", 0.1, 500, 0.5, 2, "Only used for the first constraint pass (hover line) before sizing."),
    ]),
    ("Empty weight: components", [
        ("comp_wing_areal", "Wing areal mass", "kg/m²", 0.1, 20, 0.1, 2,
         "Wing structure incl. servos and winglets, per m² of wing area.\n"
         "≈1.0 foam/film, 1.3–2.0 foam-core glass/carbon, 2–3 molded composite."),
        ("comp_fuselage_frac", "Fuselage + tail + gear", "×MTOW", 0.0, 0.6, 0.01, 2,
         "Fuselage, tail surfaces and landing gear as a fraction of MTOW."),
        ("comp_motor_kw_per_kg", "Motor + ESC specific power", "kW/kg", 0.5, 15, 0.1, 1,
         "Installed electrical power per kg of motor+ESC (≈2.5–4 for hobby outrunners + ESCs)."),
        ("comp_per_rotor_mass", "Per prop: prop, mount, boom", "kg", 0, 2, 0.01, 3,
         "Counted for every motor (lift, tilting and cruise)."),
        ("comp_tilt_mech_mass", "Per tilt mechanism", "kg", 0, 1, 0.01, 3, "Tilt servo + hinge, per tilting rotor."),
        ("comp_avionics_mass", "Avionics & wiring", "kg", 0, 20, 0.05, 2, "Autopilot, GPS, radios, wiring (fixed)."),
    ]),
    ("Empty weight: regression", [
        ("empty_A", "Regression A", "", 0.05, 3, 0.01, 3, "We/W0 = A · W0^C  (W0 in kg, everything except battery & payload)"),
        ("empty_C", "Regression C", "", -1, 1, 0.005, 3, ""),
        ("wing_areal_mass", "Wing add-on", "kg/m²", 0, 20, 0.1, 2,
         "Optional: adds ρA·S of wing mass so bigger wings cost weight. 0 = off."),
    ]),
    ("Plot range", [
        ("ws_min", "W/S min", "N/m²", 1, 2000, 5, 0, ""),
        ("ws_max", "W/S max", "N/m²", 10, 5000, 10, 0, ""),
    ]),
]
COMBOS = {
    "vtol_mode": ("VTOL", "Configuration", [
        ("tilt_quadplane", "Tilt-rotor quadplane (some rotors tilt)"),
        ("tiltrotor", "Tiltrotor (all rotors tilt)"),
        ("quadplane", "Quadplane (separate lift rotors)"),
        ("none", "Conventional (no VTOL)"),
    ]),
    "empty_model": ("Weights", "Empty-weight model", [
        ("components", "Component build-up"),
        ("regression", "Regression  A·W0^C"),
    ]),
    "loiter_mode": ("Mission", "Loiter mode", [
        ("orbit", "Orbit (circle of radius R)"),
        ("straight", "Straight & level"),
    ]),
}

INT_FIELDS = ("n_rotors", "n_tilt_rotors", "battery_cells_s")

C_ENV = "#1A3A5C"
C_OK = "#2BA57A"
C_BAD = "#D62828"
C_DP = "#E63946"


class Canvas(FigureCanvasQTAgg):
    def __init__(self, fig):
        super().__init__(fig)
        self.setMinimumSize(500, 350)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UAV Early Design — Constraint Diagram")
        self.resize(1600, 900)
        self.inp = Inputs()
        self.design_ws: float | None = None   # None → auto (max W/S corner)
        self.design_pw: float | None = None   # None → on the envelope
        self.widgets = {}
        self.last = None

        self._timer = QTimer(self, singleShot=True, interval=150)
        self._timer.timeout.connect(self.recompute)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_inputs())
        splitter.addWidget(self._build_plots())
        splitter.addWidget(self._build_results())
        splitter.setSizes([360, 880, 380])
        self.setCentralWidget(splitter)
        self._build_menu()
        self.recompute()

    # ─────────────────────────────────────────────────────────── layout ──
    def _build_inputs(self):
        inner = QWidget()
        lay = QVBoxLayout(inner)
        groups = {}
        self.boxes = {}
        for gname, items in FIELDS:
            box = QGroupBox(gname)
            self.boxes[gname] = box
            form = QFormLayout(box)
            form.setLabelAlignment(Qt.AlignRight)
            groups[gname] = form
            for key, label, unit, lo, hi, step, dec, tip in items:
                if key in INT_FIELDS:
                    w = QSpinBox()
                    w.setRange(int(lo), int(hi))
                else:
                    w = QDoubleSpinBox()
                    w.setRange(lo, hi)
                    w.setDecimals(dec)
                    w.setSingleStep(step)
                w.setValue(getattr(self.inp, key))
                w.setKeyboardTracking(False)
                if unit:
                    w.setSuffix(f"  {unit}")
                w.setToolTip(tip)
                w.valueChanged.connect(self._changed)
                self.widgets[key] = w
                form.addRow(label, w)
            lay.addWidget(box)
        for key, (gname, label, opts) in COMBOS.items():
            cb = QComboBox()
            for val, text in opts:
                cb.addItem(text, val)
            cb.setCurrentIndex([v for v, _ in opts].index(getattr(self.inp, key)))
            cb.currentIndexChanged.connect(self._changed)
            self.widgets[key] = cb
            groups[gname].insertRow(0, label, cb)
        lay.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        scroll.setMinimumWidth(330)
        return scroll

    def _build_plots(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)

        bar = QHBoxLayout()
        self.snap = QCheckBox("Snap design point to envelope")
        self.snap.setChecked(True)
        self.snap.setToolTip("Checked: click picks W/S and P/W = minimum required.\n"
                             "Unchecked: click picks both W/S and P/W freely.")
        self.snap.toggled.connect(self._snap_toggled)
        b1 = QPushButton("Max-W/S corner")
        b1.setToolTip("Smallest wing that meets every requirement.")
        b1.clicked.connect(lambda: self._pick_suggestion(0))
        b2 = QPushButton("Min-power point")
        b2.setToolTip("Lowest required P/W inside the feasible W/S range.")
        b2.clicked.connect(lambda: self._pick_suggestion(1))
        b3 = QPushButton("Min-MTOW point")
        b3.setToolTip("Wing loading with the lowest take-off mass (sizes the aircraft across the feasible range).")
        b3.clicked.connect(self._pick_min_mtow)
        bar.addWidget(QLabel("Click a chart to choose the design point."))
        bar.addStretch()
        bar.addWidget(self.snap)
        for b in (b1, b2, b3):
            bar.addWidget(b)
        lay.addLayout(bar)

        self.tabs = QTabWidget()
        self.fig = Figure(figsize=(10, 5), dpi=100, facecolor="#F4F6F9")
        self.ax_pw, self.ax_tw = self.fig.subplots(1, 2)
        self.canvas = Canvas(self.fig)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        t1 = QWidget()
        l1 = QVBoxLayout(t1)
        l1.addWidget(NavigationToolbar2QT(self.canvas, t1))
        l1.addWidget(self.canvas)
        self.tabs.addTab(t1, "Constraint diagram")

        self.fig2 = Figure(figsize=(10, 5), dpi=100, facecolor="#F4F6F9")
        self.ax_m, self.ax_b = self.fig2.subplots(1, 2)
        self.canvas2 = Canvas(self.fig2)
        self.canvas2.mpl_connect("button_press_event", self._on_click)
        t2 = QWidget()
        l2 = QVBoxLayout(t2)
        l2.addWidget(NavigationToolbar2QT(self.canvas2, t2))
        l2.addWidget(self.canvas2)
        self.tabs.addTab(t2, "Sizing vs wing loading")
        self.tabs.addTab(self._build_trade_tab(), "Trade study")
        self.tabs.addTab(self._build_prop_tab(), "Propulsion")
        self.tabs.currentChanged.connect(lambda _: self.recompute())
        lay.addWidget(self.tabs)
        return w

    def _build_prop_tab(self):
        outer = QWidget()
        ol = QVBoxLayout(outer)
        ol.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self.prop_combos: dict[str, QComboBox] = {}
        self.prop_choice: dict[str, str] = {}
        self._prop_layout_names: tuple = ()
        self.prop_combo_box = QHBoxLayout()
        bar.addLayout(self.prop_combo_box)
        bar.addStretch()
        rank = QPushButton("Rank all props…")
        rank.setToolTip("Evaluate every propeller in the database for a motor group and pick one.")
        rank.clicked.connect(self.show_prop_ranking)
        apply_ = QPushButton("Apply prop efficiencies to sizing")
        apply_.setToolTip("Copy the chosen props' forward-flight η, hover figure of merit and equivalent\n"
                          "rotor diameter into the sizing inputs, then re-size.")
        apply_.clicked.connect(self.apply_prop_efficiencies)
        bar.addWidget(rank)
        bar.addWidget(apply_)
        ol.addLayout(bar)

        t = QSplitter(Qt.Horizontal)
        self.prop_report = QTextBrowser()
        self.prop_report.setMinimumWidth(460)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.fig4 = Figure(figsize=(5, 5), dpi=100, facecolor="#F4F6F9")
        self.ax_thr = self.fig4.subplots()
        self.canvas4 = Canvas(self.fig4)
        self.canvas4.setMinimumSize(300, 300)
        rl.addWidget(NavigationToolbar2QT(self.canvas4, right))
        rl.addWidget(self.canvas4)
        t.addWidget(self.prop_report)
        t.addWidget(right)
        t.setSizes([560, 400])
        ol.addWidget(t)
        self.last_prop = None
        return outer

    def _sync_prop_combos(self):
        """One prop selector per motor group; rebuilt when the configuration changes."""
        names = tuple(g[0] for g in group_layout(self.inp))
        if names == self._prop_layout_names:
            return
        self._prop_layout_names = names
        while self.prop_combo_box.count():
            item = self.prop_combo_box.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.prop_combos.clear()
        db = load_database()
        for name in names:
            cb = QComboBox()
            cb.addItem("Auto: best within ±10 % of sizing diameter", "auto")
            cb.addItem("Generic model (sizing's FM and η)", "generic")
            for key, e in db.items():
                cb.addItem(f"{e.label}   [{e.sources}]", key)
            cur = self.prop_choice.get(name, "auto")
            cb.setCurrentIndex(max(0, cb.findData(cur)))
            cb.currentIndexChanged.connect(lambda _, n=name, c=cb: self._prop_combo_changed(n, c))
            self.prop_combo_box.addWidget(QLabel(f"{name}:"))
            self.prop_combo_box.addWidget(cb)
            self.prop_combos[name] = cb

    def _prop_combo_changed(self, name, cb):
        self.prop_choice[name] = cb.currentData()
        self.recompute()

    def show_prop_ranking(self):
        if self.last is None or not self.last.sizing.converged:
            return
        names = [g[0] for g in group_layout(self.inp)]
        dlg = QDialog(self)
        dlg.setWindowTitle("Propeller ranking")
        dlg.resize(900, 600)
        lay = QVBoxLayout(dlg)
        top = QHBoxLayout()
        grp = QComboBox()
        for n in names:
            grp.addItem(n)
        top.addWidget(QLabel("Motor group:"))
        top.addWidget(grp)
        top.addWidget(QLabel("Sorted by electrical energy per motor over the mission. "
                             "Double-click a row to use that prop."))
        top.addStretch()
        lay.addLayout(top)
        table = QTableWidget()
        cols = ["Prop", "Data", "D vs sizing", "Mission Wh/motor", "Hover FM", "Fwd η", "Max RPM",
                "RPM limit", "Max W/motor", "OK"]
        table.setColumnCount(len(cols))
        table.setHorizontalHeaderLabels(cols)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        lay.addWidget(table)
        ranked = []

        def fill():
            nonlocal ranked
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                ranked = rank_props(self.inp, self.last, grp.currentText())
            finally:
                QApplication.restoreOverrideCursor()
            table.setRowCount(len(ranked))
            for i, g in enumerate(ranked):
                p = g.prop
                rpm = max((c.rpm for c in g.conditions if math.isfinite(c.rpm)), default=math.nan)
                pw = max((c.p_elec for c in g.conditions if math.isfinite(c.p_elec)), default=math.nan)
                vals = [p.label, p.sources, f"{(p.diameter / g.target_diameter - 1) * 100:+.0f} %",
                        f"{g.energy_wh:.1f}", f"{g.fm_hover:.2f}", f"{g.eta_fwd:.2f}",
                        f"{rpm:.0f}", f"{p.max_rpm:.0f}", f"{pw:.0f}", "yes" if g.feasible else "no"]
                for j, v in enumerate(vals):
                    item = QTableWidgetItem(v.replace("nan", "–"))
                    if not g.feasible:
                        item.setForeground(Qt.red)
                    table.setItem(i, j, item)
            table.resizeColumnsToContents()

        def pick(row, _col):
            key = ranked[row].prop.key
            name = grp.currentText()
            self.prop_choice[name] = key
            cb = self.prop_combos.get(name)
            if cb is not None:
                cb.blockSignals(True)
                cb.setCurrentIndex(max(0, cb.findData(key)))
                cb.blockSignals(False)
            dlg.accept()
            self.recompute()

        grp.currentIndexChanged.connect(lambda _: fill())
        table.cellDoubleClicked.connect(pick)
        fill()
        dlg.exec()

    def apply_prop_efficiencies(self):
        p = self.last_prop
        if p is None:
            return
        eta, fm = effective_efficiencies(p)
        hover_groups = [g for g in p.groups if g.hover]
        msg = []
        if math.isfinite(eta):
            self.widgets["eta_prop"].setValue(round(eta, 2))
            msg.append(f"η_prop = {eta:.2f}")
        if math.isfinite(fm):
            self.widgets["figure_of_merit"].setValue(round(fm, 2))
            msg.append(f"figure of merit = {fm:.2f}")
        for g in hover_groups:
            d = round(g.prop.diameter, 3)
            if g.name == "Tilting motor" and self.inp.vtol_mode == "tilt_quadplane":
                self.widgets["tilt_rotor_diameter"].setValue(d)
                msg.append(f"tilting rotor diameter = {d:.3f} m")
            else:
                self.widgets["rotor_diameter"].setValue(d)
                msg.append(f"rotor diameter = {d:.3f} m")
        fwd = next((g for g in p.groups if g.forward and not g.hover), None)
        if fwd is not None:
            self.widgets["cruise_prop_diameter"].setValue(round(fwd.prop.diameter, 3))
        # keep the chosen props instead of letting "auto" re-pick after the diameter change
        for g in p.groups:
            if not isinstance(g.prop, GenericProp):
                self.prop_choice[g.name] = g.prop.key
                cb = self.prop_combos.get(g.name)
                if cb is not None:
                    cb.blockSignals(True)
                    cb.setCurrentIndex(max(0, cb.findData(g.prop.key)))
                    cb.blockSignals(False)
        self.recompute()
        self.statusBar().showMessage("Applied to sizing: " + ", ".join(msg) +
                                     ". The aircraft was re-sized; apply again to converge.", 15000)

    def _draw_propulsion(self, dp):
        inp = self.inp
        self._sync_prop_combos()
        choices = {n: cb.currentData() for n, cb in self.prop_combos.items()}
        p = match_propulsion(inp, dp, choices)
        self.last_prop = p
        if p is None:
            self.prop_report.setHtml("<p style='color:#D62828'>Sizing did not converge, so there is "
                                     "nothing to match propellers to.</p>")
            self.ax_thr.clear()
            self.canvas4.draw_idle()
            return
        th = "style='background:#E8EEF5;padding:3px'"
        html = ["<div style='font-family:Segoe UI;font-size:10pt'>"]
        html.append(f"<p><b>Battery:</b> {inp.battery_cells_s}S, {p.v_pack:.1f} V nominal, "
                    f"<b>{p.capacity_ah * 1000:.0f} mAh</b> ({dp.sizing.battery_energy_wh:.0f} Wh). "
                    f"Peak current {p.i_peak:.1f} A in {p.peak_case} → needs <b>{p.c_required:.1f} C</b> "
                    f"(pack rated {inp.battery_max_c:.0f} C).</p>")
        eta, fm = effective_efficiencies(p)
        html.append(f"<p><b>Chosen props vs sizing assumptions:</b> forward η "
                    f"{'–' if not math.isfinite(eta) else f'{eta:.2f}'} (sizing {inp.eta_prop:.2f}), "
                    f"hover FM {'–' if not math.isfinite(fm) else f'{fm:.2f}'} "
                    f"(sizing {inp.figure_of_merit:.2f}).</p>")
        for g in p.groups:
            pr = g.prop
            html.append(f"<h3 style='margin:10px 0 2px 0;color:#1A3A5C'>{g.name} × {g.count}</h3>")
            src = pr.sources + (f" (APC file dated {pr.apc_date})" if getattr(pr, "apc_date", "") else "")
            summary = [
                ("Propeller", f"{pr.label}  (D {pr.diameter:.3f} m)"),
                ("Data", src),
                ("Suggested motor Kv", f"≈ {g.kv:.0f} rpm/V on {inp.battery_cells_s}S"),
                ("Power rating", f"{g.p_rating:.0f} W electrical"),
                ("Max current", f"{g.i_max:.1f} A per motor"),
                ("Max shaft torque", f"{g.torque_max:.2f} N·m (compare with the motor's rated torque)"),
                ("Static thrust at full throttle", f"{g.thrust_static_max:.1f} N ({g.thrust_static_max / G:.2f} kgf)"),
                ("Mission energy", f"{g.energy_wh:.1f} Wh per motor"),
            ]
            html.append("<table cellspacing='0' cellpadding='2'>" + "".join(
                f"<tr><td>{a}</td><td><b>{b}</b></td></tr>" for a, b in summary) + "</table>")

            def fmt(v, spec):
                return "–" if not math.isfinite(v) else format(v, spec)
            rows = "".join(
                f"<tr style='color:{'#000' if c.ok else '#D62828'}'><td>{c.name}</td>"
                f"<td align='right'>{c.V:.1f}</td><td align='right'>{c.thrust:.2f}</td>"
                f"<td align='right'>{fmt(c.rpm, '.0f')}</td><td align='right'>{fmt(c.J, '.2f')}</td>"
                f"<td align='right'>{fmt(c.eff, '.2f')}</td><td align='right'>{fmt(c.p_elec, '.0f')}</td>"
                f"<td align='right'>{fmt(c.torque, '.2f')}</td>"
                f"<td align='right'>{fmt(c.tip_mach, '.2f')}</td><td>{c.source}</td></tr>"
                for c in g.conditions)
            html.append(f"<table cellspacing='0' cellpadding='3' width='100%' style='margin-top:4px'>"
                        f"<tr><th {th}>Condition</th><th {th}>V m/s</th><th {th}>Thrust N</th><th {th}>RPM</th>"
                        f"<th {th}>J</th><th {th}>η / FM</th><th {th}>P W</th><th {th}>Torque N·m</th><th {th}>Tip M</th>"
                        f"<th {th}>Data</th></tr>{rows}</table>")
        if p.warnings:
            html.append("<h3 style='margin:10px 0 2px 0;color:#D62828'>Check</h3><ul style='margin:2px'>"
                        + "".join(f"<li>{w}</li>" for w in p.warnings) + "</ul>")
        html.append("<p style='color:#666;font-size:9pt'>Prop data: UIUC Propeller Data Site vol. 4 "
                    "(wind-tunnel measured, 2022) where it covers the point, otherwise APC's published "
                    "computed data. η / FM: propeller efficiency in forward flight, figure of merit in hover. "
                    "Kv: highest RPM reached at ~90 % throttle. Confirm motors with the maker's data or eCalc."
                    "</p></div>")
        self.prop_report.setHtml("".join(html))

        ch = p.chart
        ax = self.ax_thr
        ax.clear()
        ax.set_facecolor("#FAFBFC")
        ax.grid(color="#DDE3EC", linewidth=0.7, linestyle="--")
        ax.set_title(f"Thrust available vs required\n{ch['group']}s, cruise altitude",
                     fontsize=10, fontweight="semibold", color="#1A2A3A")
        ax.plot(ch["V"], ch["drag"], color=C_ENV, lw=2.2, label="Required (drag, level flight)")
        ax.plot(ch["V"], ch["t_avail"], color="#E07B39", lw=2.2, label="Available (full throttle)")
        ax.plot(ch["V"], ch["t_rpm"], color="#E07B39", lw=1, ls=":", label="RPM limit")
        ax.plot(ch["V"], ch["t_power"], color="#E07B39", lw=1, ls="--", label="Power limit")
        dmax = np.nanmax(ch["drag"])
        ax.set_ylim(0, 1.2 * max(dmax, min(np.nanmax(ch["t_avail"]), 3 * dmax)))
        marks = ((ch["v_min"], "Min", "#D62828"), (ch["v_cruise"], "Cruise", "#6B8CBA"),
                 (ch["v_max"], "Dash", "#3B4FA0"), (ch["v_top"], "Top speed", "#2BA57A"))
        for i, (v, lbl, col) in enumerate(marks):
            if math.isfinite(v):
                ax.axvline(v, color=col, lw=1, ls="-.")
                # alternate heights so labels of nearby speeds don't overlap
                y = ax.get_ylim()[1] * (0.98 if i % 2 == 0 else 0.86)
                ax.text(v, y, f" {lbl}\n {v:.1f}", color=col, fontsize=8, va="top")
        ax.set_xlabel("Airspeed  [m/s]", color="#444")
        ax.set_ylabel("Thrust  [N]", color="#444")
        ax.legend(fontsize=8, loc="center right")
        self.fig4.tight_layout()
        self.canvas4.draw_idle()

    def _build_trade_tab(self):
        t = QWidget()
        lay = QVBoxLayout(t)
        bar = QHBoxLayout()
        self.trade_param = QComboBox()
        for gname, items in FIELDS:
            if gname in ("Plot range",):
                continue
            for key, label, unit, *_ in items:
                if key != "mtow_guess":
                    self.trade_param.addItem(f"{label}" + (f" [{unit}]" if unit else ""), key)
        self.trade_param.setCurrentIndex(max(0, self.trade_param.findData("battery_wh_per_kg")))
        self.trade_param.currentIndexChanged.connect(self._trade_param_changed)
        self.trade_from = QDoubleSpinBox()
        self.trade_to = QDoubleSpinBox()
        for s in (self.trade_from, self.trade_to):
            s.setRange(-1e6, 1e6)
            s.setDecimals(3)
        self.trade_n = QSpinBox()
        self.trade_n.setRange(3, 60)
        self.trade_n.setValue(12)
        self.trade_rule = QComboBox()
        self.trade_rule.addItem("W/S: max-W/S corner", "corner")
        self.trade_rule.addItem("W/S: min-MTOW (slower)", "min_mtow")
        self.trade_rule.addItem("W/S: keep current design", "fixed")
        self.trade_rule.setToolTip("How the wing loading is chosen for every value of the parameter.")
        run = QPushButton("Run")
        run.clicked.connect(self.run_trade)
        exp = QPushButton("Export CSV…")
        exp.clicked.connect(self.export_trade)
        for wdg in (QLabel("Vary"), self.trade_param, QLabel("from"), self.trade_from, QLabel("to"),
                    self.trade_to, QLabel("points"), self.trade_n, self.trade_rule, run, exp):
            bar.addWidget(wdg)
        bar.addStretch()
        lay.addLayout(bar)
        self.fig3 = Figure(figsize=(10, 5), dpi=100, facecolor="#F4F6F9")
        self.trade_axes = self.fig3.subplots(2, 2).ravel()
        self.canvas3 = Canvas(self.fig3)
        lay.addWidget(NavigationToolbar2QT(self.canvas3, t))
        lay.addWidget(self.canvas3)
        self.trade_data = None
        self._trade_param_changed()
        return t

    def _build_results(self):
        self.report = QTextBrowser()
        self.report.setMinimumWidth(340)
        return self.report

    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        for text, key, fn in [
            ("Open inputs…", QKeySequence.Open, self.load_inputs),
            ("Save inputs…", QKeySequence.Save, self.save_inputs),
            ("Export diagram PNG…", None, self.export_png),
            ("Export report HTML…", None, self.export_report),
            ("Reset to defaults", None, self.reset_defaults),
        ]:
            a = QAction(text, self)
            if key:
                a.setShortcut(key)
            a.triggered.connect(fn)
            m.addAction(a)

    # ───────────────────────────────────────────────────────── inputs io ──
    def _changed(self, *_):
        self._timer.start()

    def _read_inputs(self):
        d = {}
        for key, w in self.widgets.items():
            d[key] = w.currentData() if isinstance(w, QComboBox) else w.value()
        for k in INT_FIELDS:
            d[k] = int(d[k])
        return Inputs.from_dict(d)

    def _write_inputs(self, inp: Inputs):
        for key, w in self.widgets.items():
            w.blockSignals(True)
            if isinstance(w, QComboBox):
                w.setCurrentIndex(max(0, w.findData(getattr(inp, key))))
            else:
                w.setValue(getattr(inp, key))
            w.blockSignals(False)

    def save_inputs(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save inputs", "uav_design.json", "JSON (*.json)")
        if path:
            d = {"inputs": self._read_inputs().to_dict(),
                 "design_ws": self.design_ws, "design_pw": self.design_pw}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2)

    def load_inputs(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open inputs", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            self._write_inputs(Inputs.from_dict(d.get("inputs", d)))
            self.design_ws = d.get("design_ws")
            self.design_pw = d.get("design_pw")
            self.snap.setChecked(self.design_pw is None)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Open failed", str(e))
        self.recompute()

    def reset_defaults(self):
        self._write_inputs(Inputs())
        self.design_ws = self.design_pw = None
        self.recompute()

    def export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export diagram", "constraint_diagram.png", "PNG (*.png)")
        if path:
            self.fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=self.fig.get_facecolor())

    def export_report(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export report", "uav_design_report.html", "HTML (*.html)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.report.toHtml())

    # ──────────────────────────────────────────────────── interaction ──
    def _on_click(self, ev):
        if ev.inaxes is None or ev.xdata is None or ev.button != 1:
            return
        tb = ev.canvas.toolbar
        if tb is not None and tb.mode:          # zoom / pan active
            return
        self.design_ws = float(ev.xdata)
        if not self.snap.isChecked() and ev.inaxes is self.ax_pw:
            self.design_pw = float(ev.ydata)
        else:
            self.design_pw = None
        self.recompute()

    def _snap_toggled(self, on):
        if on:
            self.design_pw = None
        self.recompute()

    def _pick_suggestion(self, i):
        if self.last is None:
            return
        pts = suggested_points(self.last.constraints)
        if pts[i] is not None:
            self.design_ws, self.design_pw = pts[i][0], None
            self.snap.setChecked(True)
            self.recompute()

    def _pick_min_mtow(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ws = min_mtow_ws(self._read_inputs())
        finally:
            QApplication.restoreOverrideCursor()
        if ws is None:
            QMessageBox.information(self, "Min-MTOW point", "No wing loading closes the sizing.")
            return
        self.design_ws, self.design_pw = ws, None
        self.snap.setChecked(True)
        self.recompute()

    # ───────────────────────────────────────────────────── trade study ──
    def _trade_param_changed(self, *_):
        key = self.trade_param.currentData()
        w = self.widgets[key]
        v = w.value()
        lo, hi = w.minimum(), w.maximum()
        if isinstance(w, QSpinBox):
            a, b = max(lo, 1), min(hi, max(v * 2, v + 3))
            if key == "n_tilt_rotors":
                b = self.widgets["n_rotors"].value()
            self.trade_n.setValue(int(b - a + 1) if b - a + 1 <= 60 else 12)
        else:
            span = abs(v) * 0.4 if v else (hi - lo) * 0.1
            a, b = max(lo, v - span), min(hi, v + span)
        self.trade_from.setValue(a)
        self.trade_to.setValue(b)

    def run_trade(self):
        key = self.trade_param.currentData()
        a, b = self.trade_from.value(), self.trade_to.value()
        n = self.trade_n.value()
        values = np.linspace(a, b, n)
        if isinstance(self.widgets[key], QSpinBox):
            values = np.unique(np.round(values).astype(int))
        rule = self.trade_rule.currentData()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            data = trade_study(self._read_inputs(), key, values, rule,
                               ws_fixed=self.last.ws if self.last else None)
        finally:
            QApplication.restoreOverrideCursor()
        self.trade_data = (key, self.trade_param.currentText(), rule, data)
        self._draw_trade()

    def _draw_trade(self):
        if self.trade_data is None:
            return
        key, label, rule, data = self.trade_data
        x = data[:, 0]
        cur = getattr(self.inp, key)
        bad = data[:, 7] < 0.5
        specs = [
            (0, "Take-off mass", "MTOW [kg]", [(2, "MTOW", C_ENV)]),
            (1, "Mass breakdown", "Mass [kg]", [(3, "Battery", "#C2477E"), (4, "Empty", "#6B8CBA")]),
            (2, "Installed electrical power", "Power [W]", [(5, "All motors", "#E07B39")]),
            (3, "Span and wing loading", "Span [m]", [(6, "Span", "#2BA57A")]),
        ]
        for ax, (_, title, ylabel, series) in zip(self.trade_axes, specs):
            ax.clear()
            ax.set_facecolor("#FAFBFC")
            ax.grid(color="#DDE3EC", linewidth=0.7, linestyle="--")
            ax.set_title(title, fontsize=10, fontweight="semibold", color="#1A2A3A")
            ax.set_ylabel(ylabel, color="#444")
            for col, name, color in series:
                ax.plot(x, data[:, col], "-o", ms=3.5, color=color, lw=1.8, label=name)
                if bad.any():
                    ax.plot(x[bad], data[bad, col], "x", color=C_BAD, ms=8, mew=2)
            if key != "payload_mass" and title == "Mass breakdown":
                ax.axhline(self.inp.payload_mass, color="#888", lw=1, ls=":", label="Payload")
            ax.axvline(cur, color=C_DP, lw=1, ls=":")
            ax.set_ylim(bottom=0)
        ax_ws = self.trade_axes[3].twinx()
        ax_ws.plot(x, data[:, 1], "--", color="#7E57C2", lw=1.4, label="W/S")
        ax_ws.set_ylabel("W/S [N/m²]", color="#7E57C2")
        ax_ws.set_ylim(bottom=0)
        self.trade_axes[3].legend(handles=self.trade_axes[3].get_lines()[:1] + ax_ws.get_lines(),
                                  fontsize=8, loc="lower right")
        self.trade_axes[1].legend(fontsize=8, loc="best")
        for ax in self.trade_axes[2:]:
            ax.set_xlabel(label, color="#444")
        rule_txt = {"corner": "max-W/S corner", "min_mtow": "min-MTOW W/S", "fixed": "fixed W/S"}[rule]
        self.fig3.suptitle(f"Trade study: {label}   (W/S chosen by {rule_txt}; red × = infeasible, "
                           f"dotted line = current value)", fontsize=10, color="#1A2A3A")
        # the twin axis is recreated on every draw; drop the previous one
        for extra in self.fig3.axes[4:-1]:
            extra.remove()
        self.fig3.tight_layout()
        self.canvas3.draw_idle()

    def export_trade(self):
        if self.trade_data is None:
            QMessageBox.information(self, "Export", "Run a trade study first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export trade study", "trade_study.csv", "CSV (*.csv)")
        if path:
            key = self.trade_data[0]
            header = ",".join((key,) + TRADE_COLUMNS[1:])
            np.savetxt(path, self.trade_data[3], delimiter=",", header=header, comments="", fmt="%.6g")

    # ─────────────────────────────────────────────────────── compute ──
    def recompute(self):
        self.inp = inp = self._read_inputs()
        comp = inp.empty_model == "components"
        self.boxes["Empty weight: components"].setEnabled(comp)
        self.boxes["Empty weight: regression"].setEnabled(not comp)
        self.widgets["n_tilt_rotors"].setEnabled(inp.vtol_mode == "tilt_quadplane")
        self.widgets["tilt_rotor_diameter"].setEnabled(inp.vtol_mode == "tilt_quadplane")
        self.widgets["cruise_prop_diameter"].setEnabled(inp.vtol_mode in ("quadplane", "none"))
        self.widgets["lift_pitch_ratio"].setEnabled(inp.n_rotors > inp.n_tilt)
        if inp.ws_max <= inp.ws_min + 1:
            return
        # auto design point: size once with a guess, then take the max-W/S corner
        ws = self.design_ws
        if ws is None:
            probe = evaluate(inp, 0.5 * (inp.ws_min + inp.ws_max))
            corner, _ = suggested_points(probe.constraints)
            ws = corner[0] if corner else probe.ws
        ws = min(max(ws, inp.ws_min), inp.ws_max)
        dp = evaluate(inp, ws, self.design_pw)
        self.last = dp
        tab = self.tabs.currentIndex()
        if tab == 0:
            self._draw_constraints(dp)
        elif tab == 1:
            self._draw_sweep(dp)
        elif tab == 3:
            self._draw_propulsion(dp)
        self._write_report(dp)

    # ───────────────────────────────────────────────────────── drawing ──
    @staticmethod
    def _style(ax, title, ylabel, x0, x1):
        ax.clear()
        ax.set_facecolor("#FAFBFC")
        ax.grid(color="#DDE3EC", linewidth=0.7, linestyle="--")
        ax.set_title(title, fontsize=11, fontweight="semibold", color="#1A2A3A")
        ax.set_xlabel("Wing loading  W/S  [N/m²]", color="#444")
        ax.set_ylabel(ylabel, color="#444")
        ax.set_xlim(x0, x1)
        for s in ax.spines.values():
            s.set_edgecolor("#CCC")

    def _draw_constraints(self, dp):
        c, inp = dp.constraints, self.inp
        for ax, attr, env, title, ylabel, ypt in [
            (self.ax_pw, "PW", c.PW_env, "Power-to-weight (electrical)", "P/W  [W/N]", dp.pw),
            (self.ax_tw, "TW", c.TW_env, "Thrust-to-weight", "T/W  [–]", dp.derived["TW_req"]),
        ]:
            self._style(ax, title, ylabel, inp.ws_min, inp.ws_max)
            # scale to the feasible region, not to the steep low-W/S end of the curves
            ok = (c.ws <= c.ws_limit) & np.isfinite(env)
            ref = np.nanmin(env[ok]) if ok.any() else np.nanmax(env)
            ymax = 3.0 * ref
            if attr == "PW" and math.isfinite(c.hover_PW):
                ymax = max(ymax, 1.25 * c.hover_PW)
            if attr == "TW" and inp.vtol_mode != "none":
                ymax = max(ymax, 1.25 * inp.hover_tw)
            if math.isfinite(ypt):
                ymax = max(ymax, 1.25 * ypt)
            for cv in c.curves:
                y = getattr(cv, attr)
                ax.plot(c.ws, y, color=cv.color, lw=1.5, alpha=0.9,
                        ls="--" if cv.dashed else "-", label=cv.label)
            ax.plot(c.ws, env, color=C_ENV, lw=2.4, zorder=5, label="Requirement envelope")
            # feasible region: above envelope and left of the W/S limits
            feas = (c.ws <= c.ws_limit) & np.isfinite(env)
            ax.fill_between(c.ws, env, ymax, where=feas, color=C_OK, alpha=0.10, zorder=1,
                            label="Feasible region")
            ax.fill_between(c.ws, 0, ymax, where=~feas, color="#999", alpha=0.12, hatch="//",
                            edgecolor="#bbb", lw=0, zorder=1)
            for lim in c.limits:
                if inp.ws_min < lim.ws_max < inp.ws_max:
                    ax.axvline(lim.ws_max, color=lim.color, lw=1.6, ls="-.", zorder=4)
                    ax.text(lim.ws_max, ymax * 0.97, f" {lim.label}\n {lim.ws_max:.0f}", color=lim.color,
                            fontsize=8, va="top", ha="left" if lim.ws_max < inp.ws_max * 0.8 else "right")
            ax.set_ylim(0, ymax)
            col = C_OK if dp.feasible else C_BAD
            ax.plot([dp.ws], [ypt], marker="*", ms=17, color=C_DP, mec="white", mew=1.2, zorder=10)
            ax.axvline(dp.ws, color=C_DP, lw=0.8, ls=":", zorder=3)
            ax.annotate(f"W/S {dp.ws:.0f}\n{attr[0]}/W {ypt:.2f}", (dp.ws, ypt), xytext=(10, 10),
                        textcoords="offset points", fontsize=9, color=col, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, alpha=0.9), zorder=11)
        corner, minp = suggested_points(c)
        for pt, mk in ((corner, "s"), (minp, "D")):
            if pt:
                self.ax_pw.plot([pt[0]], [pt[1]], marker=mk, ms=8, mfc="none", mec=C_ENV, mew=1.5, zorder=9)

        handles, labels = self.ax_pw.get_legend_handles_labels()
        handles += [Line2D([], [], marker="s", ls="", mfc="none", mec=C_ENV, label="Max-W/S corner"),
                    Line2D([], [], marker="D", ls="", mfc="none", mec=C_ENV, label="Min-power point"),
                    Line2D([], [], marker="*", ls="", color=C_DP, ms=12, label="Design point")]
        labels += ["Max-W/S corner", "Min-power point", "Design point"]
        for leg in list(self.fig.legends):
            leg.remove()
        self.fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8, frameon=False)
        self.fig.tight_layout(rect=[0, 0.1, 1, 1])
        self.canvas.draw_idle()

    def _draw_sweep(self, dp):
        inp = self.inp
        wsv = np.linspace(inp.ws_min, inp.ws_max, 70)
        data = sizing_sweep(inp, wsv)
        lim = dp.constraints.ws_limit
        self._style(self.ax_m, "Mass vs wing loading", "Mass  [kg]", inp.ws_min, inp.ws_max)
        self.ax_m.plot(data[:, 0], data[:, 1], color=C_ENV, lw=2.2, label="MTOW")
        self.ax_m.plot(data[:, 0], data[:, 2], color="#C2477E", lw=1.6, label="Battery")
        self.ax_m.axhline(inp.payload_mass, color="#888", lw=1, ls=":", label="Payload")
        self._style(self.ax_b, "Wing size vs wing loading", "Span [m]  /  Area [m²]", inp.ws_min, inp.ws_max)
        self.ax_b.plot(data[:, 0], data[:, 4], color="#2BA57A", lw=2, label="Span")
        self.ax_b.plot(data[:, 0], data[:, 3], color="#B07D2A", lw=2, label="Wing area")
        for ax in (self.ax_m, self.ax_b):
            ax.axvspan(lim, inp.ws_max, color="#999", alpha=0.15, hatch="//", lw=0)
            ax.axvline(dp.ws, color=C_DP, lw=1.2, ls=":")
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=8, loc="upper right")
        if inp.empty_model == "regression" and inp.wing_areal_mass == 0:
            self.ax_m.text(0.02, 0.97, "Wing areal mass = 0: wing size costs no weight,\n"
                                       "so MTOW only rises with W/S. Set it to see an optimum.",
                           transform=self.ax_m.transAxes, fontsize=8, va="top", color="#8a5a00")
        self.fig2.tight_layout()
        self.canvas2.draw_idle()

    # ───────────────────────────────────────────────────────── report ──
    def _write_report(self, dp):
        inp, d, s, ae = self.inp, dp.derived, dp.sizing, dp.aero
        col = C_OK if dp.feasible else C_BAD
        head = "FEASIBLE DESIGN" if dp.feasible else "NOT FEASIBLE"

        def table(title, rows):
            r = "".join(f"<tr><td>{a}</td><td align='right'><b>{b}</b></td></tr>" for a, b in rows)
            return f"<h3 style='margin:10px 0 2px 0;color:#1A3A5C'>{title}</h3><table width='100%' cellspacing='0' cellpadding='2'>{r}</table>"

        f = lambda v, fmt=".2f", u="": ("–" if v is None or (isinstance(v, float) and not math.isfinite(v))
                                        else f"{v:{fmt}} {u}".strip())
        html = [f"<div style='font-family:Segoe UI;font-size:10pt'>"
                f"<div style='background:{col};color:white;padding:6px;font-weight:bold'>{head}</div>"]
        if dp.reasons:
            html.append("<ul style='color:#D62828;margin:4px'>" + "".join(f"<li>{r}</li>" for r in dp.reasons) + "</ul>")
        html.append(table("Design point", [
            ("Wing loading W/S", f(dp.ws, ".1f", "N/m²")),
            ("Wing loading", f(dp.ws / G, ".2f", "kg/m²")),
            ("P/W chosen", f(dp.pw, ".2f", "W/N")),
            ("P/W required", f(d["PW_req"], ".2f", "W/N")),
            ("T/W required", f(d["TW_req"], ".3f")),
        ]))
        if s.converged:
            html.append(table("Mass", [
                ("MTOW", f(s.mtow, ".2f", "kg")),
                ("Empty", f"{s.empty_mass:.2f} kg ({s.empty_fraction:.0%})"),
                ("Battery", f"{s.battery_mass:.2f} kg ({s.battery_fraction:.0%})"),
                ("Payload", f"{s.payload_mass:.2f} kg ({s.payload_mass / s.mtow:.0%})"),
            ]))
            html.append(table("Empty-mass breakdown", [
                (f"&nbsp;&nbsp;{k}", f"{v:.3f} kg") for k, v in s.empty_breakdown_kg.items()]))
            html.append(table("Wing geometry", [
                ("Wing area S", f(d["S"], ".3f", "m²")),
                ("Span b", f(d["span"], ".2f", "m")),
                ("Mean chord", f(d["chord"], ".3f", "m")),
                ("Winglet height", f(d["winglet_h"], ".3f", "m")),
            ]))
            prop = [(f"{name} × {n}", f(p, ".0f", "W each")) for name, n, p in d["motor_groups"]]
            prop += [("Total installed (electrical)", f(d["P_installed"], ".0f", "W")),
                     ("Fixed-wing power needed", f(d["P_fixed_wing"], ".0f", "W"))]
            if inp.vtol_mode != "none":
                prop += [("Hover power (T/W = %.2f)" % inp.hover_tw, f(d["P_hover"], ".0f", "W")),
                         ("…average per rotor", f(d["P_hover_per_rotor"], ".0f", "W")),
                         ("Disk loading", f(d["disk_loading"], ".1f", "N/m²")),
                         ("Rotor diam. sum / span", f(d["rotor_span_ratio"], ".2f"))]
            html.append(table("Power", prop))
            eb = [(k, f(v, ".1f", "Wh")) for k, v in s.energy_breakdown_wh.items()]
            eb += [("Battery (nameplate)", f(s.battery_energy_wh, ".0f", "Wh"))]
            html.append(table("Energy", eb))
        html.append(table("Speeds", [
            ("Stall (1 g)", f(d["V_stall"], ".1f", "m/s")),
            ("Minimum operating", f(d["V_min"], ".1f", "m/s")),
            ("Min-power speed", f(d["V_min_power"], ".1f", "m/s")),
            ("Best-L/D speed", f(d["V_best_LD"], ".1f", "m/s")),
            ("Loiter speed", f(s.loiter.get("V"), ".1f", "m/s") if s.loiter else "–"),
            ("Loiter bank angle", f(math.degrees(math.acos(1 / s.loiter["n"])), ".0f", "°") if s.loiter else "–"),
        ]))
        html.append(table("Aerodynamics", [
            ("Oswald e", f(ae.e, ".3f")),
            ("Effective AR", f(ae.ar_eff, ".2f")),
            ("CD0 / K1 / K2", f"{ae.cd0:.4f} / {ae.k1:.4f} / {ae.k2:.4f}"),
            ("(L/D)max @ CL", f"{ae.ld_max:.1f} @ {ae.cl_ld_max:.2f}"),
            ("Min-power L/D @ CL", f"{ae.ld_min_power:.1f} @ {ae.cl_min_power:.2f}"),
            ("Cruise CL, L/D", f"{d['CL_cruise']:.2f}, {d['LD_cruise']:.1f}"),
        ] + ([("Reynolds no. (min speed / cruise)",
               f"{d['Re_min_speed'] / 1e3:.0f}k / {d['Re_cruise'] / 1e3:.0f}k")] if "Re_cruise" in d else [])))
        html.append(self._hints(dp))
        html.append("</div>")
        self.report.setHtml("".join(html))

    def _hints(self, dp):
        inp, d, ae = self.inp, dp.derived, dp.aero
        tips = []
        if "P_hover" in d and inp.hover_share > 0:
            hover_part = d["P_hover"] * inp.hover_share
            if hover_part > 1.1 * d["P_fixed_wing"]:
                tips.append(f"The tilting motors are sized by hover ({hover_part:.0f} W vs "
                            f"{d['P_fixed_wing']:.0f} W for flight): wing loading barely changes their size. "
                            "Larger rotors (lower disk loading) or fewer tilting rotors would balance it.")
            elif inp.vtol_mode == "tilt_quadplane" and d["P_fixed_wing"] > 1.5 * hover_part:
                tips.append("The tilting motors are sized by forward flight, well above their hover share. "
                            "Check that one prop can serve both hover and cruise efficiently.")
        if inp.vtol_mode in ("tilt_quadplane", "quadplane"):
            where = ("are counted in the component masses" if inp.empty_model == "components"
                     else "are only implicit in the empty-weight regression")
            tips.append("Lift-only rotors are dead weight and drag in cruise (stop them aligned with the "
                        f"boom). Their motors, props and booms {where}; their cruise drag is not modelled.")
        if d.get("Re_min_speed", 1e9) < 200e3:
            tips.append(f"Reynolds number at min speed is only {d['Re_min_speed'] / 1e3:.0f}k: use low-Re "
                        "airfoil data, as CL max and CD min are often optimistic below ~200k.")
        if d.get("rotor_span_ratio", 0) > 0.9:
            tips.append("Rotors are large compared to the span — check that they physically fit.")
        cl_lim = ae.cl_max / inp.speed_margin**2
        if ae.cl_min_power > cl_lim:
            tips.append("Min-power CL is above the usable CL, so loiter and climb fly at the stall margin, "
                        "not at their aerodynamic optimum.")
        if d["CL_cruise"] < 0.5 * ae.cl_ld_max:
            tips.append(f"Cruise CL {d['CL_cruise']:.2f} is far below best-L/D CL {ae.cl_ld_max:.2f}: "
                        "the wing is oversized for cruise; a higher W/S or slower cruise is more efficient.")
        if inp.sweep_deg > 30:
            tips.append("Sweep > 30°: Oswald factor uses Raymer's swept-wing formula.")
        if not tips:
            return ""
        return ("<h3 style='margin:10px 0 2px 0;color:#1A3A5C'>Notes</h3><ul style='margin:2px'>"
                + "".join(f"<li>{t}</li>" for t in tips) + "</ul>")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
