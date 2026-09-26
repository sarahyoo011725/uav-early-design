"""
Propeller performance database: C_T and C_P versus advance ratio J and RPM.

Sources (files in ./data):
  data/uiuc_vol4/  UIUC Propeller Data Site, Volume 4: wind-tunnel MEASURED data
                   for 17 APC Thin Electric props, 12-21 in (Dantsker, Caccamo,
                   Deters & Selig, AIAA 2022-4020). Static (J = 0) sweeps plus
                   J sweeps up to the tunnel speed limit.
  data/apc/        APC PER3_*.dat files: COMPUTED by APC (vortex theory) for the
                   full J range. Newer files can be downloaded from
                   https://www.apcprop.com/technical-information/performance-data/
                   and dropped into this folder (same file names overwrite).

Lookup: measured data is used wherever it covers the requested (J, RPM);
otherwise APC's computed data. Each query reports which source it used.
"""
from __future__ import annotations

import glob
import math
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
INCH = 0.0254
BLEND_J = 0.15   # advance-ratio span over which computed data is blended onto the measured end


def _parse_size(tok: str):
    """'16x10' -> (16, 10); '13x65' -> (13, 6.5); '25x125' -> (25, 12.5); '105x45' -> (10.5, 4.5)."""
    d_s, p_s = tok.lower().split("x")
    d = float(d_s)
    if d > 40:                     # e.g. 105 -> 10.5
        d /= 10
    p = float(p_s)
    if p > 1.5 * d:                # e.g. 65 on a 13" prop -> 6.5
        p /= 10
    return d, p


@dataclass
class Table:
    rpm: float
    J: np.ndarray
    CT: np.ndarray
    CP: np.ndarray


@dataclass
class PropEntry:
    key: str                  # e.g. "16x10E"
    d_in: float
    p_in: float
    measured: list[Table] = field(default_factory=list)   # UIUC, sorted by rpm
    static: Table | None = None                            # UIUC static: J=0, "J" holds RPM
    computed: list[Table] = field(default_factory=list)   # APC, sorted by rpm
    apc_date: str = ""

    @property
    def diameter(self):
        return self.d_in * INCH

    @property
    def pitch_ratio(self):
        return self.p_in / self.d_in

    @property
    def label(self):
        p = f"{self.p_in:g}"
        return f'{self.d_in:g}×{p}E'

    @property
    def sources(self):
        s = []
        if self.measured or self.static is not None:
            s.append("UIUC measured")
        if self.computed:
            s.append("APC computed")
        return " + ".join(s)

    @property
    def max_rpm(self):
        """APC's published limit for Thin Electric props: 150,000 / diameter [in]."""
        return 150000.0 / self.d_in

    # ── lookup ──
    @staticmethod
    def _interp_tables(tables: list[Table], J, rpm):
        """C_T, C_P at (J, rpm) by linear interpolation in J then RPM; None if J outside data."""
        if not tables:
            return None
        rpms = np.array([t.rpm for t in tables])
        i = int(np.searchsorted(rpms, rpm))
        cand = [tables[max(i - 1, 0)], tables[min(i, len(tables) - 1)]]
        # keep only the bracketing tables that cover this J (tunnel data ranges differ per RPM)
        cand = [t for t in cand if t.J[0] - 1e-9 <= J <= t.J[-1] + 1e-9]
        if not cand:
            return None
        if len(cand) == 1:
            t = cand[0]
            return np.interp(J, t.J, t.CT), np.interp(J, t.J, t.CP)
        vals = [(np.interp(J, t.J, t.CT), np.interp(J, t.J, t.CP)) for t in cand]
        r0, r1 = cand[0].rpm, cand[1].rpm
        w = 0.0 if r1 == r0 else min(max((rpm - r0) / (r1 - r0), 0.0), 1.0)
        return ((1 - w) * vals[0][0] + w * vals[1][0], (1 - w) * vals[0][1] + w * vals[1][1])

    def coeffs(self, J, rpm):
        """(C_T, C_P, source) at advance ratio J and rpm. C_T may be <= 0 past zero thrust."""
        if J <= 1e-6 and self.static is not None:
            s = self.static
            r = min(max(rpm, s.J[0]), s.J[-1])
            return float(np.interp(r, s.J, s.CT)), float(np.interp(r, s.J, s.CP)), "measured"
        v = self._interp_tables(self.measured, J, rpm)
        if v is not None:
            return float(v[0]), float(v[1]), "measured"
        # between the static test (J = 0) and the first wind-tunnel point: interpolate measured data
        if self.static is not None and self.measured:
            j_start = self._measured_j_start(rpm)
            if 0 < J < j_start:
                s = self.static
                r = min(max(rpm, s.J[0]), s.J[-1])
                ct0, cp0 = np.interp(r, s.J, s.CT), np.interp(r, s.J, s.CP)
                ct1, cp1 = self._interp_tables(self.measured, j_start, rpm)
                w = J / j_start
                return float((1 - w) * ct0 + w * ct1), float((1 - w) * cp0 + w * cp1), "measured"
        v = self._interp_tables(self.computed, J, rpm)
        if v is not None:
            ct, cp = float(v[0]), float(v[1])
            # just past the end of the measured range, scale the computed data so the two
            # sources join smoothly; the correction fades out over BLEND_J in advance ratio
            jm = self._measured_j_end(rpm)
            if jm is not None and 0 < J - jm < BLEND_J:
                meas = self._interp_tables(self.measured, jm, rpm)
                comp = self._interp_tables(self.computed, jm, rpm)
                if meas is not None and comp is not None and comp[0] > 0 and comp[1] > 0:
                    w = 1 - (J - jm) / BLEND_J
                    ct *= 1 + w * (meas[0] / comp[0] - 1)
                    cp *= 1 + w * (meas[1] / comp[1] - 1)
                    return ct, cp, "APC, blended to measured"
            return ct, cp, "APC computed"
        # outside the tables: use the table nearest in RPM, clamped at its ends
        tables = self.computed or self.measured
        t = min(tables, key=lambda tb: abs(tb.rpm - rpm))
        if J >= t.J[-1]:
            return min(float(t.CT[-1]), 0.0), float(t.CP[-1]), "extrapolated"
        if J <= t.J[0]:
            return float(t.CT[0]), float(t.CP[0]), "extrapolated"
        return float(np.interp(J, t.J, t.CT)), float(np.interp(J, t.J, t.CP)), "extrapolated"

    def _measured_j_start(self, rpm):
        """Lowest J covered by the measured tables bracketing this rpm."""
        rpms = np.array([t.rpm for t in self.measured])
        i = int(np.searchsorted(rpms, rpm))
        cand = {max(i - 1, 0), min(i, len(self.measured) - 1)}
        return float(min(self.measured[k].J[0] for k in cand))

    def _measured_j_end(self, rpm):
        """Highest J covered by the measured tables bracketing this rpm (None if no measured data)."""
        if not self.measured:
            return None
        rpms = np.array([t.rpm for t in self.measured])
        i = int(np.searchsorted(rpms, rpm))
        cand = {max(i - 1, 0), min(i, len(self.measured) - 1)}
        return float(max(self.measured[k].J[-1] for k in cand))

    def solve(self, T, V, rho):
        """
        Rotational speed giving thrust T [N] at airspeed V [m/s].
        Returns dict(rps, rpm, J, CT, CP, P_shaft, eta, FM, source) or None if not reachable.
        """
        D = self.diameter

        def thrust(n):
            J = V / (n * D) if V > 0 else 0.0
            ct, _, _ = self.coeffs(J, n * 60)
            return ct * rho * n * n * D**4

        lo, hi = 2.0, 1.3 * self.max_rpm / 60     # rev/s; allow 30 % over the limit for reporting
        if thrust(hi) < T:
            return None
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if thrust(mid) < T:
                lo = mid
            else:
                hi = mid
        n = hi
        J = V / (n * D) if V > 0 else 0.0
        ct, cp, src = self.coeffs(J, n * 60)
        P = cp * rho * n**3 * D**5
        out = dict(rps=n, rpm=n * 60, J=J, CT=ct, CP=cp, P_shaft=P, source=src,
                   eta=(ct * J / cp) if V > 0 and cp > 0 else math.nan)
        if V == 0:
            A = math.pi * D * D / 4
            out["FM"] = T**1.5 / (math.sqrt(2 * rho * A) * P) if P > 0 else math.nan
        return out

    def max_thrust(self, V, rho, rpm):
        D = self.diameter
        n = rpm / 60
        J = V / (n * D)
        ct, _, _ = self.coeffs(J, rpm)
        return max(ct, 0.0) * rho * n * n * D**4


# ─────────────────────────────────────────────────────────────── parsing ──

def _read_uiuc_file(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    return np.array(rows) if rows else None


def _read_apc_file(path):
    tables, date = [], ""
    rpm, rows = None, []

    def flush():
        if rpm is not None and len(rows) > 2:
            a = np.array(rows)
            # columns: V(mph) J Pe Ct Cp ...  keep the part up to (and including) the first Ct <= 0
            J, CT, CP = a[:, 1], a[:, 3], a[:, 4]
            order = np.argsort(J, kind="stable")
            J, CT, CP = J[order], CT[order], CP[order]
            _, keep = np.unique(J, return_index=True)
            tables.append(Table(rpm, J[keep], CT[keep], CP[keep]))

    with open(path, errors="ignore") as f:
        for i, line in enumerate(f):
            if i == 0:
                m = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})", line)
                date = m.group(1) if m else ""
            m = re.search(r"PROP RPM\s*=\s*([\d.]+)", line)
            if m:
                flush()
                rpm, rows = float(m.group(1)), []
                continue
            parts = line.split()
            if rpm is not None and len(parts) >= 5:
                try:
                    rows.append([float(x) for x in parts[:5]])
                except ValueError:
                    pass
    flush()
    return sorted(tables, key=lambda t: t.rpm), date


def _merge_runs(tables: list[Table], tol=0.05) -> list[Table]:
    """UIUC tests each nominal RPM in two tunnel runs (low and high speed) at slightly
    different RPM; join runs within `tol` of each other into one table over the full J range."""
    tables = sorted(tables, key=lambda t: t.rpm)
    groups: list[list[Table]] = []
    for t in tables:
        if groups and t.rpm <= groups[-1][0].rpm * (1 + tol):
            groups[-1].append(t)
        else:
            groups.append([t])
    out = []
    for g in groups:
        J = np.concatenate([t.J for t in g])
        CT = np.concatenate([t.CT for t in g])
        CP = np.concatenate([t.CP for t in g])
        o = np.argsort(J, kind="stable")
        J, CT, CP = J[o], CT[o], CP[o]
        # overlapping runs: average points that share (almost) the same J
        uj, inv = np.unique(np.round(J, 3), return_inverse=True)
        ct = np.bincount(inv, CT) / np.bincount(inv)
        cp = np.bincount(inv, CP) / np.bincount(inv)
        out.append(Table(float(np.mean([t.rpm for t in g])), uj, ct, cp))
    return out


@lru_cache(maxsize=1)
def load_database(data_dir: str = DATA_DIR) -> dict[str, PropEntry]:
    db: dict[str, PropEntry] = {}

    def entry(d, p):
        key = f"{d:g}x{p:g}E"
        if key not in db:
            db[key] = PropEntry(key, d, p)
        return db[key]

    for path in glob.glob(os.path.join(data_dir, "apc", "PER3_*.dat")):
        name = os.path.basename(path)[5:-4]           # e.g. 16x10E
        m = re.fullmatch(r"(\d+x\d+)E", name, re.I)
        if not m:
            continue                                     # thin-electric only
        tables, date = _read_apc_file(path)
        if tables:
            e = entry(*_parse_size(m.group(1)))
            e.computed, e.apc_date = tables, date

    for path in glob.glob(os.path.join(data_dir, "uiuc_vol4", "apce_*.txt")):
        name = os.path.basename(path)[5:-4]           # e.g. 16x10_static_2150od / 16x10_2158od_4960
        parts = name.split("_")
        e = entry(*_parse_size(parts[0]))
        a = _read_uiuc_file(path)
        if a is None or len(a) < 2:
            continue
        if parts[1] == "static":
            o = np.argsort(a[:, 0])
            e.static = Table(0.0, a[o, 0], a[o, 1], a[o, 2])
        else:
            rpm = float(parts[-1])
            o = np.argsort(a[:, 0])
            e.measured.append(Table(rpm, a[o, 0], a[o, 1], a[o, 2]))
    for e in db.values():
        e.measured = _merge_runs(e.measured)
    return dict(sorted(db.items(), key=lambda kv: (kv[1].d_in, kv[1].p_in)))
