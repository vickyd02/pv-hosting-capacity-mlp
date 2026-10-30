"""OpenDSS wrapper for the IEEE 123-bus feeder: load scenarios, PV injection, HC by bisection."""
import os
import numpy as np
import opendssdirect as dss

FEEDER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "feeder")

# Hosting-capacity limits (edit to match the paper's definitions)
V_MAX = 1.05      # pu, overvoltage limit
V_MIN = 0.95      # pu, undervoltage limit
I_MAX = 1.00      # fraction of line NormAmps


class Feeder:
    def __init__(self, pv_buses=None):
        cwd = os.getcwd()
        os.chdir(FEEDER_DIR)
        dss.Text.Command("Clear")
        dss.Text.Command("Redirect IEEE123Master.dss")
        dss.Text.Command("Set VoltageBases=[4.16, 0.48]")
        dss.Text.Command("CalcVoltageBases")
        self.regs = list(dss.RegControls.AllNames())
        self.reg_xfmrs = []
        for reg in self.regs:
            dss.RegControls.Name(reg)
            self.reg_xfmrs.append(dss.RegControls.Transformer())
            dss.Text.Command(f"RegControl.{reg}.maxtapchange=16")
        dss.Text.Command("Set MaxControlIter=50")
        os.chdir(cwd)

        # nominal loads and their buses
        self.load_names = list(dss.Loads.AllNames())
        self.load_kw0, self.load_kvar0, self.load_bus = {}, {}, {}
        for name in self.load_names:
            dss.Loads.Name(name)
            self.load_kw0[name] = dss.Loads.kW()
            self.load_kvar0[name] = dss.Loads.kvar()
            self.load_bus[name] = dss.CktElement.BusNames()[0]

        # PV: one unity-power-factor generator per PV location.
        # Default = every load bus, total PV split in proportion to nominal load.
        self.pv_loads = self.load_names if pv_buses is None else [
            n for n in self.load_names if self.load_bus[n].split(".")[0] in pv_buses]
        w = np.array([self.load_kw0[n] for n in self.pv_loads])
        self.pv_share = w / w.sum()
        for n in self.pv_loads:
            dss.Loads.Name(n)
            ph, kv = dss.CktElement.NumPhases(), dss.Loads.kV()
            dss.Text.Command(f"New Generator.pv_{n} bus1={self.load_bus[n]} phases={ph} "
                             f"kV={kv} kW=0 pf=1 model=1")

        self.node_names = list(dss.Circuit.AllNodeNames())

        # Line ratings. The feeder file gives every line the 400 A default, which the main
        # trunk already exceeds at nominal load. Assumption: each line is rated at
        # max(400 A, 1.25 x its current at nominal load, no PV).
        self.solve()
        i = dss.Lines.First()
        while i:
            nph = dss.CktElement.NumPhases()
            amps = np.array(dss.CktElement.CurrentsMagAng()[0:2 * nph:2]).max()
            dss.Lines.NormAmps(max(400.0, 1.25 * amps))
            i = dss.Lines.Next()

    def set_loads(self, mult):
        """Apply per-load scaling factors (same order as self.load_names) and solve with no PV."""
        for name, m in zip(self.load_names, mult):
            dss.Loads.Name(name)
            dss.Loads.kW(self.load_kw0[name] * m)
            dss.Loads.kvar(self.load_kvar0[name] * m)
        self.set_pv(0.0)
        return self.solve()

    def set_pv(self, total_kw):
        for n, s in zip(self.pv_loads, self.pv_share):
            dss.Generators.Name(f"pv_{n}")
            dss.Generators.kW(total_kw * s)

    def solve(self):
        """Reset regulator taps to neutral, then let the regulators settle for the current
        loads and PV. Resetting first makes every solve independent of solve history, so the
        bisection on PV is well defined."""
        for x in self.reg_xfmrs:
            dss.Transformers.Name(x)
            dss.Transformers.Wdg(2)
            dss.Transformers.Tap(1.0)
        dss.Solution.Solve()
        return dss.Solution.Converged()

    def voltages(self):
        """Per-unit voltage magnitude at every node, ordered as self.node_names."""
        return np.array(dss.Circuit.AllBusMagPu())

    def max_line_loading(self):
        worst = 0.0
        i = dss.Lines.First()
        while i:
            nph = dss.CktElement.NumPhases()
            amps = np.array(dss.CktElement.CurrentsMagAng()[0:2 * nph:2])
            worst = max(worst, amps.max() / dss.Lines.NormAmps())
            i = dss.Lines.Next()
        return worst

    def feasible(self, total_kw):
        self.set_pv(total_kw)
        if not self.solve():
            return False
        v = self.voltages()
        return v.max() <= V_MAX and v.min() >= V_MIN and self.max_line_loading() <= I_MAX

    def hosting_capacity(self, hi_kw=20000.0, tol_kw=1.0):
        """Largest total PV (kW) with no limit violation, found by bisection."""
        lo = 0.0
        if not self.feasible(lo):
            return 0.0
        while self.feasible(hi_kw):      # make sure the upper bracket is infeasible
            hi_kw *= 2
        while hi_kw - lo > tol_kw:
            mid = 0.5 * (lo + hi_kw)
            if self.feasible(mid):
                lo = mid
            else:
                hi_kw = mid
        return lo
