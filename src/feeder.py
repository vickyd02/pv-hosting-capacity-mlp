"""OpenDSS wrapper for the IEEE 123-bus feeder, following Siddique et al. (2026), Sec. 2.1 and 3.1.

PV units sit at randomly chosen load buses. All units grow together in 10 kW steps until a
voltage (outside 0.90-1.10 pu) or thermal (line current >= rating) violation occurs. The
hosting capacity (HC) is the total installed PV at the last step without a violation.
"""
import os
import numpy as np
import opendssdirect as dss

FEEDER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "feeder")

V_MAX = 1.10      # pu, paper Step 7: 10% of nominal voltage
V_MIN = 0.90      # pu
I_MAX = 1.00      # line current / rated current, paper Step 7
STEP_KW = 10.0    # PV size step per unit, paper Step 7


class Feeder:
    def __init__(self, thermal=True):
        """thermal=False drops the line-current limit, so only voltage limits define HC."""
        self.thermal = thermal
        cwd = os.getcwd()
        os.chdir(FEEDER_DIR)
        dss.Text.Command("Clear")
        dss.Text.Command("Redirect IEEE123Master.dss")
        dss.Text.Command("Set VoltageBases=[4.16, 0.48]")
        dss.Text.Command("CalcVoltageBases")
        self.reg_xfmrs = []
        for reg in dss.RegControls.AllNames():
            dss.RegControls.Name(reg)
            self.reg_xfmrs.append(dss.RegControls.Transformer())
            dss.Text.Command(f"RegControl.{reg}.maxtapchange=16")
        dss.Text.Command("Set MaxControlIter=50")
        os.chdir(cwd)

        # 24-hour profile: OpenDSS's built-in "default" daily loadshape (60-minute resolution)
        dss.LoadShape.Name("default")
        self.loadshape = np.array(dss.LoadShape.PMult())

        # nominal loads and their buses
        self.load_names = list(dss.Loads.AllNames())
        self.load_kw0, self.load_kvar0, self.load_bus = {}, {}, {}
        for name in self.load_names:
            dss.Loads.Name(name)
            self.load_kw0[name] = dss.Loads.kW()
            self.load_kvar0[name] = dss.Loads.kvar()
            self.load_bus[name] = dss.CktElement.BusNames()[0]

        # One candidate PV site per load bus (85 buses). Buses 49, 65 and 76 carry one load per
        # phase, so a PV unit there is split evenly across that bus's per-phase generators.
        self.pv_sites = sorted({self.load_bus[n].split(".")[0] for n in self.load_names}, key=int)
        self.site_gens = {b: [] for b in self.pv_sites}
        for n in self.load_names:
            dss.Loads.Name(n)
            ph, kv = dss.CktElement.NumPhases(), dss.Loads.kV()
            dss.Text.Command(f"New Generator.pv_{n} bus1={self.load_bus[n]} phases={ph} "
                             f"kV={kv} kW=0 pf=1 model=1")
            self.site_gens[self.load_bus[n].split(".")[0]].append(f"pv_{n}")
        self.all_gens = [g for b in self.pv_sites for g in self.site_gens[b]]

        self.node_names = list(dss.Circuit.AllNodeNames())

        # Line ratings. The paper does not give them, and the feeder file's 400 A default is
        # exceeded with no PV whenever load is above about 65% of nominal. Assumption: each line
        # is rated at max(400 A, 1.25 x its current at nominal load with no PV).
        self.set_hour_mult(1.0)
        i = dss.Lines.First()
        while i:
            nph = dss.CktElement.NumPhases()
            amps = np.array(dss.CktElement.CurrentsMagAng()[0:2 * nph:2]).max()
            dss.Lines.NormAmps(max(400.0, 1.25 * amps))
            i = dss.Lines.Next()

    def set_hour_mult(self, mult):
        """Scale every load by the same loadshape multiplier, with no PV connected."""
        for name in self.load_names:
            dss.Loads.Name(name)
            dss.Loads.kW(self.load_kw0[name] * mult)
            dss.Loads.kvar(self.load_kvar0[name] * mult)
        self.set_pv([], 0.0)
        return self.solve()

    def set_pv(self, sites, kw_each):
        """Put a PV unit of kw_each kW (unity power factor) at each site; all others off."""
        for g in self.all_gens:
            dss.Generators.Name(g)
            dss.Generators.kW(0.0)
        for b in sites:
            gens = self.site_gens[b]
            for g in gens:
                dss.Generators.Name(g)
                dss.Generators.kW(kw_each / len(gens))

    def solve(self):
        """Reset regulator taps to neutral, then let the regulators settle. Resetting first makes
        every solve independent of solve history, so the HC search is well defined."""
        for x in self.reg_xfmrs:
            dss.Transformers.Name(x)
            dss.Transformers.Wdg(2)
            dss.Transformers.Tap(1.0)
        dss.Solution.Solve()
        return dss.Solution.Converged()

    def voltages(self):
        """Per-unit magnitude and angle (degrees) at every node, ordered as self.node_names."""
        mag = np.array(dss.Circuit.AllBusMagPu())
        v = np.array(dss.Circuit.AllBusVolts())
        ang = np.degrees(np.arctan2(v[1::2], v[0::2]))
        return mag, ang

    def max_line_loading(self):
        worst = 0.0
        i = dss.Lines.First()
        while i:
            nph = dss.CktElement.NumPhases()
            amps = np.array(dss.CktElement.CurrentsMagAng()[0:2 * nph:2])
            worst = max(worst, amps.max() / dss.Lines.NormAmps())
            i = dss.Lines.Next()
        return worst

    def feasible(self, sites, k):
        """True if PV of k x STEP_KW at every site causes no violation."""
        self.set_pv(sites, k * STEP_KW)
        if not self.solve():
            return False
        mag, _ = self.voltages()
        if mag.max() > V_MAX or mag.min() < V_MIN:
            return False
        return not self.thermal or self.max_line_loading() < I_MAX

    def hosting_capacity(self, sites, k_hi=64):
        """Largest number of 10 kW steps k with no violation, found by bisection over integers.
        This gives the same answer as the paper's incremental search (Step 7), which stops at the
        first violating step, as long as feasibility does not return after the first violation.
        Returns (k, HC in kW); the solved state is left at the HC point."""
        if not self.feasible(sites, 0):
            return 0, 0.0
        lo = 0
        while self.feasible(sites, k_hi):
            lo, k_hi = k_hi, 2 * k_hi
        while k_hi - lo > 1:
            mid = (lo + k_hi) // 2
            if self.feasible(sites, mid):
                lo = mid
            else:
                k_hi = mid
        self.feasible(sites, lo)
        return lo, lo * STEP_KW * len(sites)
