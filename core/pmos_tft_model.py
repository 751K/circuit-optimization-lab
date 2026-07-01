import math
import numpy as np
from scipy.optimize import fsolve
try:
    from .numba_kernels import (
        eval_currents_numba,
        newton_internal_numba,
        capacitances_numba,
        capacitance_charges_numba,
        terminal_derivatives_numba,
    )
except Exception:  # pragma: no cover - optional acceleration only
    eval_currents_numba = None
    newton_internal_numba = None
    capacitances_numba = None
    capacitance_charges_numba = None
    terminal_derivatives_numba = None

from .device_model import TransistorModel, NumbaParams, register_pdk
from . import diagnostics

class PMOS_TFT(TransistorModel):
    """
    Python equivalent of the Verilog-A model for AT4000TG pmos_TFT.
    Now includes DC solving, Parasitic Capacitances, and Noise Power Spectral Density evaluation.
    Note: Python is mainly used here for DC Operating Point, Capacitance, and Noise evaluation.
    Full transient (ddt) simulation would require an external ODE solver like SPICE.

    Implements :class:`~device_model.TransistorModel` — the abstract interface
    consumed by all solvers in the stack.
    """

    # Process/polarity identity — this model is the AT4000TG PMOS device
    # (``pmos_TFT`` in PDK/veriloga.va).  Carried for registry/introspection so
    # multiple PDKs and PMOS/NMOS polarities stay distinguishable.
    PDK = "at4000tg"
    POLARITY = "pmos"
    def __init__(self, W=1000, L=20, VT=-3.03, Roff=1, NF=1, Reg=1,
                 C1=37.5, C2=50, C3=35, C4=35, Ci=2.4, kv=1, kh=1,
                 temperature=300.15, pvt0=0, mvt0=0, pbeta0=0, mbeta0=0):
        # User defined parameters
        self.W = W
        self.L = L
        self.VT = VT
        self.Roff = Roff
        self.NF = NF
        self.Reg = Reg
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.Ci = Ci
        self.kv = kv
        self.kh = kh
        self.temperature = temperature
        
        # Monte carlo variation parameters
        self.pvt0 = pvt0
        self.mvt0 = mvt0
        self.pbeta0 = pbeta0
        self.mbeta0 = mbeta0

        # Physical constants
        self.q = 1.6e-19
        self.Kb = 1.38064e-23
        self.Kbe = 8.617343e-5
        self.E0 = 8.85418781e-14
        
        # Material properties
        self.Ks = 3
        self.alfa = 4.5455e7
        self.VE = 3.6
        self.Nt = 1.0000e21
        self.Nss = 2.0006e11
        self.s0 = 5.9658e6
        self.Tt = 304.9889
        self.Bc = 2.8

        # internal-node (Vs1,Vd1) solve cache: warm start + exact-point memo.
        # The transient/AC stack hammers get_op at nearly-identical biases
        # (residual, caps, and the gm/gds finite-diff all share a center point);
        # reusing the last root as a Newton seed turns a cold 5-guess fsolve into
        # a 1-2 step solve, and the exact-key memo skips duplicate center solves.
        self._op_cache = None     # last converged (Vs1, Vd1)
        self._op_key = None       # (Vs, Vd, Vg) it was solved at

        self._precompute_constants()

    def _precompute_constants(self):
        # Effective temperature
        self.T = 295 - (self.temperature - 295) / 4
        self.wt = self.Tt / self.T
        
        # Geometry in meters/cm
        self.w = self.W * 1e-6
        self.l = self.L * 1e-6
        self._two_over_pi = 2.0 / np.pi
        
        self.Cpvt = (1 + self.pvt0 + self.mvt0 / np.sqrt(self.W * self.L * 1e-12))
        self.Vfb = self.VT * (3.25 / self.Ci) * self.Cpvt
        self.beta0 = (1 + self.mbeta0 + self.pbeta0)
        
        self.Ceff = (self.Ci / 3.25 * 5.000) * 1e-9
        self.CI = self.Ci / self.Cpvt * 1e-9
        self.Es = self.E0 * self.Ks
        
        self.G0 = self.s0 * (((np.pi * (self.Tt/self.T)**3 * self.Nt) / (self.Bc * (2*self.alfa)**3)) ** (self.Tt/self.T))
        
        # Channel parameters
        self.Rleak = (2 * 1e12 / self.W) * self.L * self.Roff
        self._inv_Rleak = 1.0 / self.Rleak
        # beta equation
        term1 = (self.Es * self.G0 / self.Ceff)
        term2 = (self.Kbe * self.T) * (self.T / (2 * self.Tt - self.T))
        term3 = ((self.Ceff**2 * np.sin(np.pi * self.T / self.Tt)) / (np.pi * self.Nt * 2 * self.q * self.Kbe * self.T * self.Es)) ** (self.Tt / self.T)
        self.beta = self.beta0 * 0.8 * term1 * term2 * term3
        self._channel_exponent = 2 * self.Tt / self.T
        self._current_scale = (self.W / self.L) * self.beta
        
        self.Vss = 2 * self.Kbe * self.Tt * (1 + self.q * self.Nss / self.Ceff) * 2 * self.Tt / (2 * self.Tt - self.T)
        self.Esat = self.VE
        self.lambda_ = 1 / (self.L * self.Esat * (self.Ci / 3.25))
        
        # Contact parameters
        self.Lc = 5
        self.k0 = 1.1e-8
        self.Rcleak = (1e16 / self.W) * self.Lc
        self._contact_scale = (self.W / self.Lc) * self.k0
        self._min_rout_denom = 1.0 / np.finfo(float).max
        
        # Capacitance / AC parameters
        self.fw = self.W / self.NF
        self.cl = 10.0
        self.OSC_O1 = self.C2 - self.C3 + self.C4
        self.EDGE_OX = 2*self.C3 + 2*self.C1 * np.ceil((np.ceil((self.cl+self.L)*self.NF + self.cl + 2*self.OSC_O1 + self.kv*self.C2)/self.C1)/2)
        self.EDGE_OY = 2*self.C3 + 2*self.C1 * np.ceil(np.ceil((self.fw + 2*self.OSC_O1 + (self.kh-1)*self.C2)/self.C1)/2)
        self.g_area = (self.EDGE_OX + 2*self.C1) * (self.EDGE_OY + 2*self.C1)
        self.AOSC = self.EDGE_OX * self.EDGE_OY
        self._cap_cgs1 = (np.floor(self.NF / 2) + 1) * (self.fw + 210) * self.cl * self.CI
        self._cap_cgd1 = np.ceil(self.NF / 2) * (self.fw + 210) * self.cl * self.CI
        self._cap_half_wl_ci = 0.5 * self.W * self.L * self.CI
        self._cap_wl_ci = self.W * self.L * self.CI
        self._cap_cgs3_base = 0.5 * self.AOSC * self.CI - 1.43 * self._cap_wl_ci
        self._cap_cgd3_base = 0.5 * self.AOSC * self.CI - 0.33 * self._cap_wl_ci
        
        if self.Reg == 1: self.k1 = 1
        elif self.Reg == 2: self.k1 = 0
        elif self.Reg == 3: self.k1 = 0.1
        elif self.Reg == 4: self.k1 = 0.001
        else: self.k1 = 0
        
        # Gate internal node resistance
        self.R_cap = 100.0
        self.R_cap2 = 1e12 * self.Cpvt

    @staticmethod
    def _softplus(x):
        """ln(1 + exp(x)), overflow-safe. Scalar fast path (math) matches the
        stable form np.logaddexp(0,x) uses bit-for-bit; arrays fall back to numpy."""
        try:
            if x > 0.0:
                return x + math.log1p(math.exp(-x))
            return math.log1p(math.exp(x))
        except (TypeError, ValueError):              # array input
            return np.logaddexp(0.0, x)

    @staticmethod
    def _sigmoid(x):
        """1/(1+exp(-x)) = exp(-logaddexp(0,-x)), overflow-safe scalar fast path."""
        try:
            if x >= 0.0:
                z = math.exp(-x)
                return 1.0 / (1.0 + z)
            z = math.exp(x)
            return z / (1.0 + z)
        except (TypeError, ValueError):              # array input
            return np.exp(-np.logaddexp(0.0, -x))

    def _va_sorted_nodes(self, Vs, Vd, Vs1, Vd1):
        """Mirror the Verilog-A internal voltage reassignment exactly."""
        v_s = Vs if Vs > Vs1 else Vs1
        v_s1 = Vs1 if Vs > Vs1 else Vs
        v_d = Vd if Vd1 > Vd else Vd1
        v_d1 = Vd1 if Vd1 > Vd else Vd
        return v_s, v_s1, v_d, v_d1

    def _gate1_dc(self, Vs, Vd, Vg):
        # KCL at gate1 for the three DC resistive branches in the Verilog-A model.
        numerator = Vg / self.R_cap + (Vs + Vd) / self.R_cap2
        denominator = 1 / self.R_cap + 2 / self.R_cap2
        return numerator / denominator

    def _eval_channel(self, Vs, Vd, Vg, Vs1, Vd1):
        """Evaluate Verilog-A channel annotations from the solved operating point."""
        _, _, v_d, v_d1 = self._va_sorted_nodes(Vs, Vd, Vs1, Vd1)
        arg_d1 = (v_d1 - Vg + self.Vfb) / self.Vss
        arg_d = (v_d - Vg + self.Vfb) / self.Vss
        Vods = self.Vss * self._softplus(arg_d1)
        Vodd = self.Vss * self._softplus(arg_d)
        exponent = self._channel_exponent
        chmod = 1 + self.lambda_ * (v_d1 - v_d)
        current_scale = self._current_scale
        Ich = current_scale * (Vods**exponent - Vodd**exponent) * chmod
        lambda_ich = self.lambda_ * Ich
        if lambda_ich == 0.0:
            rout = np.inf
        elif abs(lambda_ich) < self._min_rout_denom:
            rout = np.copysign(np.inf, lambda_ich)
        else:
            rout = 1.0 / lambda_ich
        gm = current_scale * exponent * (
            Vods**(exponent - 1) * self._sigmoid(arg_d1)
            - Vodd**(exponent - 1) * self._sigmoid(arg_d)
        ) * chmod
        return {
            "Vodd": Vodd,
            "Vods": Vods,
            "chmod": chmod,
            "Ich": Ich,
            "rout": rout,
            "gm": gm,
        }

    def _eval_channel_ich_sorted(self, v_d, v_d1, Vg):
        """Fast Ich-only channel path for residual evaluations."""
        arg_d1 = (v_d1 - Vg + self.Vfb) / self.Vss
        arg_d = (v_d - Vg + self.Vfb) / self.Vss
        Vods = self.Vss * self._softplus(arg_d1)
        Vodd = self.Vss * self._softplus(arg_d)
        exponent = self._channel_exponent
        chmod = 1 + self.lambda_ * (v_d1 - v_d)
        return self._current_scale * (Vods**exponent - Vodd**exponent) * chmod

    def _eval_currents(self, Vs, Vd, Vg, Vs1, Vd1):
        """ Evaluates the DC branch currents given external and internal nodes """
        if eval_currents_numba is not None:
            return eval_currents_numba(
                Vs, Vd, Vg, Vs1, Vd1, self.Vfb, self.Vss, self.Lc, self.lambda_,
                self._contact_scale, self._channel_exponent, self._current_scale,
                self._inv_Rleak,
            )

        # Voltages sorting based on Verilog-A ternary operators.
        v_s, v_s1, v_d, v_d1 = self._va_sorted_nodes(Vs, Vd, Vs1, Vd1)
        
        # --- Contact Model (between s and s1) ---
        Vt = -(0.0045 * (v_s - Vg)**2 + 0.7125 * (v_s - Vg) + 0.9625)
        
        Vods1 = self.Vss * self._softplus((v_s - Vg + Vt) / self.Vss)
        Vodd1 = self.Vss * self._softplus((v_s1 - Vg + Vt) / self.Vss)
        
        Ecsat = 0.85 * 20 / (abs(v_s - Vg) + 0.1)
        lambdac = 1 / (self.Lc * Ecsat)
        cmod = 1 + lambdac * (v_s - v_s1)
        
        exponent = self._channel_exponent
        Icont = self._contact_scale * (Vods1**exponent - Vodd1**exponent) * cmod
        I_s_s1 = Icont if Vs > Vs1 else -Icont
        
        # --- Channel Model (between d1 and d) ---
        Ich = self._eval_channel_ich_sorted(v_d, v_d1, Vg)
        
        I_d1_d_ch = Ich if Vs1 > Vd else -Ich
        I_d1_d_leak = (Vd1 - Vd + 0.1) * self._inv_Rleak
        I_d1_d = I_d1_d_ch + I_d1_d_leak
        
        # --- Internal resistor (between s1 and d1) ---
        I_s1_d1 = (Vs1 - Vd1) / 0.1
        
        return I_s_s1, I_s1_d1, I_d1_d, Ich, I_d1_d_leak

    def _eval_ich(self, Vs, Vd, Vg, Vs1, Vd1):
        """Evaluate the Verilog-A Ich expression without branch sign or leakage."""
        return self._eval_channel(Vs, Vd, Vg, Vs1, Vd1)["Ich"]

    def _residuals(self, x, Vs, Vd, Vg):
        Vs1, Vd1 = x
        I_s_s1, I_s1_d1, I_d1_d, _, _ = self._eval_currents(Vs, Vd, Vg, Vs1, Vd1)
        
        # KCL at node s1
        res1 = I_s_s1 - I_s1_d1
        # KCL at node d1
        res2 = I_s1_d1 - I_d1_d
        return [res1, res2]

    def _newton_internal(self, Vs, Vd, Vg, x0, tol=1e-12, maxit=40):
        """Damped 2x2 Newton on the internal-node KCL, seeded from x0.

        Same residual (_residuals) and acceptance (||res|| < 1e-12) as the old
        fsolve path, but with an analytic 2x2 inverse and a warm start — so a seed
        within ~HH of the root converges in 1-2 iterations with no scipy overhead.
        Returns (Vs1, Vd1) on success, or None to let the robust path take over."""
        x0 = np.asarray(x0, float)
        if newton_internal_numba is not None:
            try:
                ok, Vs1, Vd1 = newton_internal_numba(
                    Vs, Vd, Vg, x0[0], x0[1], tol, maxit,
                    self.Vfb, self.Vss, self.Lc, self.lambda_,
                    self._contact_scale, self._channel_exponent,
                    self._current_scale, self._inv_Rleak)
                if ok:
                    return np.array([Vs1, Vd1])
                return None
            except Exception as exc:
                diagnostics.note("model.internal_newton_numba_fallback", exc)
        Vs1, Vd1 = x0[0], x0[1]
        hj = 1e-6                                    # finite-diff step for the 2x2 jac
        for _ in range(maxit):
            r0a, r0b = self._residuals((Vs1, Vd1), Vs, Vd, Vg)
            if abs(r0a) + abs(r0b) < tol:
                return np.array([Vs1, Vd1])
            r1a, r1b = self._residuals((Vs1 + hj, Vd1), Vs, Vd, Vg)
            r2a, r2b = self._residuals((Vs1, Vd1 + hj), Vs, Vd, Vg)
            j00 = (r1a - r0a) / hj; j01 = (r2a - r0a) / hj
            j10 = (r1b - r0b) / hj; j11 = (r2b - r0b) / hj
            det = j00 * j11 - j01 * j10
            if det == 0.0 or not np.isfinite(det):
                return None
            d0 = -(j11 * r0a - j01 * r0b) / det
            d1 = -(-j10 * r0a + j00 * r0b) / det
            mx = max(abs(d0), abs(d1))
            if mx > 2.0:                             # branch-safety: cap the jump
                s = 2.0 / mx; d0 *= s; d1 *= s
            Vs1 += d0; Vd1 += d1
            if max(abs(d0), abs(d1)) < 1e-13:        # stalled at the numeric floor
                if abs(r0a) + abs(r0b) < 1e-9:
                    return np.array([Vs1, Vd1])
                return None
        return None

    def _robust_op(self, Vs, Vd, Vg):
        """Cold multi-guess fsolve — the original, branch-robust internal solve.
        Used on the first call and whenever the warm-started Newton bails out."""
        guesses = (
            [Vs - 0.01*(Vs-Vd), Vd + 0.01*(Vs-Vd)],
            [Vs, Vd],
            [(Vs + Vd) / 2, (Vs + Vd) / 2],
            [Vs, Vs],
            [Vd, Vd],
        )
        best_sol = None
        best_norm = np.inf
        best_msg = ""
        for x0 in guesses:
            sol, _, ier, mesg = fsolve(
                self._residuals,
                x0,
                args=(Vs, Vd, Vg),
                full_output=True,
                xtol=1e-12,
                maxfev=2000,
            )
            residual_norm = np.linalg.norm(self._residuals(sol, Vs, Vd, Vg))
            if residual_norm < best_norm:
                best_sol = sol
                best_norm = residual_norm
                best_msg = f"ier={ier}\n{mesg}"
            if ier == 1 or residual_norm < 1e-12:
                break
        if best_norm >= 1e-12:
            raise RuntimeError(f"fsolve 未能收敛 ({best_msg})")
        return best_sol

    def _solve_internal(self, Vs, Vd, Vg):
        """Solve internal nodes (Vs1, Vd1), with exact-key memo + warm-start Newton.
        Falls back to the robust cold fsolve.  Warm-start enables a ~8× speed-up when
        the same instance is re-used across nearby biases (sweeps / continuation);
        the exact-key memo makes repeat calls at the identical bias near-instant."""
        key = (Vs, Vd, Vg)
        if self._op_key == key and self._op_cache is not None:
            return self._op_cache
        sol = None
        if self._op_cache is not None:               # warm start from the last root
            sol = self._newton_internal(Vs, Vd, Vg, self._op_cache)
        if sol is None:                              # cold guess, then robust fallback
            sol = self._newton_internal(
                Vs, Vd, Vg, (Vs - 0.01*(Vs-Vd), Vd + 0.01*(Vs-Vd)))
        if sol is None:
            sol = self._robust_op(Vs, Vd, Vg)
        self._op_key = key
        self._op_cache = sol
        return sol

    def get_op(self, Vs, Vd, Vg, include_gate1=False):
        """
        Solves DC operating point (internal nodes) for a given bias.
        By default returns (Vs1, Vd1) for backward compatibility.
        Set include_gate1=True to also return the Verilog-A gate1 DC solution.
        """
        sol = self._solve_internal(Vs, Vd, Vg)
        if include_gate1:
            return sol[0], sol[1], self._gate1_dc(Vs, Vd, Vg) # Vs1, Vd1, Vg1
        return sol[0], sol[1] # Vs1, Vd1

    def get_Idc(self, Vs, Vd, Vg):
        """ Calculates the DC drain current. """
        Vs1, Vd1 = self.get_op(Vs, Vd, Vg)
        I_s_s1, I_s1_d1, I_d1_d, _, _ = self._eval_currents(Vs, Vd, Vg, Vs1, Vd1)
        return -I_d1_d

    # ── TransistorModel interface methods ────────────────────────────────

    def get_numba_params(self):
        """Return the scalar parameter bundle consumed by numba kernels."""
        return NumbaParams(
            Vfb=self.Vfb,
            Vss=self.Vss,
            Lc=self.Lc,
            lambda_=self.lambda_,
            contact_scale=self._contact_scale,
            channel_exponent=self._channel_exponent,
            current_scale=self._current_scale,
            inv_Rleak=self._inv_Rleak,
            two_over_pi=self._two_over_pi,
            cap_cgs1=self._cap_cgs1,
            cap_cgd1=self._cap_cgd1,
            cap_half_wl_ci=self._cap_half_wl_ci,
            cap_cgs3_base=self._cap_cgs3_base,
            cap_cgd3_base=self._cap_cgd3_base,
            k1=self.k1,
            gate_leak_g=1.0 / self.R_cap2,
        )

    def get_ss_params(self, Vs, Vd, Vg):
        """Terminal small-signal parameters at the given bias.

        Overrides the base-class finite-difference default with an
        optimised path that uses :func:`terminal_derivatives_numba` for
        gm/gds and reuses the internal OP solve for capacitances.
        """
        h = 1e-3
        if terminal_derivatives_numba is not None:
            try:
                s1, d1 = self.get_op(Vs, Vd, Vg)
                _, _, I_d1_d, _, _ = self._eval_currents(Vs, Vd, Vg, s1, d1)
                Idc0 = -I_d1_d
                if abs(Idc0) < 1e-10:
                    raise RuntimeError("small-current finite-difference fallback")
                ok, gm_neg, gds_neg = terminal_derivatives_numba(
                    Vs, Vd, Vg, s1, d1, True, True, False, h, 1e-6,
                    self.Vfb, self.Vss, self.Lc, self.lambda_,
                    self._contact_scale, self._channel_exponent,
                    self._current_scale, self._inv_Rleak)
                if ok and np.isfinite(gm_neg) and np.isfinite(gds_neg):
                    gm = -gm_neg
                    gds = -gds_neg
                else:
                    raise RuntimeError("terminal derivative fallback")
                Cgss, Cgdd = self._capacitances_from_op(Vs, Vd, Vg, s1, d1)
                Ich = self._eval_channel(Vs, Vd, Vg, s1, d1)["Ich"]
                return {"gm": gm, "gds": gds, "Cgs": Cgss, "Cgd": Cgdd, "Ich": Ich}
            except Exception as exc:
                diagnostics.note("model.ss_params_numba_fallback", exc)

        # Finite-difference fallback (pure Python)
        try:
            Id = lambda vs, vd, vg: self.get_Idc(vs, vd, vg)
            gm = (Id(Vs, Vd, Vg + h) - Id(Vs, Vd, Vg - h)) / (2 * h)
            gds = (Id(Vs, Vd + h, Vg) - Id(Vs, Vd - h, Vg)) / (2 * h)
            Cgss, Cgdd = self.get_capacitances(Vs, Vd, Vg)
            s1, d1 = self.get_op(Vs, Vd, Vg)
            Ich = self._eval_channel(Vs, Vd, Vg, s1, d1)["Ich"]
            return {"gm": gm, "gds": gds, "Cgs": Cgss, "Cgd": Cgdd, "Ich": Ich}
        except Exception as exc:
            diagnostics.note_critical(
                "model.ss_params_zeroed", exc,
                detail="gm/gds/Cgs/Cgd/Ich -> 0/1e-12 (small-signal params fabricated)")
            return {"gm": 0.0, "gds": 1e-12, "Cgs": 0.0, "Cgd": 0.0, "Ich": 0.0}

    # ── Private helpers ──────────────────────────────────────────────────

    def _capacitances_from_op(self, Vs, Vd, Vg, Vs1, Vd1):
        """Capacitance equations evaluated from an already-solved internal OP."""
        if capacitances_numba is not None:
            try:
                return capacitances_numba(
                    Vs, Vd, Vg, Vs1, Vd1, self.Vfb, self._two_over_pi,
                    self._cap_cgs1, self._cap_cgd1, self._cap_half_wl_ci,
                    self._cap_cgs3_base, self._cap_cgd3_base, self.k1)
            except Exception as exc:
                diagnostics.note("model.caps_numba_fallback", exc)
        v_s, _, v_d, _ = self._va_sorted_nodes(Vs, Vd, Vs1, Vd1)

        Cgs1 = self._cap_cgs1
        Cgd1 = self._cap_cgd1

        arg_gs = v_s - Vg + self.Vfb
        Cgs2 = 1.43 * self._cap_half_wl_ci * (self._two_over_pi * np.arctan(arg_gs * 0.6) + 1)
        Cgd2 = 0.33 * self._cap_half_wl_ci * (self._two_over_pi * np.arctan(arg_gs * 2.01) + 1)

        arg_gd = -Vg + self.Vfb + v_d
        Cgs3 = 0.34 * self._cap_cgs3_base * (self._two_over_pi * np.arctan(arg_gd * 0.21) + 1)
        Cgd3 = 0.52 * self._cap_cgd3_base * (self._two_over_pi * np.arctan(arg_gd * 0.42) + 1)

        Cgss = self.k1 * (Cgs1 + Cgs2 + Cgs3) * 1e4 * 1e-12
        Cgdd = self.k1 * (Cgd1 + Cgd2 + Cgd3) * 1e4 * 1e-12
        return Cgss, Cgdd

    @staticmethod
    def _atan_cap_integral(y, scale, two_over_pi):
        """Integral of ``2/pi*atan(scale*y)+1`` with respect to ``y``."""
        ay = scale * y
        return y + two_over_pi * (y * math.atan(ay) -
                                  0.5 * math.log1p(ay * ay) / scale)

    def _capacitance_charges_from_op(self, Vs, Vd, Vg, Vs1, Vd1):
        """Return branch charges and local caps from an already-solved OP.

        The AT_4000TG Verilog-A capacitance equations are not a full
        reciprocal multi-terminal charge model. These integrated branch charges
        are kept for diagnostics and charge-oriented experiments; production
        transient uses a step-integrated displacement-current companion based on
        the same local Cgss/Cgdd equations.
        """
        if capacitance_charges_numba is not None:
            try:
                return capacitance_charges_numba(
                    Vs, Vd, Vg, Vs1, Vd1, self.Vfb, self._two_over_pi,
                    self._cap_cgs1, self._cap_cgd1, self._cap_half_wl_ci,
                    self._cap_cgs3_base, self._cap_cgd3_base, self.k1)
            except Exception as exc:
                diagnostics.note("model.cap_charges_numba_fallback", exc)

        v_s, _, v_d, _ = self._va_sorted_nodes(Vs, Vd, Vs1, Vd1)
        y_s = v_s - Vg + self.Vfb
        y_d = v_d - Vg + self.Vfb
        x_gs = Vg - Vs
        x_gd = Vg - Vd

        cgs2_coeff = 1.43 * self._cap_half_wl_ci
        cgd2_coeff = 0.33 * self._cap_half_wl_ci
        cgs3_coeff = 0.34 * self._cap_cgs3_base
        cgd3_coeff = 0.52 * self._cap_cgd3_base

        f_s_060 = self._two_over_pi * math.atan(y_s * 0.6) + 1.0
        f_s_201 = self._two_over_pi * math.atan(y_s * 2.01) + 1.0
        f_d_021 = self._two_over_pi * math.atan(y_d * 0.21) + 1.0
        f_d_042 = self._two_over_pi * math.atan(y_d * 0.42) + 1.0

        cgs_cross = cgs3_coeff * f_d_021
        cgd_cross = cgd2_coeff * f_s_201
        qscale = self.k1 * 1e4 * 1e-12

        qgs = qscale * (
            self._cap_cgs1 * x_gs
            - cgs2_coeff * self._atan_cap_integral(y_s, 0.6, self._two_over_pi)
            + cgs_cross * x_gs
        )
        qgd = qscale * (
            self._cap_cgd1 * x_gd
            + cgd_cross * x_gd
            - cgd3_coeff * self._atan_cap_integral(y_d, 0.42, self._two_over_pi)
        )
        Cgss = qscale * (self._cap_cgs1 + cgs2_coeff * f_s_060 + cgs_cross)
        Cgdd = qscale * (self._cap_cgd1 + cgd_cross + cgd3_coeff * f_d_042)
        return qgs, qgd, Cgss, Cgdd

    def _capacitance_branch_terms_from_op(self, Vs, Vd, Vg, Vs1, Vd1):
        """Branch self-charge terms for step-integrated C(V)*dV experiments.

        The returned q terms integrate only the capacitance components controlled
        by the same branch voltage. Cross-dependent overlap pieces are returned
        separately so a timestep can multiply them by the branch voltage change
        without adding an artificial dC_cross/dt feedthrough current.
        """
        v_s, _, v_d, _ = self._va_sorted_nodes(Vs, Vd, Vs1, Vd1)
        y_s = v_s - Vg + self.Vfb
        y_d = v_d - Vg + self.Vfb
        x_gs = Vg - Vs
        x_gd = Vg - Vd

        cgs2_coeff = 1.43 * self._cap_half_wl_ci
        cgd2_coeff = 0.33 * self._cap_half_wl_ci
        cgs3_coeff = 0.34 * self._cap_cgs3_base
        cgd3_coeff = 0.52 * self._cap_cgd3_base

        f_s_060 = self._two_over_pi * math.atan(y_s * 0.6) + 1.0
        f_s_201 = self._two_over_pi * math.atan(y_s * 2.01) + 1.0
        f_d_021 = self._two_over_pi * math.atan(y_d * 0.21) + 1.0
        f_d_042 = self._two_over_pi * math.atan(y_d * 0.42) + 1.0

        qscale = self.k1 * 1e4 * 1e-12
        cgs_cross = qscale * cgs3_coeff * f_d_021
        cgd_cross = qscale * cgd2_coeff * f_s_201
        qgs_self = qscale * (
            self._cap_cgs1 * x_gs
            - cgs2_coeff * self._atan_cap_integral(y_s, 0.6, self._two_over_pi)
        )
        qgd_self = qscale * (
            self._cap_cgd1 * x_gd
            - cgd3_coeff * self._atan_cap_integral(y_d, 0.42, self._two_over_pi)
        )
        Cgss = qscale * (self._cap_cgs1 + cgs2_coeff * f_s_060) + cgs_cross
        Cgdd = qscale * (self._cap_cgd1 + cgd3_coeff * f_d_042) + cgd_cross
        return qgs_self, qgd_self, cgs_cross, cgd_cross, Cgss, Cgdd

    # ── TransistorModel abstract capacitance interface ──
    # These are the public ABC methods; they delegate to the private
    # ``_from_op`` helpers that are also used internally by get_ss_params
    # and the transient solver closures.

    def get_capacitance_charges_from_op(self, Vs, Vd, Vg, Vs1, Vd1):
        """Return branch charges from a pre‑solved operating point."""
        return self._capacitance_charges_from_op(Vs, Vd, Vg, Vs1, Vd1)

    def get_capacitance_branch_terms_from_op(self, Vs, Vd, Vg, Vs1, Vd1):
        """Return self‑charge branch terms from a pre‑solved operating point."""
        return self._capacitance_branch_terms_from_op(Vs, Vd, Vg, Vs1, Vd1)

    def get_capacitance_charges(self, Vs, Vd, Vg):
        """Return diagnostic branch charges (gate->source, gate->drain)."""
        Vs1, Vd1 = self.get_op(Vs, Vd, Vg)
        return self._capacitance_charges_from_op(Vs, Vd, Vg, Vs1, Vd1)

    def _capacitance_components_from_op(self, Vs, Vd, Vg, Vs1, Vd1):
        """Return Verilog-A Cgs*/Cgd* components in farads.

        This is intentionally kept off the hot AC/transient path. It is used by
        the chopper charge-injection helper to derive a first-order channel-charge
        estimate from the same PDK capacitance equations instead of introducing an
        unrelated empirical capacitance.
        """
        v_s, _, v_d, _ = self._va_sorted_nodes(Vs, Vd, Vs1, Vd1)
        scale = self.k1 * 1e4 * 1e-12

        Cgs1 = self._cap_cgs1
        Cgd1 = self._cap_cgd1
        arg_gs = v_s - Vg + self.Vfb
        Cgs2 = 1.43 * self._cap_half_wl_ci * (self._two_over_pi * np.arctan(arg_gs * 0.6) + 1)
        Cgd2 = 0.33 * self._cap_half_wl_ci * (self._two_over_pi * np.arctan(arg_gs * 2.01) + 1)
        arg_gd = -Vg + self.Vfb + v_d
        Cgs3 = 0.34 * self._cap_cgs3_base * (self._two_over_pi * np.arctan(arg_gd * 0.21) + 1)
        Cgd3 = 0.52 * self._cap_cgd3_base * (self._two_over_pi * np.arctan(arg_gd * 0.42) + 1)

        out = {
            "Cgs1": Cgs1 * scale,
            "Cgd1": Cgd1 * scale,
            "Cgs2": Cgs2 * scale,
            "Cgd2": Cgd2 * scale,
            "Cgs3": Cgs3 * scale,
            "Cgd3": Cgd3 * scale,
        }
        out["Cgss"] = out["Cgs1"] + out["Cgs2"] + out["Cgs3"]
        out["Cgdd"] = out["Cgd1"] + out["Cgd2"] + out["Cgd3"]
        return out

    def get_capacitance_components(self, Vs, Vd, Vg):
        Vs1, Vd1 = self.get_op(Vs, Vd, Vg)
        return self._capacitance_components_from_op(Vs, Vd, Vg, Vs1, Vd1)

    def estimate_channel_charge(self, Vs, Vd, Vg, mobile_only=True):
        """Estimate turn-off channel charge for switch charge-injection modeling.

        The PDK Verilog-A model is capacitance based and does not provide a
        charge-conserving channel-charge state. This estimate uses the same
        capacitance equations and a local overdrive proxy, so it scales with W/L,
        NF, bias, and process parameters instead of using a fixed ad-hoc charge.
        """
        Vs1, Vd1 = self.get_op(Vs, Vd, Vg)
        comps = self._capacitance_components_from_op(Vs, Vd, Vg, Vs1, Vd1)
        v_s, _, v_d, _ = self._va_sorted_nodes(Vs, Vd, Vs1, Vd1)
        vov = max(0.0, 0.5 * ((v_s - Vg + self.Vfb) + (v_d - Vg + self.Vfb)))
        if mobile_only:
            cap = comps["Cgs2"] + comps["Cgd2"]
        else:
            cap = comps["Cgss"] + comps["Cgdd"]
        return cap * vov

    def get_Idc_and_capacitances(self, Vs, Vd, Vg):
        """Return drain current and capacitances from one shared internal OP solve."""
        Vs1, Vd1 = self.get_op(Vs, Vd, Vg)
        _, _, I_d1_d, _, _ = self._eval_currents(Vs, Vd, Vg, Vs1, Vd1)
        Cgss, Cgdd = self._capacitances_from_op(Vs, Vd, Vg, Vs1, Vd1)
        return -I_d1_d, Cgss, Cgdd

    def get_capacitances(self, Vs, Vd, Vg):
        """
        Calculates small-signal parasitic capacitances Cgss and Cgdd
        at the specified DC operating point.
        """
        Vs1, Vd1 = self.get_op(Vs, Vd, Vg)

        # From the Verilog-A, gate1 is connected to s and d via these capacitors
        # I(s,gate1) = Cgss * ddt(V(s,gate1)) + ...
        # I(d,gate1) = Cgdd * ddt(V(d,gate1)) + ...
        return self._capacitances_from_op(Vs, Vd, Vg, Vs1, Vd1)

    def get_os(self, Vs, Vd, Vg):
        """
        Return the same operating-point quantities used by Cadence OS("/M0" "...").
        Values are named after the Verilog-A real variables where applicable.
        """
        Vs1, Vd1, Vg1 = self.get_op(Vs, Vd, Vg, include_gate1=True)
        v_s, v_s1, v_d, v_d1 = self._va_sorted_nodes(Vs, Vd, Vs1, Vd1)
        I_s_s1, I_s1_d1, I_d1_d, Ich, Ioff = self._eval_currents(Vs, Vd, Vg, Vs1, Vd1)
        channel = self._eval_channel(Vs, Vd, Vg, Vs1, Vd1)
        Cgss, Cgdd = self.get_capacitances(Vs, Vd, Vg)
        Vsg = Vs - Vg
        Vsd = Vs - Vd
        return {
            "Vg": Vg,
            "Vg1": Vg1,
            "Vs": v_s,
            "Vs1": v_s1,
            "Vd": v_d,
            "Vd1": v_d1,
            "Vsg": Vsg,
            "Vsd": Vsd,
            "Vsd_sat": Vsd - Vsg - self.Vfb,
            "Ich": Ich,
            "Ioff": Ioff,
            "I_tft": Ich + Ioff,
            "Idc": -I_d1_d,
            "gm": channel["gm"],
            "rout": channel["rout"],
            "Cgss": Cgss,
            "Cgdd": Cgdd,
            "w": self.w,
            "l": self.l,
            "W": self.W,
            "L": self.L,
        }

    def get_cadence_metrics(self, Vs, Vd, Vg, width=None):
        """
        Match the calculator expressions in the Cadence screenshot:
        gain, ft, id/w, and gm/id.
        width defaults to the Cadence VAR("w") convention, matching the W
        instance parameter value in um. Use width=os_values["w"] for A/m.
        """
        os_values = self.get_os(Vs, Vd, Vg)
        gm = os_values["gm"]
        Ich = os_values["Ich"]
        capacitance = os_values["Cgdd"] + os_values["Cgss"]
        width_value = os_values["W"] if width is None else width
        return {
            "gain": 20 * np.log10(gm * os_values["rout"]),
            "ft": gm / (2 * np.pi * capacitance),
            "idw": Ich / width_value,
            "gm_id": gm / Ich,
            **os_values,
        }

    def get_noise_psd(self, Vs, Vd, Vg, frequency):
        """
        Calculates thermal and flicker noise Power Spectral Density (A^2/Hz)
        at the given frequency.
        """
        Vs1, Vd1 = self.get_op(Vs, Vd, Vg)
        _, _, _, Ich, Ioff = self._eval_currents(Vs, Vd, Vg, Vs1, Vd1)
        
        gm = self._eval_channel(Vs, Vd, Vg, Vs1, Vd1)["gm"]
        
        # 1. Thermal Noise (white noise)
        # I(s,d) <+ white_noise(2*q*(Ich+Ioff),"thermal");
        # I(s,d) <+ white_noise(4*Kb*($temperature)*gm*2.0/3.0,"thermal");
        S_th1 = 2 * self.q * (Ich + Ioff)
        S_th2 = 4 * self.Kb * self.temperature * gm * 2.0 / 3.0
        S_thermal = S_th1 + S_th2
        
        # 2. Flicker Noise (1/f noise)
        hooge = 0.05
        _, _, _, v_d1 = self._va_sorted_nodes(Vs, Vd, Vs1, Vd1)
        denom = self.w * self.l * self.CI * 1e4 * (v_d1 - Vg + self.Vfb)
        
        # Cadence/Spectre AT4000TG behavior verified with the real PDK model:
        # a negative flicker_noise coefficient is reported as a positive
        # contribution with the same magnitude in PSFASCII noise output.
        S_flicker_1Hz = (hooge * self.q * (Ich**2)) / abs(denom)
        S_flicker = S_flicker_1Hz / frequency
        
        return S_thermal, S_flicker

if __name__ == "__main__":
    # Test the complete model
    tft = PMOS_TFT(W=1000, L=20)
    Vs, Vd, Vg = 40.0, 0.0, 20.0

    Id = tft.get_Idc(Vs, Vd, Vg)
    Cgss, Cgdd = tft.get_capacitances(Vs, Vd, Vg)
    S_th, S_fl = tft.get_noise_psd(Vs, Vd, Vg, frequency=100.0)

    print("=== PMOS TFT Python Model Output ===")
    print(f"Drain Current (Id): {Id*1e6:.2f} uA")
    print(f"Cgss: {Cgss*1e12:.4f} pF, Cgdd: {Cgdd*1e12:.4f} pF")
    print(f"Thermal Noise PSD: {S_th:.4e} A^2/Hz")
    print(f"Flicker Noise PSD @ 100Hz: {S_fl:.4e} A^2/Hz")


# Register the AT4000TG process as the (currently only, hence default) PDK.
# This exposes the PMOS under the structured key ``"at4000tg.pmos"`` and keeps
# the legacy alias ``"pmos_tft"`` working, so solvers resolve the default via
# ``create_device(get_default_model_type(), …)`` and a future process or an
# NMOS polarity slots in with another ``register_pdk`` call — no solver edits.
register_pdk("at4000tg", {"pmos": PMOS_TFT}, default=True,
             aliases={"pmos_tft": "pmos"})
