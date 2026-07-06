"""
Single source of truth for the AFE circuit topology.

Everything the solvers need — DC KCL, per-device bias mapping, and the AC/noise
small-signal terminal list — is DERIVED from one device table, instead of being
hand-written (and duplicated) in ac_solver and noise_solver. This removes the
4-way duplication and the transcription-bug class (e.g. the asymmetric-DC bug
where M8's drain was wrongly mapped to VOP instead of VON).

A device is (name, drain, gate, source) given by NODE NAME. A node name is either
  - a SOLVED node (unknown, an MNA index), or
  - a RAIL / bias (known voltage): VDD, GND (=0), VB, VC, VCM.

Pass a Topology instance into ac_solve / noise_analysis via `topo=...`.
The default AFE topology is `AFE_TOPO`.
"""


class Topology:
    def __init__(self, solved, devices, rails, outputs=None, input_drives=None,
                 load_caps=None, dc_guesses=None, aliases=None, transient_inputs=None,
                 resistors=None, capacitors=None, isources=None, vccs=None,
                 vsources=None, vcvs=None, cccs=None, ccvs=None, ac_drives=None,
                 dc_tol=None, require_dc_in_box=False):
        self.solved = list(solved)                 # MNA node order (index = position)
        self.idx = {n: i for i, n in enumerate(self.solved)}
        self.n = len(self.solved)
        self.devices = list(devices)               # (name, drain, gate, source) by node name
        self.rails = dict(rails)                   # rail name -> bias-key (str) or constant (float)
        # Passive / source elements (two-terminal). Transistors stay in `devices`
        # and are handled by the :class:`~device_model.TransistorModel` interface;
        # these never touch the transistor model machinery.
        self.resistors = list(resistors or [])     # (name, a, b, R[ohm])
        self.capacitors = list(capacitors or [])   # (name, a, b, C[farad])
        self.isources = list(isources or [])       # (name, nplus, nminus, I[amp]); I flows
        #                                            nplus->nminus inside the source.
        self.vccs = list(vccs or [])               # (name, p, q, ctrl_p, ctrl_n, gm[A/V])
        #                                            I_p->q = gm * (Vctrl_p - Vctrl_n)
        # Ideal voltage sources (true MNA). Each adds one branch-current UNKNOWN and
        # one constraint row V_p - V_q = E, growing the system from n to n_aug = n + m.
        # `value` is a constant E (float) or a transient input-waveform key (str); for
        # DC/AC the waveform-keyed source uses E=0 (pure stimulus). Branch current k has
        # global index n + k.
        self.vsources = list(vsources or [])       # (name, p, q, value)
        # Controlled sources — three remaining dependent-source types.
        # VCVS: (name, p, q, cp, cn, mu) — V_p - V_q = mu*(V_cp - V_cn), adds branch current.
        self.vcvs = list(vcvs or [])
        # CCCS: (name, p, q, ctrl_name, beta) — I_out = beta*I_ctrl, no branch current.
        self.cccs = list(cccs or [])
        # CCVS: (name, p, q, ctrl_name, gamma) — V_p - V_q = gamma*I_ctrl, adds branch current.
        self.ccvs = list(ccvs or [])
        self.n_branches = len(self.vsources) + len(self.vcvs) + len(self.ccvs)
        self.n_aug = self.n + self.n_branches
        # Branch-current index for all voltage-source types (vsource / VCVS / CCVS).
        # Order: vsources first, then VCVS, then CCVS.
        self.vsource_index = {}
        for k, (name, *_rest) in enumerate(self.vsources):
            self.vsource_index[name] = self.n + k
        for k, (name, *_rest) in enumerate(self.vcvs):
            self.vsource_index[name] = self.n + len(self.vsources) + k
        for k, (name, *_rest) in enumerate(self.ccvs):
            self.vsource_index[name] = self.n + len(self.vsources) + len(self.vcvs) + k
        # Analysis metadata. These keep the solvers topology-driven instead of
        # hard-coding AFE node/device names.
        self.outputs = tuple(outputs or ())
        self.input_drives = dict(input_drives or {})        # AC drive per device gate
        # AC stimulus applied at (rail) NODES — used to drive a front-end network
        # (electrode / coupling) instead of driving device gates directly. Treated
        # as AC ground in noise analysis (inputs carry no signal there).
        self.ac_drives = dict(ac_drives or {})              # node name -> AC drive value
        self.load_caps = list(load_caps or [])              # (node_a, node_b, C)
        self.dc_guesses = list(dc_guesses or [])            # dict guesses or callables
        self.aliases = dict(aliases or {})                  # dc_op alias -> solved node
        self.transient_inputs = dict(transient_inputs or {})# device -> input key
        self.dc_tol = dc_tol
        self.require_dc_in_box = bool(require_dc_in_box)

    # ── node-voltage lookup ──────────────────────────────────────────
    def node_v(self, name, node_vals, bias):
        """Voltage of a node: solved -> from node_vals dict; rail -> bias/const."""
        if name in self.idx:
            return node_vals[name]
        r = self.rails[name]
        return bias[r] if isinstance(r, str) else r

    # ── DC: build the KCL residual vector from the topology ──────────
    def dc_residuals(self, x, bias, Idfun, gmin):
        """KCL at every solved node (residual = net current INTO node = 0):
        +I at a device's drain, -I at its source, resistor branch currents,
        ideal current-source injections, minus gmin*V. Idfun(name, Vs, Vd, Vg)
        returns the device current; capacitors are open at DC."""
        nv = {self.solved[k]: x[k] for k in range(self.n)}
        res = [0.0] * self.n_aug
        for name, d, g, s in self.devices:
            i = Idfun(name, self.node_v(s, nv, bias),
                      self.node_v(d, nv, bias), self.node_v(g, nv, bias))
            if d in self.idx:
                res[self.idx[d]] += i
            if s in self.idx:
                res[self.idx[s]] -= i
        for name, a, b, R in self.resistors:
            i_ab = (self.node_v(a, nv, bias) - self.node_v(b, nv, bias)) / R  # a -> b
            if a in self.idx:
                res[self.idx[a]] -= i_ab           # current leaves node a
            if b in self.idx:
                res[self.idx[b]] += i_ab           # current enters node b
        for name, p, q, I in self.isources:
            if p in self.idx:
                res[self.idx[p]] -= I              # source draws I from nplus
            if q in self.idx:
                res[self.idx[q]] += I              # and injects I into nminus
        for name, p, q, cp, cn, gm in self.vccs:
            I_vccs = gm * (self.node_v(cp, nv, bias) - self.node_v(cn, nv, bias))
            if p in self.idx:
                res[self.idx[p]] += I_vccs         # injects into p
            if q in self.idx:
                res[self.idx[q]] -= I_vccs         # extracts from q
        # Ideal voltage sources (true MNA): branch current x[n+k] flows p->q inside the
        # source; the constraint row pins V_p - V_q = E. gmin is applied to NODE rows only.
        for k, (name, p, q, value) in enumerate(self.vsources):
            bi = self.n + k
            i = x[bi]
            if p in self.idx:
                res[self.idx[p]] -= i              # current leaves p
            if q in self.idx:
                res[self.idx[q]] += i              # and enters q
            E = float(value) if isinstance(value, (int, float)) else 0.0
            res[bi] = self.node_v(p, nv, bias) - self.node_v(q, nv, bias) - E
        for k, (name, p, q, cp, cn, mu) in enumerate(self.vcvs):
            bi = self.vsource_index[name]
            i = x[bi]
            if p in self.idx:
                res[self.idx[p]] -= i              # current leaves p
            if q in self.idx:
                res[self.idx[q]] += i              # and enters q
            res[bi] = (self.node_v(p, nv, bias) - self.node_v(q, nv, bias)
                       - mu * (self.node_v(cp, nv, bias) - self.node_v(cn, nv, bias)))
        for name, p, q, ctrl_name, beta in self.cccs:
            ctrl_bi = self.vsource_index.get(ctrl_name)
            if ctrl_bi is None:
                raise ValueError(f"CCCS {name!r} references unknown branch source {ctrl_name!r}")
            I_out = beta * x[ctrl_bi]
            if p in self.idx:
                res[self.idx[p]] += I_out          # injects into p
            if q in self.idx:
                res[self.idx[q]] -= I_out          # extracts from q
        for k, (name, p, q, ctrl_name, gamma) in enumerate(self.ccvs):
            bi = self.vsource_index[name]
            ctrl_bi = self.vsource_index.get(ctrl_name)
            if ctrl_bi is None:
                raise ValueError(f"CCVS {name!r} references unknown branch source {ctrl_name!r}")
            i = x[bi]
            I_ctrl = x[ctrl_bi]
            if p in self.idx:
                res[self.idx[p]] -= i              # current leaves p
            if q in self.idx:
                res[self.idx[q]] += i              # and enters q
            res[bi] = (self.node_v(p, nv, bias) - self.node_v(q, nv, bias)
                       - gamma * I_ctrl)
        for k in range(self.n):
            res[k] -= x[k] * gmin
        return res

    def cap_list(self):
        """All linear capacitor branches as (a, b, C): explicit load_caps plus
        capacitor elements. AC and transient stamp these identically."""
        caps = list(self.load_caps)
        caps.extend((a, b, C) for _, a, b, C in self.capacitors)
        return caps

    def node_vals(self, sol):
        """Map a solved vector back to a {node_name: voltage} dict."""
        return {self.solved[k]: sol[k] for k in range(self.n)}

    def guess_vector(self, node_vals, default=0.0):
        """Turn a {name: voltage} guess dict into a vector in solved order. Ideal
        voltage-source branch currents (if any) are appended as zeros -> length n_aug."""
        return ([node_vals.get(self.solved[k], default) for k in range(self.n)]
                + [0.0] * self.n_branches)

    def rail_values(self, bias):
        out = {}
        for name, ref in self.rails.items():
            out[name] = bias[ref] if isinstance(ref, str) else ref
        return out

    def default_guess_value(self, bias):
        if "VCM" in bias:
            return bias["VCM"]
        rails = [v for v in self.rail_values(bias).values() if isinstance(v, (int, float))]
        if rails:
            return 0.5 * (min(rails) + max(rails))
        return 0.0

    def dc_guess_vectors(self, bias):
        default = self.default_guess_value(bias)
        guesses = []
        for guess in self.dc_guesses:
            g = guess(bias) if callable(guess) else guess
            guesses.append(self.guess_vector(g, default=default))
        rails = [v for v in self.rail_values(bias).values() if isinstance(v, (int, float))]
        pad = [0.0] * self.n_branches
        if rails:
            lo, hi = min(rails), max(rails)
            guesses.extend([[default] * self.n + pad,
                            [0.5 * (lo + hi)] * self.n + pad,
                            [lo] * self.n + pad,
                            [hi] * self.n + pad])
        else:
            guesses.append([default] * self.n + pad)
        return guesses

    def in_voltage_box(self, node_vals, bias, margin=0.5):
        rails = [v for v in self.rail_values(bias).values() if isinstance(v, (int, float))]
        if not rails:
            return True
        lo, hi = min(rails) - margin, max(rails) + margin
        return all(lo <= v <= hi for v in node_vals.values())

    def dc_op_with_aliases(self, node_vals):
        out = dict(node_vals)
        for alias, node in self.aliases.items():
            if node in node_vals:
                out[alias] = node_vals[node]
        return out

    # ── per-device DC bias (Vs, Vd, Vg) at a solved operating point ──
    def bias_points(self, node_vals, bias):
        return {name: (self.node_v(s, node_vals, bias),
                       self.node_v(d, node_vals, bias),
                       self.node_v(g, node_vals, bias))
                for name, d, g, s in self.devices}

    # ── AC/noise small-signal terminal list ──────────────────────────
    def ac_devices(self, drive=None, node_drives=None):
        """(name, d_term, g_term, s_term) with terminals encoded as
              ("n", idx)  -> solved node
              ("v", val)  -> known AC voltage (rails -> 0; gate input -> drive[name],
                             node AC source -> node_drives[node])
        `drive` maps device name -> gate AC drive (e.g. +/-0.5 for the input pair
        in the gain analysis; empty dict for noise where gates are AC ground)."""
        drive = drive or {}
        node_drives = node_drives or {}

        def term(node, role, dev):
            if node in self.idx:
                return ("n", self.idx[node])
            if node in node_drives:
                return ("v", float(node_drives[node]))
            if role == "g":
                return ("v", float(drive.get(dev, 0.0)))
            return ("v", 0.0)                      # rail -> AC ground

        return [(name, term(d, "d", name), term(g, "g", name), term(s, "s", name))
                for name, d, g, s in self.devices]

    def ac_term(self, node, drives=None):
        """AC terminal token for a node. Solved -> ("n", idx); otherwise a known AC
        voltage ("v", val): a stimulus value from `drives` if listed, else AC ground."""
        if node in self.idx:
            return ("n", self.idx[node])
        if drives and node in drives:
            return ("v", float(drives[node]))
        return ("v", 0.0)

    def output_weights(self):
        """Linear output sense vector. One output means node-to-ground; two outputs
        mean first minus second."""
        if not self.outputs:
            raise ValueError("Topology.outputs must name one or two solved output nodes")
        weights = {}
        if len(self.outputs) == 1:
            weights[self.outputs[0]] = 1.0
        elif len(self.outputs) == 2:
            weights[self.outputs[0]] = 1.0
            weights[self.outputs[1]] = -1.0
        else:
            raise ValueError("Topology.outputs supports one output or a differential pair")
        for node in weights:
            if node not in self.idx:
                raise ValueError(f"Output node {node!r} is not a solved node")
        return weights

    def output_value(self, node_vals):
        return sum(node_vals[n] * w for n, w in self.output_weights().items())


def _afe_guesses(bias):
    VCM = bias["VCM"]
    return {"VOP": VCM - 4, "VON": VCM - 4, "VFBP": VCM - 8,
            "VFBN": VCM - 8, "NET20": VCM + 15, "NET2": VCM + 7}


# ── the AFE: fully-differential amp, cross-coupled positive-feedback level shifter ──
AFE_TOPO = Topology(
    solved=["VOP", "VON", "VFBP", "VFBN", "NET20", "NET2"],   # MNA index 0..5
    devices=[
        ("M6",  "NET2",  "VB",   "VDD"),     # tail current source
        ("M7",  "VOP",   "VCM",  "NET2"),    # input pair +
        ("M8",  "VON",   "VCM",  "NET2"),    # input pair -
        ("M9",  "GND",   "VFBP", "VOP"),     # output stage +
        ("M10", "GND",   "VFBN", "VON"),     # output stage -
        ("M11", "NET20", "VC",   "VDD"),     # level-shifter tail
        ("M12", "VFBN",  "VOP",  "NET20"),   # cross-coupled +fb
        ("M13", "VFBP",  "VON",  "NET20"),   # cross-coupled +fb
        ("M14", "GND",   "GND",  "VFBN"),    # level-shifter load
        ("M15", "GND",   "GND",  "VFBP"),    # level-shifter load
    ],
    rails={"VDD": "VDD", "GND": 0.0, "VB": "VB", "VC": "VC", "VCM": "VCM"},
    outputs=("VOP", "VON"),
    input_drives={"M7": +0.5, "M8": -0.5},
    load_caps=(("VOP", "GND", 5e-12), ("VON", "GND", 5e-12)),
    dc_guesses=(_afe_guesses,
                lambda b: {"VOP": b["VCM"] - 2, "VON": b["VCM"] - 2,
                           "VFBP": b["VCM"] - 10, "VFBN": b["VCM"] - 10,
                           "NET20": b["VCM"] + 12, "NET2": b["VCM"] + 9},
                lambda b: {"VOP": b["VCM"] - 6, "VON": b["VCM"] - 6,
                           "VFBP": b["VCM"] - 6, "VFBN": b["VCM"] - 6,
                           "NET20": b["VCM"] + 18, "NET2": b["VCM"] + 5}),
    aliases={"net2": "NET2", "n20": "NET20", "vfb": "VFBP",
             "vfbp": "VFBP", "vfbn": "VFBN"},
    transient_inputs={"M7": "vip", "M8": "vin"},
)
