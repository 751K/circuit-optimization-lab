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
    def __init__(self, solved, devices, rails):
        self.solved = list(solved)                 # MNA node order (index = position)
        self.idx = {n: i for i, n in enumerate(self.solved)}
        self.n = len(self.solved)
        self.devices = list(devices)               # (name, drain, gate, source) by node name
        self.rails = dict(rails)                   # rail name -> bias-key (str) or constant (float)

    # ── node-voltage lookup ──────────────────────────────────────────
    def node_v(self, name, node_vals, bias):
        """Voltage of a node: solved -> from node_vals dict; rail -> bias/const."""
        if name in self.idx:
            return node_vals[name]
        r = self.rails[name]
        return bias[r] if isinstance(r, str) else r

    # ── DC: build the KCL residual vector from the topology ──────────
    def dc_residuals(self, x, bias, Idfun, gmin):
        """KCL at every solved node: +I at a device's drain, -I at its source,
        minus gmin*V. Idfun(name, Vs, Vd, Vg) returns the device current."""
        nv = {self.solved[k]: x[k] for k in range(self.n)}
        res = [0.0] * self.n
        for name, d, g, s in self.devices:
            i = Idfun(name, self.node_v(s, nv, bias),
                      self.node_v(d, nv, bias), self.node_v(g, nv, bias))
            if d in self.idx:
                res[self.idx[d]] += i
            if s in self.idx:
                res[self.idx[s]] -= i
        for k in range(self.n):
            res[k] -= x[k] * gmin
        return res

    def node_vals(self, sol):
        """Map a solved vector back to a {node_name: voltage} dict."""
        return {self.solved[k]: sol[k] for k in range(self.n)}

    def guess_vector(self, node_vals, default=0.0):
        """Turn a {name: voltage} guess dict into a vector in solved order."""
        return [node_vals.get(self.solved[k], default) for k in range(self.n)]

    # ── per-device DC bias (Vs, Vd, Vg) at a solved operating point ──
    def bias_points(self, node_vals, bias):
        return {name: (self.node_v(s, node_vals, bias),
                       self.node_v(d, node_vals, bias),
                       self.node_v(g, node_vals, bias))
                for name, d, g, s in self.devices}

    # ── AC/noise small-signal terminal list ──────────────────────────
    def ac_devices(self, drive=None):
        """(name, d_term, g_term, s_term) with terminals encoded as
              ("n", idx)  -> solved node
              ("v", val)  -> known AC voltage (rails -> 0; gate input -> drive[name])
        `drive` maps device name -> gate AC drive (e.g. +/-0.5 for the input pair
        in the gain analysis; empty dict for noise where gates are AC ground)."""
        drive = drive or {}

        def term(node, role, dev):
            if node in self.idx:
                return ("n", self.idx[node])
            if role == "g":
                return ("v", float(drive.get(dev, 0.0)))
            return ("v", 0.0)                      # rail -> AC ground

        return [(name, term(d, "d", name), term(g, "g", name), term(s, "s", name))
                for name, d, g, s in self.devices]


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
)
