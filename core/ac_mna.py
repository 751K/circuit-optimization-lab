"""
Small-signal MNA stamping primitives for the AFE.

Terminal encoding: each transistor terminal is either
    ("n", i)  -> solved node index i
    ("v", x)  -> known AC voltage x (AC grounds use x=0.0; driven inputs use x=Vin)

These three stamps (_stamp_adm, _stamp_vccs, _stamp_mos) are the small-signal
engine used by ac_solver.ac_solve and noise_solver.noise_analysis. The full
DC+AC solve (with corner support and terminal-gm extraction) lives in
ac_solver.py; this module is just the reusable stamp kernel.
"""

# Node-index convention used by the callers (for reference; callers define their own)
VOP, VON, VFBP, VFBN, NET20, NET2 = 0, 1, 2, 3, 4, 5
NNODES = 6
GND = ("v", 0.0)        # AC ground
VDD = ("v", 0.0)        # ideal supply -> AC ground


def _stamp_adm(Y, RHS, P, Q, y):
    """Two-terminal admittance y between terminals P and Q."""
    if P[0] == "n":
        Y[P[1], P[1]] += y
        if Q[0] == "n":
            Y[P[1], Q[1]] -= y
        else:
            RHS[P[1]] += y * Q[1]
    if Q[0] == "n":
        Y[Q[1], Q[1]] += y
        if P[0] == "n":
            Y[Q[1], P[1]] -= y
        else:
            RHS[Q[1]] += y * P[1]


def _stamp_vccs(Y, RHS, d, g, s, gm):
    """Transconductance VCCS: drain current i_d = gm*(Vg-Vs) flows from drain
    to source through the device. Canonical MNA stamp:

        Y[d,g] += gm   Y[d,s] -= gm
        Y[s,g] -= gm   Y[s,s] += gm

    Terminals that are known AC voltages move to the RHS as -c*Vknown.
    """
    def addrow(node, term, c):
        # adds c*term to the KCL equation of `node`
        if node[0] != "n":
            return
        if term[0] == "n":
            Y[node[1], term[1]] += c
        else:                       # known AC voltage -> move to RHS
            RHS[node[1]] -= c * term[1]
    addrow(d, g, +gm); addrow(d, s, -gm)
    addrow(s, g, -gm); addrow(s, s, +gm)


def _branch_incidence(vsources, idx, n):
    """n×m incidence matrix B of ideal voltage sources for the bordered MNA system
    ``[[Y, B], [B^T, 0]]``: column k has +1 at the source's p node and -1 at its q node
    (solved nodes only; rail terminals drop out). Used by the periodic solvers (PSS
    monodromy, PAC/PNoise harmonic balance) to carry the branch-current unknowns.
    Each ``vsources`` entry is ``(name, p, q, value)``."""
    import numpy as np
    B = np.zeros((n, len(vsources)))
    for k, vs in enumerate(vsources):
        p, q = vs[1], vs[2]
        if p in idx:
            B[idx[p], k] += 1.0
        if q in idx:
            B[idx[q], k] -= 1.0
    return B


def _stamp_vsource(Y, RHS, P, Q, bi, E):
    """Ideal voltage source (true MNA). Branch current I_b (P->Q through the source
    interior) is the unknown at index `bi`; the constraint row enforces V_P - V_Q = E.
    P/Q are ("n", idx) solved nodes or ("v", val) known AC voltages. A DC bias source
    has E=0 -> an AC short. Frequency-independent: stamp into G / RHS_G.

        KCL_P:  Y[P,bi] += 1     KCL_Q:  Y[Q,bi] -= 1
        BR_b:   Y[bi,P] += 1, Y[bi,Q] -= 1, RHS[bi] = E - V_P(known) + V_Q(known)
    """
    rhs_b = complex(E)
    if P[0] == "n":
        Y[P[1], bi] += 1.0          # branch current leaves node P
        Y[bi, P[1]] += 1.0          # constraint: +V_P
    else:
        rhs_b -= P[1]               # known V_P -> RHS
    if Q[0] == "n":
        Y[Q[1], bi] -= 1.0          # branch current enters node Q
        Y[bi, Q[1]] -= 1.0          # constraint: -V_Q
    else:
        rhs_b += Q[1]               # known V_Q -> RHS
    RHS[bi] += rhs_b


def _stamp_mos(Y, RHS, d, g, s, gm, gds, Cgs, Cgd, jw):
    # gds between drain and source
    _stamp_adm(Y, RHS, d, s, gds)
    # Cgs between gate and source, Cgd between gate and drain
    _stamp_adm(Y, RHS, g, s, jw * Cgs)
    _stamp_adm(Y, RHS, g, d, jw * Cgd)
    # transconductance
    _stamp_vccs(Y, RHS, d, g, s, gm)


def _stamp_mos_lti(G, C, RHS_G, RHS_C, d, g, s, gm, gds, Cgs, Cgd):
    """Split a MOS small-signal stamp into frequency-independent G and C.

    The per-frequency matrix is exactly ``Y(w) = G + jw*C`` and the RHS is
    ``RHS_G + jw*RHS_C``.  This lets callers batch many frequency points without
    re-running the Python stamping loops for every point.
    """
    _stamp_adm(G, RHS_G, d, s, gds)
    _stamp_adm(C, RHS_C, g, s, Cgs)
    _stamp_adm(C, RHS_C, g, d, Cgd)
    _stamp_vccs(G, RHS_G, d, g, s, gm)
