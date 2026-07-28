/**
 * Schematic auto-layout — rows are the supply stack, columns are the signal path.
 *
 * The layout this replaces bucketed nodes by *kind*: rails in one column, then
 * every mosfet in the next, then resistors, then capacitors. That is a
 * categorical arrangement, and it hides the two things an analog schematic is
 * actually read by:
 *
 *   vertical   — the supply stack. VDD on top, ground at the bottom, and each
 *                device drawn at the height of the conduction path it sits in.
 *   horizontal — the branch. One column per signal path, so a differential pair
 *                sits side by side with its load devices directly above it.
 *
 * Both are reconstructed here from the netlist alone, with no topology
 * templates and nothing hard-coded about "OTA" or "mirror":
 *
 *  1. Build the *conduction* graph over nets: drain-source for a mosfet, a-b for
 *     a resistor. A gate is control, not conduction, and a capacitor conducts
 *     only at AC — including one would let a compensation cap collapse two
 *     stages onto the same rank.
 *  2. Give each net an **altitude** in [0, 1] from its hop distance to the
 *     highest and the lowest rail: `dTop / (dTop + dBot)`. The top rail is 0,
 *     ground is 1, and an internal node lands proportionally between them. A
 *     device's altitude is the midpoint of the two nets its channel spans.
 *  3. Distinct altitudes become rows. A 5T OTA sorts itself into load pair /
 *     input pair / tail source — the textbook drawing — purely from hop counts.
 *  4. Order within each row by the median-of-neighbours crossing heuristic, then
 *     solve the x coordinates as an **isotonic regression**: every node is
 *     pulled to the median x of its neighbours in other rows, subject to keeping
 *     its row's left-to-right order and a minimum column gap. Pool-adjacent-
 *     violators gives the exact least-squares answer to "sit as close to your
 *     neighbours as the ordering allows", so a mirror lands over the branch it
 *     feeds instead of merely somewhere in the same row.
 *
 * Two placements are special-cased because a schematic draws them that way:
 *  - A supply rail is one node spanning the top (or bottom) of the drawing, at
 *    the median column of what it feeds — a bus bar, not a point with N wires
 *    fanning out of it.
 *  - A rail that only drives gates (`vinp`, `vbias`) is a *tap*. With one
 *    consumer it is spliced into that consumer's row immediately to its left, so
 *    the wire is one column long; with several it joins the row nearest its
 *    consumers and is placed by the same median rule.
 *
 * Everything is deterministic: the same netlist gives the same coordinates, so
 * the round-trip invariant that positions are reproducible still holds.
 */
import type { NodeKind, Position } from "./graph";

/** Horizontal pitch between columns. Wider than the widest node plus its label. */
export const COL_W = 220;
/** Vertical pitch between rows. Leaves room for the net labels above/below. */
export const ROW_H = 150;

/**
 * A net with more ports than this is a bus. It is skipped when building the
 * adjacency used for column placement: a global rail touching twenty devices
 * says nothing about which of them belongs left of which, and letting it vote
 * would drag every column toward one mean.
 */
const BUS_PORTS = 10;

/** Number of ordering sweeps and of coordinate relaxation passes. */
const ORDER_PASSES = 4;
const COORD_PASSES = 6;

/** The node shape this module needs: an id, a kind, and each port's net. */
export interface LayoutNode {
  id: string;
  kind: NodeKind;
  ports: { id: string; net: string }[];
  /**
   * Device drawn with its gate on the right. Only affects which side a bias tap
   * is spliced onto — put it left of a mirrored device and its wire has to loop
   * back around the body, which is the thing a tap exists to avoid.
   */
  mirrored?: boolean;
}

/**
 * Ports that carry DC conduction current, per kind. Deliberately excludes the
 * mosfet gate (control) and the capacitor (an AC-only path — see the header).
 */
const CONDUCTION: Partial<Record<NodeKind, [string, string]>> = {
  mosfet: ["D", "S"],
  resistor: ["a", "b"],
};

/**
 * Ports whose two nets *position* a node, whether or not they conduct DC. A
 * capacitor is placed between the altitudes of its plates even though it never
 * defines them.
 */
const SPAN: Partial<Record<NodeKind, [string, string]>> = {
  mosfet: ["D", "S"],
  resistor: ["a", "b"],
  capacitor: ["a", "b"],
};

/** Multi-source BFS over an undirected adjacency. Insertion order → deterministic. */
function bfs(adj: Map<string, Set<string>>, sources: Iterable<string>): Map<string, number> {
  const dist = new Map<string, number>();
  const queue: string[] = [];
  for (const s of sources) {
    if (!dist.has(s)) {
      dist.set(s, 0);
      queue.push(s);
    }
  }
  for (let i = 0; i < queue.length; i++) {
    const u = queue[i]!;
    const d = dist.get(u)! + 1;
    for (const v of adj.get(u) ?? []) {
      if (!dist.has(v)) {
        dist.set(v, d);
        queue.push(v);
      }
    }
  }
  return dist;
}

/** Median of a non-empty numeric list (mean of the two middles when even). */
function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 === 1 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2;
}

/**
 * Pool-adjacent-violators: the least-squares non-decreasing fit to `v`.
 * Exported for test — it is the one piece of real numerical machinery here, and
 * the property that matters (output is non-decreasing and is the L2 projection)
 * is worth asserting directly.
 */
export function isotonic(v: number[]): number[] {
  const val: number[] = [];
  const cnt: number[] = [];
  for (const x of v) {
    val.push(x);
    cnt.push(1);
    while (val.length > 1 && val[val.length - 2]! > val[val.length - 1]!) {
      const w1 = cnt.pop()!;
      const m1 = val.pop()!;
      const w0 = cnt.pop()!;
      const m0 = val.pop()!;
      cnt.push(w0 + w1);
      val.push((m0 * w0 + m1 * w1) / (w0 + w1));
    }
  }
  const out: number[] = [];
  for (let i = 0; i < val.length; i++) {
    for (let k = 0; k < cnt[i]!; k++) out.push(val[i]!);
  }
  return out;
}

/**
 * Place one row's nodes at their desired x while keeping the row's order and a
 * minimum gap. Minimises `sum (x_i - desired_i)^2` subject to
 * `x_{i+1} >= x_i + gap`, which is exactly `isotonic` after subtracting the gap
 * ramp — so this is the optimum, not a heuristic nudge. Exported for test.
 */
export function placeRow(desired: number[], gap: number): number[] {
  const shifted = desired.map((d, i) => d - i * gap);
  return isotonic(shifted).map((y, i) => y + i * gap);
}

/** Round an altitude to a stable bucket key so equal ranks share one row. */
function rowKey(altitude: number): string {
  return altitude.toFixed(6);
}

/**
 * Lay a circuit out as a schematic. Returns a position per node id, or `null`
 * when the circuit has no resolvable top and bottom rail — with nothing to
 * measure altitude against there is no supply stack to draw, and the caller
 * falls back to the kind-column layout.
 *
 * `railPotentials` maps a rail net name to its resolved DC potential; a rail
 * whose value could not be resolved to a number is simply left out and takes no
 * part in choosing the anchors.
 */
export function schematicLayout(
  nodes: LayoutNode[],
  railPotentials: Map<string, number>,
): Map<string, Position> | null {
  if (nodes.length === 0) return null;

  // ── index the netlist ───────────────────────────────────────────────────
  const netPorts = new Map<string, { node: string; port: string }[]>();
  const byId = new Map<string, LayoutNode>();
  for (const n of nodes) {
    byId.set(n.id, n);
    for (const p of n.ports) {
      let arr = netPorts.get(p.net);
      if (!arr) {
        arr = [];
        netPorts.set(p.net, arr);
      }
      arr.push({ node: n.id, port: p.id });
    }
  }
  const netOf = (id: string, port: string): string | undefined =>
    byId.get(id)?.ports.find((p) => p.id === port)?.net;

  // ── pick the supply anchors ─────────────────────────────────────────────
  // The highest-potential rails are the top of the drawing, the lowest are the
  // bottom. Rails in between (a common-mode or bias source) are not anchors —
  // they are taps, handled further down. A deck with only one distinct rail
  // potential still anchors one end: a rail at or below 0 V is a ground and
  // everything hangs above it; anything else is a supply and everything hangs
  // below. Ranking against one end beats not ranking at all.
  // Only a rail some *element* touches can anchor anything — the rail node's own
  // port does not count. Unusable rails are dropped *before* the highest and
  // lowest are picked, not after: a rails entry nothing references (a leftover,
  // or a supply belonging to a sibling testbench) would otherwise win the
  // contest, contribute no conduction path, and leave the real supply demoted
  // from the top bus to a slot beside the device it feeds.
  const elementPorts = (net: string): number =>
    (netPorts.get(net) ?? []).filter((p) => byId.get(p.node)?.kind !== "rail").length;

  // A rail whose every element port is a mosfet gate is a *tap*: an input, a
  // bias, a clock. It is drawn beside what it drives rather than as a bus, and
  // it is not eligible to anchor the drawing — a 1.8 V clock ties a 1.8 V supply
  // on potential, and letting it share the top row puts two rails in the same
  // place with no conduction path to tell them apart.
  const gateOnly = (net: string): boolean => {
    const others = (netPorts.get(net) ?? []).filter((p) => byId.get(p.node)?.kind !== "rail");
    return others.length > 0
      && others.every((p) => byId.get(p.node)?.kind === "mosfet" && p.port === "G");
  };
  const usable = [...railPotentials]
    .filter(([net]) => elementPorts(net) > 0 && !gateOnly(net));
  if (usable.length === 0) return null;
  const potentials = usable.map(([, v]) => v);
  const hi = Math.max(...potentials);
  const lo = Math.min(...potentials);
  const topNets = new Set<string>();
  const botNets = new Set<string>();
  for (const [net, v] of usable) {
    if (hi > lo) {
      if (v === hi) topNets.add(net);
      else if (v === lo) botNets.add(net);
    } else {
      (hi <= 0 ? botNets : topNets).add(net);
    }
  }

  // ── the conduction graph over nets ──────────────────────────────────────
  const buildNetAdj = (withCaps: boolean): Map<string, Set<string>> => {
    const m = new Map<string, Set<string>>();
    const linkNets = (a: string, b: string): void => {
      if (a === b) return;
      (m.get(a) ?? m.set(a, new Set()).get(a)!).add(b);
      (m.get(b) ?? m.set(b, new Set()).get(b)!).add(a);
    };
    for (const n of nodes) {
      const pair = withCaps ? SPAN[n.kind] : CONDUCTION[n.kind];
      if (!pair) continue;
      const a = netOf(n.id, pair[0]);
      const b = netOf(n.id, pair[1]);
      if (a !== undefined && b !== undefined) linkNets(a, b);
    }
    return m;
  };
  const dcAdj = buildNetAdj(false);
  const acAdj = buildNetAdj(true);

  // ── each net's altitude, in three widening passes ───────────────────────
  // 1. DC conduction alone. This is the ranking that means something, and it is
  //    computed first so a compensation capacitor can never pull two stages onto
  //    the same rank.
  // 2. Capacitors included, for nets the DC pass left unranked — an AC-coupled
  //    input network is DC-isolated by its coupling caps but still belongs at
  //    the height it drives.
  // 3. Gate control, then diffusion. A net that reaches no rail by any passive
  //    path is placed at the height of the devices it *controls*, and that
  //    height then spreads outward along the network hanging off it. Without
  //    this the whole input network of a fully-differential OTA lands in a heap.
  const netAltitude = new Map<string, number>();
  const rankFrom = (adj: Map<string, Set<string>>): void => {
    const dTop = bfs(adj, topNets);
    const dBot = bfs(adj, botNets);
    // Depth of the deepest stack seen, so a net hanging off only one rail gets a
    // plausible altitude instead of being dropped.
    let span = 1;
    for (const d of dTop.values()) span = Math.max(span, d);
    for (const d of dBot.values()) span = Math.max(span, d);
    for (const net of netPorts.keys()) {
      if (netAltitude.has(net)) continue;
      const t = dTop.get(net);
      const b = dBot.get(net);
      if (t !== undefined && b !== undefined) {
        netAltitude.set(net, t + b === 0 ? 0.5 : t / (t + b));
      } else if (t !== undefined) {
        netAltitude.set(net, t / (t + span + 1));
      } else if (b !== undefined) {
        netAltitude.set(net, 1 - b / (b + span + 1));
      }
    }
  };
  rankFrom(dcAdj);
  rankFrom(acAdj);

  /** Mean altitude of a node's spanning nets, or undefined when neither is ranked. */
  const spanAltitude = (n: LayoutNode): number | undefined => {
    const pair = SPAN[n.kind];
    if (!pair) return undefined;
    const alts = pair
      .map((port) => netOf(n.id, port))
      .map((net) => (net === undefined ? undefined : netAltitude.get(net)))
      .filter((a): a is number => a !== undefined);
    return alts.length === 0 ? undefined : alts.reduce((s, a) => s + a, 0) / alts.length;
  };

  // Pass 3, seeded from gates: an unranked net that drives a ranked device's
  // gate sits at that device's height.
  const gatedBy = new Map<string, number[]>();
  for (const n of nodes) {
    if (n.kind !== "mosfet") continue;
    const g = netOf(n.id, "G");
    const alt = spanAltitude(n);
    if (g === undefined || alt === undefined || netAltitude.has(g)) continue;
    (gatedBy.get(g) ?? gatedBy.set(g, []).get(g)!).push(alt);
  }
  // How many diffusion hops from a conduction-ranked net. Zero means the
  // altitude is a real measurement; anything higher means it was inherited.
  const diffusion = new Map<string, number>();
  for (const [net, alts] of gatedBy) {
    netAltitude.set(net, alts.reduce((s, a) => s + a, 0) / alts.length);
    diffusion.set(net, 1);
  }
  // …and diffuse outward along the passive network hanging off those gates.
  // Each round reads the previous round's map so the result cannot depend on
  // iteration order.
  for (let round = 0; round < 12; round++) {
    const additions = new Map<string, number>();
    for (const net of netPorts.keys()) {
      if (netAltitude.has(net)) continue;
      const known = [...(acAdj.get(net) ?? [])]
        .map((nb) => netAltitude.get(nb))
        .filter((a): a is number => a !== undefined);
      if (known.length > 0) {
        additions.set(net, known.reduce((s, a) => s + a, 0) / known.length);
      }
    }
    if (additions.size === 0) break;
    for (const [net, a] of additions) {
      netAltitude.set(net, a);
      diffusion.set(net, round + 2);
    }
  }

  /**
   * How far a node hangs off the ranked circuit, as a nudge to its altitude.
   *
   * Every element of a diffusion-ranked network inherits one altitude — an
   * AC-coupled input chain all sits at the height of the gate it drives — so
   * without this the whole thing has to lay itself out along a single row, and
   * a nine-element input network makes the drawing four times wider than the
   * amplifier it feeds. Stepping each hop onto its own row instead turns that
   * run into a short descent beside the gate.
   *
   * The step is small enough that a hop can never cross a rank measured by
   * conduction: real ranks are spaced by roughly one over the stack depth, which
   * is orders of magnitude above this.
   */
  const HANG_STEP = 0.002;
  const hangOffset = (n: LayoutNode): number => {
    // A rail feeding such a network hangs off it too — its own net's depth —
    // or the source of an input chain ends up two rows from the resistor it
    // drives, which is the long wire this is meant to remove.
    const ports = SPAN[n.kind] ?? n.ports.map((p) => p.id);
    const depths = ports
      .map((port) => netOf(n.id, port))
      .map((net) => (net === undefined ? 0 : diffusion.get(net) ?? 0));
    return depths.length === 0 ? 0 : Math.max(...depths) * HANG_STEP;
  };

  // ── classify the rails ──────────────────────────────────────────────────
  const isRail = (n: LayoutNode): boolean => n.kind === "rail";
  const gateTaps = new Map<string, string[]>(); // rail node id -> consumer node ids
  const supplyIds = new Set<string>();
  for (const n of nodes) {
    if (!isRail(n)) continue;
    const net = n.ports[0]?.net;
    if (net === undefined) continue;
    // Tap first: `gateOnly` already kept these out of the anchor set, so a rail
    // reaching here as a supply really does carry current.
    if (gateOnly(net)) {
      const others = (netPorts.get(net) ?? []).filter((p) => p.node !== n.id);
      gateTaps.set(n.id, [...new Set(others.map((p) => p.node))]);
    } else if (topNets.has(net) || botNets.has(net)) {
      supplyIds.add(n.id);
    }
  }
  const supplyRails = nodes.filter((n) => supplyIds.has(n.id));

  // ── each node's altitude ────────────────────────────────────────────────
  const nodeAltitude = new Map<string, number>();
  for (const n of nodes) {
    if (n.kind === "output") continue; // a marker, placed as a satellite below
    if (isRail(n)) {
      // Supply rails get their own end rows and a tap follows its consumers; any
      // other rail sits at its net's altitude like an ordinary element.
      if (!gateTaps.has(n.id) && !supplyIds.has(n.id)) {
        const a = n.ports[0] ? netAltitude.get(n.ports[0].net) : undefined;
        if (a !== undefined) nodeAltitude.set(n.id, a + hangOffset(n));
      }
      continue;
    }
    const alt = spanAltitude(n);
    if (alt !== undefined) nodeAltitude.set(n.id, alt + hangOffset(n));
  }
  // A tap sits at the height of what it drives.
  for (const [railId, consumers] of gateTaps) {
    const alts = consumers
      .map((c) => nodeAltitude.get(c))
      .filter((a): a is number => a !== undefined);
    if (alts.length > 0) {
      nodeAltitude.set(railId, alts.reduce((s, a) => s + a, 0) / alts.length);
    }
  }

  // ── altitudes become rows ───────────────────────────────────────────────
  // Only element altitudes define the row grid; a multi-consumer tap then snaps
  // to the nearest of those rows rather than opening a row of its own.
  const gridAlts = [...new Set(
    nodes
      .filter((n) => !isRail(n) && n.kind !== "output")
      .map((n) => nodeAltitude.get(n.id))
      .filter((a): a is number => a !== undefined)
      .map(rowKey),
  )].map(Number).sort((a, b) => a - b);
  if (gridAlts.length === 0) return null;

  const nearestRow = (alt: number): number => {
    let best = 0;
    let bestD = Infinity;
    for (let i = 0; i < gridAlts.length; i++) {
      const d = Math.abs(gridAlts[i]! - alt);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best + 1; // row 0 is reserved for the top supply bus
  };

  const BOTTOM_ROW = gridAlts.length + 1;
  const ORPHAN_ROW = gridAlts.length + 2;

  const rowOf = new Map<string, number>();
  for (const n of supplyRails) {
    const net = n.ports[0]!.net;
    rowOf.set(n.id, topNets.has(net) ? 0 : BOTTOM_ROW);
  }
  for (const n of nodes) {
    if (rowOf.has(n.id) || n.kind === "output") continue;
    const alt = nodeAltitude.get(n.id);
    rowOf.set(n.id, alt === undefined ? ORPHAN_ROW : nearestRow(alt));
  }

  // ── adjacency used for ordering and column placement ────────────────────
  // Bus nets are skipped (see BUS_PORTS): they connect everything to everything
  // and would flatten the columns instead of separating the branches.
  const adj = new Map<string, Set<string>>();
  const link = (a: string, b: string): void => {
    if (a === b) return;
    (adj.get(a) ?? adj.set(a, new Set()).get(a)!).add(b);
    (adj.get(b) ?? adj.set(b, new Set()).get(b)!).add(a);
  };
  for (const [, ports] of netPorts) {
    if (ports.length > BUS_PORTS) continue;
    const owners = [...new Set(ports.map((p) => p.node))];
    for (let i = 0; i < owners.length; i++) {
      for (let k = i + 1; k < owners.length; k++) link(owners[i]!, owners[k]!);
    }
  }

  // ── row membership, in author order ─────────────────────────────────────
  // A tap driving exactly one gate is held out of the ordering entirely and
  // spliced back beside its consumer once the columns are solved, so no sweep
  // can separate the two — the point of a stub is that its wire is one column
  // long.
  const stubFor = new Map<string, string[]>(); // consumer node id -> tap rail ids
  for (const [railId, consumers] of gateTaps) {
    if (consumers.length !== 1) continue;
    const owner = consumers[0]!;
    (stubFor.get(owner) ?? stubFor.set(owner, []).get(owner)!).push(railId);
  }
  const stubIds = new Set([...stubFor.values()].flat());

  const rows = new Map<number, string[]>();
  for (const n of nodes) {
    if (n.kind === "output" || stubIds.has(n.id)) continue;
    const r = rowOf.get(n.id);
    if (r === undefined) continue;
    (rows.get(r) ?? rows.set(r, []).get(r)!).push(n.id);
  }

  // Seed each row's left-to-right order by walking its *own* connectivity rather
  // than by author order. The sweeps below only compare a node against the row
  // above or below, so anything whose neighbours all sit in its own row — an
  // AC-coupled input chain, a bias ladder — would otherwise stay wherever the
  // JSON happened to list it, which is how a nine-element input network ends up
  // stranded on the far side of the amplifier it feeds. Deterministic:
  // components and neighbours are both visited in author order.
  const authorRank = new Map(nodes.map((n, i) => [n.id, i]));
  for (const [r, members] of rows) {
    const inRow = new Set(members);
    const seen = new Set<string>();
    const seq: string[] = [];
    const visit = (id: string): void => {
      if (seen.has(id)) return;
      seen.add(id);
      seq.push(id);
      const next = [...(adj.get(id) ?? [])]
        .filter((nb) => inRow.has(nb) && !seen.has(nb))
        .sort((a, b) => authorRank.get(a)! - authorRank.get(b)!);
      for (const nb of next) visit(nb);
    };
    for (const id of members) visit(id);
    rows.set(r, seq);
  }

  // ── crossing reduction: median-of-neighbours sweeps ─────────────────────
  const rowIds = [...rows.keys()].sort((a, b) => a - b);
  // A node's key is the median *relative* position of its neighbours in the
  // reference row, so rows of different lengths compare on one scale.
  const relIndex = (r: number): Map<string, number> => {
    const members = rows.get(r) ?? [];
    const m = new Map<string, number>();
    members.forEach((id, i) =>
      m.set(id, members.length === 1 ? 0.5 : i / (members.length - 1)));
    return m;
  };
  for (let pass = 0; pass < ORDER_PASSES; pass++) {
    const down = pass % 2 === 0;
    const seq = down ? rowIds : [...rowIds].reverse();
    for (const r of seq) {
      const ref = relIndex(down ? r - 1 : r + 1);
      if (ref.size === 0) continue;
      const members = rows.get(r)!;
      // Only nodes with a neighbour in the reference row have anything to say
      // about where they belong. The rest hold their slots and the movable ones
      // are permuted among the slots they already occupy — the standard
      // treatment, and without it a node with no opinion would be sorted against
      // keys drawn from a different row and drift across the drawing.
      const movable = members
        .map((id, i) => {
          const near = [...(adj.get(id) ?? [])]
            .map((nb) => ref.get(nb))
            .filter((v): v is number => v !== undefined);
          return near.length > 0 ? { id, i, key: median(near) } : null;
        })
        .filter((v): v is { id: string; i: number; key: number } => v !== null);
      const slots = movable.map((m) => m.i);
      const sorted = [...movable].sort((a, b) => (a.key !== b.key ? a.key - b.key : a.i - b.i));
      const next = [...members];
      slots.forEach((slot, k) => {
        next[slot] = sorted[k]!.id;
      });
      rows.set(r, next);
    }
  }

  // ── column coordinates: pull to neighbours, project back onto the order ──
  const x = new Map<string, number>();
  for (const r of rowIds) rows.get(r)!.forEach((id, i) => x.set(id, i * COL_W));
  for (let pass = 0; pass < COORD_PASSES; pass++) {
    const seq = pass % 2 === 0 ? rowIds : [...rowIds].reverse();
    for (const r of seq) {
      const members = rows.get(r)!;
      const inRow = new Set(members);
      const desired = members.map((id) => {
        const near = [...(adj.get(id) ?? [])]
          .filter((nb) => !inRow.has(nb))
          .map((nb) => x.get(nb))
          .filter((v): v is number => v !== undefined);
        return near.length > 0 ? median(near) : x.get(id)!;
      });
      placeRow(desired, COL_W).forEach((v, i) => x.set(members[i]!, v));
    }
  }

  // ── splice each single-gate tap in beside its consumer ──────────────────
  // The tap wants the column immediately outside the gate it drives — left
  // normally, right for a mirrored device, since that is the side the gate is
  // on. Everyone else wants to stay put. One more projection resolves that into
  // an order with no overlap, moving the row as little as the constraint allows.
  for (const r of rowIds) {
    const members = rows.get(r)!;
    const spliced: string[] = [];
    const desired: number[] = [];
    for (const id of members) {
      const stubs = stubFor.get(id) ?? [];
      const after = byId.get(id)?.mirrored === true;
      if (!after) {
        for (const stub of stubs) {
          rowOf.set(stub, r);
          spliced.push(stub);
          desired.push(x.get(id)! - COL_W);
        }
      }
      spliced.push(id);
      desired.push(x.get(id)!);
      if (after) {
        for (const stub of stubs) {
          rowOf.set(stub, r);
          spliced.push(stub);
          desired.push(x.get(id)! + COL_W);
        }
      }
    }
    if (spliced.length === members.length) continue;
    rows.set(r, spliced);
    placeRow(desired, COL_W).forEach((v, i) => x.set(spliced[i]!, v));
  }

  // ── supply rails span their consumers; outputs hang off the right ───────
  const out = new Map<string, Position>();
  for (const r of rowIds) {
    for (const id of rows.get(r)!) out.set(id, [x.get(id)!, r * ROW_H]);
  }
  // A supply bus centres on what it feeds. Two of them do not need keeping
  // apart: the ordering sweeps pull the devices on one rail together, so their
  // medians separate on their own — see the no-overlap assertion over the
  // example corpus in layout.test.ts.
  for (const n of supplyRails) {
    const net = n.ports[0]!.net;
    const consumers = (netPorts.get(net) ?? [])
      .filter((p) => p.node !== n.id)
      .map((p) => out.get(p.node)?.[0])
      .filter((v): v is number => v !== undefined);
    out.set(n.id, [
      consumers.length > 0 ? median(consumers) : 0,
      (topNets.has(net) ? 0 : BOTTOM_ROW) * ROW_H,
    ]);
  }
  // An output marker continues its net to the right of the last thing on it, on
  // the nearest row. It belongs to no row, so it is the one node that can land
  // on top of another — step it right until the cell is free.
  const occupied = (px: number, py: number, self: string): boolean => {
    for (const [id, p] of out) {
      if (id === self) continue;
      if (Math.abs(p[1] - py) < ROW_H / 2 && Math.abs(p[0] - px) < COL_W) return true;
    }
    return false;
  };
  for (const n of nodes) {
    if (n.kind !== "output") continue;
    const net = n.ports[0]?.net;
    const peers = (netPorts.get(net ?? "") ?? [])
      .filter((p) => p.node !== n.id)
      .map((p) => out.get(p.node))
      .filter((p): p is Position => p !== undefined);
    if (peers.length === 0) {
      out.set(n.id, [0, ORPHAN_ROW * ROW_H]);
      continue;
    }
    const py = Math.round(median(peers.map((p) => p[1])) / ROW_H) * ROW_H;
    let px = Math.max(...peers.map((p) => p[0])) + COL_W;
    for (let guard = 0; guard < nodes.length && occupied(px, py, n.id); guard++) px += COL_W;
    out.set(n.id, [px, py]);
  }

  // Every node must come back placed — a caller that silently lost one would
  // stack it on the origin under whatever is already there.
  for (const n of nodes) if (!out.has(n.id)) out.set(n.id, [0, ORPHAN_ROW * ROW_H]);

  // Normalise so the drawing starts near the origin regardless of how far the
  // relaxation drifted; React Flow's fitView does the rest.
  let minX = Infinity;
  for (const p of out.values()) minX = Math.min(minX, p[0]);
  if (Number.isFinite(minX) && minX !== 0) {
    for (const [id, p] of out) out.set(id, [Math.round(p[0] - minX), p[1]]);
  } else {
    for (const [id, p] of out) out.set(id, [Math.round(p[0]), p[1]]);
  }
  return out;
}
