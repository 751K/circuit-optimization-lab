/**
 * F4 edge-synthesis & layout tests.
 *
 * The import-side edge synthesis changed from a star (all ports -> first port)
 * to a nearest-neighbor chain. The hard invariant is that this must NOT change
 * connectivity: for every fixture, the connected components induced by the new
 * chain edges must equal the components induced by a star over the same ports,
 * which in turn must equal the ports grouped by resolved net name. If any of
 * these three partitions diverge, a round-trip would silently rewire the
 * circuit. We assert all three agree, per fixture.
 *
 * Also covered:
 *  - junctionPortsByNode: a pure edge-count -> tee-dot predicate.
 *  - barycenterReorder: deterministic (same input -> same layout) and pinned-
 *    column safety (a column with a stored position never moves).
 */
import { describe, expect, it } from "vitest";
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { CircuitJson } from "./circuit";
import { circuitJsonToGraph } from "./toGraph";
import { resolveNets } from "./toJson";
import { barycenterReorder } from "./util";
import { junctionPortsByNode } from "../canvas/adapter";
import type { GraphEdge } from "./graph";

const FIX_DIR = join(dirname(fileURLToPath(import.meta.url)), "__fixtures__");

function loadFixtures(): { name: string; json: CircuitJson }[] {
  return readdirSync(FIX_DIR)
    .filter((f) => f.endsWith(".json"))
    .sort()
    .map((f) => ({
      name: f,
      json: JSON.parse(readFileSync(join(FIX_DIR, f), "utf-8")) as CircuitJson,
    }));
}

const fixtures = loadFixtures();

// ── union-find, standalone (independent of the production one) ────────────
class UF {
  private p = new Map<string, string>();
  find(x: string): string {
    let r = this.p.get(x);
    if (r === undefined) {
      this.p.set(x, x);
      return x;
    }
    while (r !== x) {
      x = r;
      r = this.p.get(x) ?? x;
    }
    return x;
  }
  union(a: string, b: string): void {
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra !== rb) this.p.set(ra, rb);
  }
}

const SEP = String.fromCharCode(31);
const pkey = (node: string, port: string): string => `${node}${SEP}${port}`;

/** Partition port keys into a canonical set-of-sets given an edge list. */
function componentsFromEdges(
  portKeys: string[],
  edges: { source: { node: string; port: string }; target: { node: string; port: string } }[],
): Set<string>[] {
  const uf = new UF();
  for (const k of portKeys) uf.find(k);
  for (const e of edges) uf.union(pkey(e.source.node, e.source.port), pkey(e.target.node, e.target.port));
  const groups = new Map<string, Set<string>>();
  for (const k of portKeys) {
    const root = uf.find(k);
    (groups.get(root) ?? groups.set(root, new Set()).get(root)!).add(k);
  }
  return [...groups.values()];
}

/** Canonicalize a partition into sorted "a|b|c" strings for set comparison. */
function canon(parts: Set<string>[]): Set<string> {
  return new Set(parts.map((s) => [...s].sort().join("|")).sort());
}

/** A star over the same port groups (all -> first), for the reference partition. */
function starEdges(graph: ReturnType<typeof circuitJsonToGraph>["graph"]): GraphEdge[] {
  const byNet = new Map<string, { node: string; port: string }[]>();
  for (const n of graph.nodes) {
    for (const p of n.ports) {
      if (p.originalNet === undefined) continue;
      (byNet.get(p.originalNet) ?? byNet.set(p.originalNet, []).get(p.originalNet)!).push({
        node: n.id,
        port: p.id,
      });
    }
  }
  const edges: GraphEdge[] = [];
  for (const [net, ports] of byNet) {
    for (let i = 1; i < ports.length; i++) {
      edges.push({ id: `s:${net}:${i}`, source: ports[0]!, target: ports[i]! });
    }
  }
  return edges;
}

describe("chain edge synthesis preserves connectivity", () => {
  for (const { name, json } of fixtures) {
    it(`${name}: chain components == star components == net groups`, () => {
      const { graph } = circuitJsonToGraph(json);
      const portKeys = graph.nodes.flatMap((n) => n.ports.map((p) => pkey(n.id, p.id)));

      // 1) partition induced by the (new) chain edges the loader synthesized.
      const chainParts = canon(componentsFromEdges(portKeys, graph.edges));

      // 2) partition induced by a star over the same net-shared ports.
      const starParts = canon(componentsFromEdges(portKeys, starEdges(graph)));

      // 3) partition by resolved net name (the ground-truth electrical grouping).
      const { portNet } = resolveNets(graph);
      const byNet = new Map<string, Set<string>>();
      for (const k of portKeys) {
        const net = portNet.get(k)!;
        (byNet.get(net) ?? byNet.set(net, new Set()).get(net)!).add(k);
      }
      const netParts = canon([...byNet.values()]);

      expect(chainParts).toEqual(starParts);
      expect(chainParts).toEqual(netParts);
    });
  }

  it("a chain over N ports uses exactly N-1 edges (same count as a star)", () => {
    for (const { json } of fixtures) {
      const { graph } = circuitJsonToGraph(json);
      // group synthesized edges' endpoints back by resolved net, count ports/edges
      const { portNet } = resolveNets(graph);
      const portsPerNet = new Map<string, number>();
      for (const n of graph.nodes)
        for (const p of n.ports) {
          const net = portNet.get(pkey(n.id, p.id))!;
          portsPerNet.set(net, (portsPerNet.get(net) ?? 0) + 1);
        }
      const edgesPerNet = new Map<string, number>();
      for (const e of graph.edges) {
        const net = portNet.get(pkey(e.source.node, e.source.port))!;
        edgesPerNet.set(net, (edgesPerNet.get(net) ?? 0) + 1);
      }
      for (const [net, ports] of portsPerNet) {
        const edges = edgesPerNet.get(net) ?? 0;
        // A single-port net has 0 edges; an N-port net has N-1.
        expect(edges).toBe(Math.max(0, ports - 1));
      }
    }
  });
});

describe("junctionPortsByNode", () => {
  it("marks a port that carries >=2 edges, not one that carries a single edge", () => {
    const edges: GraphEdge[] = [
      { id: "e1", source: { node: "A", port: "p" }, target: { node: "B", port: "q" } },
      { id: "e2", source: { node: "A", port: "p" }, target: { node: "C", port: "r" } },
      { id: "e3", source: { node: "D", port: "x" }, target: { node: "E", port: "y" } },
    ];
    const j = junctionPortsByNode(edges);
    // A.p sits on two edges -> a tee.
    expect(j.get("A")?.has("p")).toBe(true);
    // B.q / C.r / D.x / E.y each sit on exactly one edge -> no dot.
    expect(j.get("B")).toBeUndefined();
    expect(j.get("D")).toBeUndefined();
  });

  it("counts a port appearing as both source and target of different edges", () => {
    const edges: GraphEdge[] = [
      { id: "e1", source: { node: "N", port: "k" }, target: { node: "A", port: "a" } },
      { id: "e2", source: { node: "B", port: "b" }, target: { node: "N", port: "k" } },
    ];
    const j = junctionPortsByNode(edges);
    expect(j.get("N")?.has("k")).toBe(true);
  });
});

describe("barycenterReorder", () => {
  /** Two columns: col x=0 has {A,B,C}; col x=240 has {P,Q}. Edges pull order. */
  function scene(): { id: string; position: [number, number] }[] {
    return [
      { id: "A", position: [0, 40] },
      { id: "B", position: [0, 160] },
      { id: "C", position: [0, 280] },
      { id: "P", position: [240, 40] },
      { id: "Q", position: [240, 160] },
    ];
  }

  it("is deterministic: same input -> identical layout", () => {
    const adj = new Map<string, Set<string>>([
      ["A", new Set(["Q"])],
      ["Q", new Set(["A"])],
      ["C", new Set(["P"])],
      ["P", new Set(["C"])],
    ]);
    const auto = new Set(["A", "B", "C", "P", "Q"]);
    const n1 = scene();
    const n2 = scene();
    barycenterReorder(n1, adj, auto);
    barycenterReorder(n2, adj, auto);
    expect(n1).toEqual(n2);
  });

  it("only permutes onto the column's existing y-slots (spacing unchanged)", () => {
    const nodes = scene();
    const before = nodes.filter((n) => n.position[0] === 0).map((n) => n.position[1]).sort((a, b) => a - b);
    const adj = new Map<string, Set<string>>([
      ["A", new Set(["Q"])],
      ["C", new Set(["P"])],
    ]);
    barycenterReorder(nodes, adj, new Set(nodes.map((n) => n.id)));
    const after = nodes.filter((n) => n.position[0] === 0).map((n) => n.position[1]).sort((a, b) => a - b);
    expect(after).toEqual(before);
  });

  it("never moves a column that contains a pinned (non-auto) node", () => {
    const nodes = scene();
    const adj = new Map<string, Set<string>>([
      ["A", new Set(["Q"])],
      ["C", new Set(["P"])],
    ]);
    // B is pinned -> its whole x=0 column is frozen.
    const auto = new Set(["A", "C", "P", "Q"]);
    const snapshot = nodes.filter((n) => n.position[0] === 0).map((n) => ({ id: n.id, y: n.position[1] }));
    barycenterReorder(nodes, adj, auto);
    const now = nodes.filter((n) => n.position[0] === 0).map((n) => ({ id: n.id, y: n.position[1] }));
    expect(now).toEqual(snapshot);
  });

  it("a node with no neighbors keeps a stable slot", () => {
    const nodes = scene();
    // Only A<->Q linked; B and C have no neighbors. Result must stay deterministic.
    const adj = new Map<string, Set<string>>([["A", new Set(["Q"])]]);
    const out1 = scene();
    barycenterReorder(nodes, adj, new Set(nodes.map((n) => n.id)));
    barycenterReorder(out1, adj, new Set(out1.map((n) => n.id)));
    expect(nodes).toEqual(out1);
  });
});
