/**
 * A schematic junction dot: a small filled circle drawn over a handle when that
 * port carries >=2 edges (a tee). Positioned to sit exactly on the handle via a
 * per-side class (`jdot-top` / `jdot-left` / ...), mirroring the Handle's
 * Position so it lands on the connection point regardless of node size.
 */
export type JunctionSide = "top" | "bottom" | "left" | "right";

export default function Junction({
  active,
  side,
}: {
  active: boolean;
  side: JunctionSide;
}) {
  if (!active) return null;
  return <span className={`jdot jdot-${side}`} aria-hidden="true" />;
}
