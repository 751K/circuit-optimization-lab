/**
 * Device-symbol helpers used by the mosfet node component + inspector.
 *
 * Polarity is inferred from the model-type key: a key whose last dotted segment
 * (or the whole key) contains "pmos" is P-type; everything else N-type. This
 * covers "sky130.pmos", "freepdk45.pmos", bare "pmos", and "pmos_tft". A device
 * with no model set defaults to N-type (the neutral display).
 */
export type Polarity = "nmos" | "pmos";

export function polarityOf(model: string | undefined): Polarity {
  if (!model) return "nmos";
  const seg = model.includes(".") ? model.slice(model.lastIndexOf(".") + 1) : model;
  return seg.toLowerCase().includes("pmos") ? "pmos" : "nmos";
}

/** Short model label for on-node display: drop the PDK prefix ("sky130.nmos" -> "nmos"). */
export function shortModel(model: string | undefined): string {
  if (!model) return "";
  return model.includes(".") ? model.slice(model.lastIndexOf(".") + 1) : model;
}

/** Compact engineering-ish formatting of a scalar for on-node labels (2 sig figs-ish). */
export function fmtValue(v: number): string {
  if (v === 0) return "0";
  const abs = Math.abs(v);
  if (abs >= 1e9) return `${trimNum(v / 1e9)}G`;
  if (abs >= 1e6) return `${trimNum(v / 1e6)}M`;
  if (abs >= 1e3) return `${trimNum(v / 1e3)}k`;
  if (abs >= 1) return trimNum(v);
  if (abs >= 1e-3) return `${trimNum(v * 1e3)}m`;
  if (abs >= 1e-6) return `${trimNum(v * 1e6)}u`;
  if (abs >= 1e-9) return `${trimNum(v * 1e9)}n`;
  if (abs >= 1e-12) return `${trimNum(v * 1e12)}p`;
  if (abs >= 1e-15) return `${trimNum(v * 1e15)}f`;
  return v.toExponential(2);
}

function trimNum(v: number): string {
  return Number(v.toFixed(3)).toString();
}
