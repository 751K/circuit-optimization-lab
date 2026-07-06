/** Public surface of the RF adapter/canvas layer (F2). */
export { default as Canvas } from "./Canvas";
export { nodeTypes } from "./nodeTypes";
export {
  domainToRf,
  domainToRfNode,
  rfToDomainNode,
  domainToRfEdge,
  rfToDomainEdge,
  netClass,
  railNetsOf,
  junctionPortsByNode,
} from "./adapter";
export type { RfNode, RfEdge, RfNodeData, RfEdgeData, PortNets } from "./adapter";
export { polarityOf, shortModel, fmtValue } from "./polarity";
export type { Polarity } from "./polarity";
