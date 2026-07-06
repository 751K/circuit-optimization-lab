/** Public surface of the editor state layer (F2). */
export { useEditor } from "./store";
export type { EditorState, Selection } from "./store";
export {
  newNode,
  newMosfet,
  newResistor,
  newCapacitor,
  newRail,
  newOutput,
  nextName,
  nextRailName,
} from "./factory";
export type { NewNodeOptions } from "./factory";
