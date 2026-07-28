/**
 * Small controlled/uncontrolled field primitives for the inspector. They commit
 * on blur (and Enter) so a partial keystroke never triggers a store write /
 * revalidation.
 *
 * Numbers accept engineering notation — `20u`, `1n`, `2.2k` — as well as plain
 * and exponent forms. Circuit values are written with suffixes, not exponents,
 * so requiring `2e-5` for a 20 µs stop time makes the field hostile to the
 * notation the domain actually uses. Unparseable input reverts rather than
 * committing a wrong number.
 */
import { useEffect, useState } from "react";
import { formatValue, parseEngineering } from "../results/format";

export function TextField({
  label,
  value,
  onCommit,
}: {
  label: string;
  value: string;
  onCommit: (v: string) => void;
}) {
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);
  const commit = (): void => {
    if (draft !== value) onCommit(draft);
  };
  return (
    <label className="field">
      <span>{label}</span>
      <input
        type="text"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") (e.target as HTMLInputElement).blur();
        }}
        spellCheck={false}
      />
    </label>
  );
}

export function NumberField({
  label,
  value,
  allowEmpty = false,
  unit,
  onCommit,
}: {
  label: string;
  value: number | undefined;
  /** When true, an empty entry commits `undefined` (clears the field). */
  allowEmpty?: boolean;
  /** SI unit; when given, the committed value is echoed back in engineering form. */
  unit?: string;
  onCommit: (v: number | undefined) => void;
}) {
  const [draft, setDraft] = useState(value === undefined ? "" : String(value));
  useEffect(() => setDraft(value === undefined ? "" : String(value)), [value]);

  const commit = (): void => {
    const trimmed = draft.trim();
    if (trimmed === "") {
      if (allowEmpty) onCommit(undefined);
      else setDraft(value === undefined ? "" : String(value)); // revert
      return;
    }
    const n = parseEngineering(trimmed);
    if (n === null) {
      setDraft(value === undefined ? "" : String(value)); // revert bad input
      return;
    }
    if (n !== value) onCommit(n);
    else setDraft(String(n));   // normalize "20u" -> "0.00002" on re-entry
  };

  // Echo what the entry parsed to, so "20u" is visibly 20 µs before it is run.
  const parsed = unit ? parseEngineering(draft) : null;
  const echo = parsed !== null && String(parsed) !== draft.trim()
    ? formatValue(parsed, unit)
    : null;

  return (
    <label className="field">
      <span>
        {label}
        {echo && <span className="field-echo"> = {echo}</span>}
      </span>
      <input
        type="text"
        inputMode="decimal"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") (e.target as HTMLInputElement).blur();
        }}
        spellCheck={false}
      />
    </label>
  );
}
