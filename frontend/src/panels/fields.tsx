/**
 * Small controlled/uncontrolled field primitives for the inspector. They commit
 * on blur (and Enter) so a partial keystroke never triggers a store write /
 * revalidation. Numbers parse scientific notation (2e-12) and reject NaN.
 */
import { useEffect, useState } from "react";

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
  onCommit,
}: {
  label: string;
  value: number | undefined;
  /** When true, an empty entry commits `undefined` (clears the field). */
  allowEmpty?: boolean;
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
    const n = Number(trimmed);
    if (Number.isNaN(n)) {
      setDraft(value === undefined ? "" : String(value)); // revert bad input
      return;
    }
    if (n !== value) onCommit(n);
  };

  return (
    <label className="field">
      <span>{label}</span>
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
