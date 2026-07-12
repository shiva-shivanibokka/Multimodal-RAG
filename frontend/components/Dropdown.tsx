"use client";

import { useEffect, useRef, useState } from "react";

export type Opt = { value: string; label: string; note?: string };

/** Custom dark-glass dropdown — replaces the native <select> so the option
 * list matches the theme (native option popups are browser-chrome white). */
export function Dropdown({
  value,
  options,
  onChange,
  ariaLabel,
  tip,
}: {
  value: string;
  options: Opt[];
  onChange: (v: string) => void;
  ariaLabel?: string;
  tip?: string;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function onDoc(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  const current = options.find((o) => o.value === value) ?? options[0];

  return (
    <div className="dd" ref={ref} data-tip={tip}>
      <button
        type="button"
        className="dd-trigger"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={ariaLabel}
        onClick={() => setOpen((o) => !o)}
        onKeyDown={(e) => {
          if (e.key === "Escape") setOpen(false);
        }}
      >
        <span className="dd-val">{current?.label ?? "—"}</span>
        <span className="dd-caret" aria-hidden="true" />
      </button>
      {open && (
        <ul className="dd-list" role="listbox" aria-label={ariaLabel}>
          {options.map((o) => (
            <li
              key={o.value}
              role="option"
              aria-selected={o.value === value}
              className={`dd-opt ${o.value === value ? "sel" : ""}`}
              onClick={() => {
                onChange(o.value);
                setOpen(false);
              }}
            >
              <span className="dd-opt-label">{o.label}</span>
              {o.note && <span className="dd-opt-note">{o.note}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
