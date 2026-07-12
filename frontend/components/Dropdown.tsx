"use client";

import { useEffect, useRef, useState } from "react";

export type Opt = { value: string; label: string; note?: string };

/** Custom dark-glass dropdown — replaces the native <select> so the option
 * list matches the theme (native option popups are browser-chrome white).
 *
 * Keyboard (WAI-ARIA "Collapsible Dropdown Listbox" pattern):
 * - Trigger: Enter / Space / ArrowDown / ArrowUp opens the list.
 * - List: ArrowDown/ArrowUp move `highlightedIndex` (exposed via
 *   `aria-activedescendant` on the listbox), Home/End jump to the ends,
 *   Enter/Space select the highlighted option and close, Escape closes
 *   without selecting and returns focus to the trigger.
 */
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
  const [highlightedIndex, setHighlightedIndex] = useState(0);
  const ref = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  const listId = useRef(`dd-list-${Math.random().toString(36).slice(2)}`).current;

  useEffect(() => {
    function onDoc(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  // Move focus onto the listbox itself when it opens, so arrow keys go
  // straight to the list's own onKeyDown handler.
  useEffect(() => {
    if (open) listRef.current?.focus();
  }, [open]);

  const current = options.find((o) => o.value === value) ?? options[0];

  function openList() {
    const idx = options.findIndex((o) => o.value === value);
    setHighlightedIndex(idx >= 0 ? idx : 0);
    setOpen(true);
  }

  function closeList(returnFocus: boolean) {
    setOpen(false);
    if (returnFocus) triggerRef.current?.focus();
  }

  function selectHighlighted() {
    const opt = options[highlightedIndex];
    if (opt) onChange(opt.value);
    closeList(true);
  }

  function optionId(i: number) {
    return `${listId}-opt-${i}`;
  }

  return (
    <div className="dd" ref={ref} data-tip={tip}>
      <button
        ref={triggerRef}
        type="button"
        className="dd-trigger"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={listId}
        aria-label={ariaLabel}
        onClick={() => (open ? closeList(false) : openList())}
        onKeyDown={(e) => {
          if (e.key === "Escape") {
            setOpen(false);
            return;
          }
          if (!open && (e.key === "ArrowDown" || e.key === "ArrowUp" || e.key === "Enter" || e.key === " ")) {
            e.preventDefault();
            openList();
          }
        }}
      >
        <span className="dd-val">{current?.label ?? "—"}</span>
        <span className="dd-caret" aria-hidden="true" />
      </button>
      {open && (
        <ul
          ref={listRef}
          id={listId}
          className="dd-list"
          role="listbox"
          aria-label={ariaLabel}
          tabIndex={-1}
          aria-activedescendant={optionId(highlightedIndex)}
          onKeyDown={(e) => {
            switch (e.key) {
              case "ArrowDown":
                e.preventDefault();
                setHighlightedIndex((i) => Math.min(i + 1, options.length - 1));
                break;
              case "ArrowUp":
                e.preventDefault();
                setHighlightedIndex((i) => Math.max(i - 1, 0));
                break;
              case "Home":
                e.preventDefault();
                setHighlightedIndex(0);
                break;
              case "End":
                e.preventDefault();
                setHighlightedIndex(options.length - 1);
                break;
              case "Enter":
              case " ":
                e.preventDefault();
                selectHighlighted();
                break;
              case "Escape":
                e.preventDefault();
                closeList(true);
                break;
              case "Tab":
                setOpen(false);
                break;
            }
          }}
        >
          {options.map((o, i) => (
            <li
              key={o.value}
              id={optionId(i)}
              role="option"
              aria-selected={o.value === value}
              className={`dd-opt ${o.value === value ? "sel" : ""} ${i === highlightedIndex ? "hl" : ""}`}
              onMouseEnter={() => setHighlightedIndex(i)}
              onClick={() => {
                onChange(o.value);
                closeList(true);
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
