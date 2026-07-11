"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { fetchEvalReport, type EvalReport } from "@/lib/backend";
import sampleReport from "./report.sample.json";

const MODES = ["dense", "hybrid", "cross_modal", "caption_baseline"] as const;
const MODE_LABELS: Record<string, string> = {
  dense: "Dense",
  hybrid: "Hybrid",
  cross_modal: "Cross-modal",
  caption_baseline: "Caption baseline",
};
// Fixed categorical slot order (dataviz skill palette.md, slots 1-4) --
// never cycled or reassigned per render. Referenced via CSS custom
// properties (see .eval-viz below) so light/dark swap in one place.
const MODE_VARS: Record<string, string> = {
  dense: "var(--series-dense)",
  hybrid: "var(--series-hybrid)",
  cross_modal: "var(--series-cross-modal)",
  caption_baseline: "var(--series-caption-baseline)",
};

const METRICS = ["recall_at_1", "recall_at_5", "mrr", "citation_accuracy"] as const;
const METRIC_LABELS: Record<string, string> = {
  recall_at_1: "Recall@1",
  recall_at_5: "Recall@5",
  mrr: "MRR",
  citation_accuracy: "Citation accuracy",
};

const RUN_CMD = "python backend/eval/run_eval.py --out backend/eval/report.json";

function pct(n: number): string {
  return `${(n * 100).toFixed(1)}%`;
}

/** Grouped column chart: one group per metric, one bar per retrieval mode --
 * adjacent bar order (dense, hybrid, cross_modal, caption_baseline) puts the
 * two headline comparisons (hybrid vs dense; cross-modal vs caption-baseline)
 * side by side in every group. Follows the dataviz skill: fixed hue order,
 * 24px-capped columns with rounded caps, 2px surface gaps, legend + a table
 * view for full accessibility (this palette's aqua/yellow slots fall under
 * the skill's 3:1 relief rule -- the table below is that relief). */
function EvalBarChart({ modes }: { modes: Record<string, EvalReport["modes"][string]> }) {
  const barW = 18;
  const barGap = 2;
  const groupGap = 32;
  const groupW = MODES.length * barW + (MODES.length - 1) * barGap;
  const plotH = 200;
  const marginLeft = 40;
  const marginTop = 12;
  const width = marginLeft + METRICS.length * groupW + (METRICS.length - 1) * groupGap + 12;
  const height = plotH + marginTop + 40;
  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div className="eval-viz flex flex-col gap-3">
      <style>{`
        .eval-viz { --tick: #e1e0d9; --axis: #c3c2b7; --ink-secondary: #52514e; --ink-muted: #898781; }
        @media (prefers-color-scheme: dark) {
          .eval-viz { --tick: #2c2c2a; --axis: #383835; --ink-secondary: #c3c2b7; --ink-muted: #898781; }
        }
      `}</style>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
        {MODES.map((m) => (
          <span key={m} className="flex items-center gap-1.5">
            <span className="inline-block size-2.5 rounded-full" style={{ background: MODE_VARS[m] }} />
            <span className="text-muted-foreground">{MODE_LABELS[m]}</span>
          </span>
        ))}
      </div>
      <div className="overflow-x-auto">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full min-w-[420px]" role="img" aria-label="Retrieval quality by mode and metric">
          {ticks.map((t) => {
            const y = marginTop + plotH - t * plotH;
            return (
              <g key={t}>
                <line x1={marginLeft} x2={width - 4} y1={y} y2={y} stroke="var(--tick)" strokeWidth={1} />
                <text x={marginLeft - 6} y={y + 3} textAnchor="end" fontSize={9} fill="var(--ink-muted)">
                  {Math.round(t * 100)}%
                </text>
              </g>
            );
          })}
          <line x1={marginLeft} x2={marginLeft} y1={marginTop} y2={marginTop + plotH} stroke="var(--axis)" strokeWidth={1} />
          <line x1={marginLeft} x2={width - 4} y1={marginTop + plotH} y2={marginTop + plotH} stroke="var(--axis)" strokeWidth={1} />

          {METRICS.map((metric, gi) => {
            const groupX = marginLeft + gi * (groupW + groupGap);
            return (
              <g key={metric}>
                {MODES.map((mode, bi) => {
                  const value = modes[mode]?.[metric] ?? 0;
                  const barH = value * plotH;
                  const x = groupX + bi * (barW + barGap);
                  const y = marginTop + plotH - barH;
                  return (
                    <rect
                      key={mode}
                      x={x}
                      y={y}
                      width={barW}
                      height={Math.max(barH, 1)}
                      rx={4}
                      fill={MODE_VARS[mode]}
                      tabIndex={0}
                    >
                      <title>
                        {MODE_LABELS[mode]} · {METRIC_LABELS[metric]}: {pct(value)}
                      </title>
                    </rect>
                  );
                })}
                <text
                  x={groupX + groupW / 2}
                  y={marginTop + plotH + 16}
                  textAnchor="middle"
                  fontSize={10}
                  fill="var(--ink-secondary)"
                >
                  {METRIC_LABELS[metric]}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Relief for the aqua/yellow slots' sub-3:1 contrast (dataviz skill):
          exact values stay available in a table, not color-matching alone. */}
      <div className="overflow-x-auto">
        <table className="w-full min-w-[420px] text-xs">
          <thead>
            <tr className="border-b text-left text-muted-foreground">
              <th className="py-1 pr-2 font-medium">Mode</th>
              {METRICS.map((m) => (
                <th key={m} className="py-1 pr-2 text-right font-medium">
                  {METRIC_LABELS[m]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {MODES.map((mode) => (
              <tr key={mode} className="border-b last:border-0">
                <td className="py-1 pr-2">{MODE_LABELS[mode]}</td>
                {METRICS.map((m) => (
                  <td key={m} className="py-1 pr-2 text-right tabular-nums">
                    {pct(modes[mode]?.[m] ?? 0)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function RefusalAccuracy({ modes }: { modes: Record<string, EvalReport["modes"][string]> }) {
  return (
    <div className="flex flex-col gap-2">
      {MODES.map((mode) => {
        const value = modes[mode]?.refusal_accuracy ?? 0;
        return (
          <div key={mode} className="flex items-center gap-3 text-sm">
            <span className="w-32 shrink-0 text-muted-foreground">{MODE_LABELS[mode]}</span>
            <div className="h-2 flex-1 rounded-full bg-muted">
              <div
                className="h-2 rounded-full"
                style={{ width: `${Math.min(value, 1) * 100}%`, background: MODE_VARS[mode] }}
              />
            </div>
            <span className="w-12 shrink-0 text-right tabular-nums">{pct(value)}</span>
          </div>
        );
      })}
    </div>
  );
}

function Headline({ report }: { report: EvalReport }) {
  const m = report.modes;
  const items: { title: string; a: string; b: string; delta: number }[] = [];
  if (m.hybrid && m.dense) {
    items.push({ title: "Hybrid vs dense", a: "hybrid", b: "dense", delta: m.hybrid.recall_at_1 - m.dense.recall_at_1 });
  }
  if (m.cross_modal && m.caption_baseline) {
    items.push({
      title: "Cross-modal vs caption-baseline",
      a: "cross_modal",
      b: "caption_baseline",
      delta: m.cross_modal.recall_at_1 - m.caption_baseline.recall_at_1,
    });
  }
  if (items.length === 0) return null;
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {items.map((it) => (
        <Card key={it.title}>
          <CardContent className="flex flex-col gap-1 pt-6">
            <span className="text-sm text-muted-foreground">{it.title} · Recall@1</span>
            <span
              className="text-2xl font-semibold"
              style={{ color: it.delta >= 0 ? "#0ca30c" : "#d03b3b" }}
            >
              {it.delta >= 0 ? "+" : ""}
              {(it.delta * 100).toFixed(1)} pts
            </span>
            <span className="text-xs text-muted-foreground">
              {MODE_LABELS[it.a]} {pct(m[it.a].recall_at_1)} vs {MODE_LABELS[it.b]} {pct(m[it.b].recall_at_1)}
            </span>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

export default function EvalDashboard() {
  const [report, setReport] = useState<EvalReport | null>(null);
  const [isSample, setIsSample] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    fetchEvalReport()
      .then((real) => {
        if (cancelled) return;
        if (real) {
          setReport(real);
          setIsSample(false);
        } else {
          setReport(sampleReport as EvalReport);
          setIsSample(true);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setReport(sampleReport as EvalReport);
          setIsSample(true);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="eval-viz mx-auto flex w-full max-w-3xl flex-col gap-6 p-8">
      <style>{`
        .eval-viz {
          --series-dense: #2a78d6; --series-hybrid: #1baf7a;
          --series-cross-modal: #eda100; --series-caption-baseline: #008300;
        }
        @media (prefers-color-scheme: dark) {
          .eval-viz {
            --series-dense: #3987e5; --series-hybrid: #199e70;
            --series-cross-modal: #c98500; --series-caption-baseline: #008300;
          }
        }
      `}</style>
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Evaluation dashboard</h1>
        <Link href="/" className="text-sm text-primary underline underline-offset-2">
          back to app
        </Link>
      </div>

      {loading && <p className="text-sm text-muted-foreground">Loading benchmark report...</p>}

      {!loading && isSample && (
        <div className="flex items-center gap-2 rounded-md border border-yellow-600/30 bg-yellow-500/10 px-3 py-2 text-sm">
          <Badge variant="outline" className="border-yellow-600/50 text-yellow-700 dark:text-yellow-400">
            Sample data
          </Badge>
          <span className="text-muted-foreground">
            illustrative only -- no benchmark has been run yet. Run it locally to see real numbers.
          </span>
        </div>
      )}

      {!loading && report && (
        <>
          <Headline report={report} />

          <Card>
            <CardHeader>
              <CardTitle>Dataset</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-wrap gap-x-8 gap-y-2 text-sm">
              <div>
                <div className="text-muted-foreground">Dataset</div>
                <div>{report.dataset}</div>
              </div>
              <div>
                <div className="text-muted-foreground">Docs</div>
                <div className="tabular-nums">{report.n_docs}</div>
              </div>
              <div>
                <div className="text-muted-foreground">Answerable</div>
                <div className="tabular-nums">{report.n_answerable}</div>
              </div>
              <div>
                <div className="text-muted-foreground">Out-of-domain</div>
                <div className="tabular-nums">{report.n_ood}</div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Retrieval quality by mode</CardTitle>
            </CardHeader>
            <CardContent>
              <EvalBarChart modes={report.modes} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Refusal accuracy</CardTitle>
            </CardHeader>
            <CardContent>
              <RefusalAccuracy modes={report.modes} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Faithfulness</CardTitle>
            </CardHeader>
            <CardContent className="text-sm">
              {report.faithfulness ? (
                <div className="flex flex-wrap gap-x-8 gap-y-2">
                  <div>
                    <div className="text-muted-foreground">Mode</div>
                    <div>{MODE_LABELS[report.faithfulness.mode] ?? report.faithfulness.mode}</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Faithfulness rate</div>
                    <div className="tabular-nums">{pct(report.faithfulness.faithfulness_rate)}</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Generation refusal accuracy</div>
                    <div className="tabular-nums">{pct(report.faithfulness.generation_refusal_accuracy)}</div>
                  </div>
                </div>
              ) : (
                <p className="text-muted-foreground">
                  {report.note ?? "faithfulness requires a key -- run with --api-key to populate"}
                </p>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>How these numbers are generated</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              <p>
                Run the benchmark against the DocVQA eval corpus and gold set, no API key required for
                retrieval metrics:
              </p>
              <pre className="mt-2 overflow-x-auto rounded-md bg-muted p-2 text-xs">{RUN_CMD}</pre>
              <p className="mt-2">
                Add <code>--api-key</code> to also populate the faithfulness path (real generation + NLI
                verification through the same code the <code>/answer</code> route uses).
              </p>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
