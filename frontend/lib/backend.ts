export type Citation = { page: number; bbox: number[]; snippet: string };
export type Claim = { text: string; supported: boolean; score: number; citations: Citation[] };
export type AnswerResponse = { answer: string; refused: boolean; claims: Claim[]; citations: Citation[] };
export type Doc = { doc_id: number; filename: string; n_pages: number };
export type IngestResponse = { session_id: string; docs: Doc[]; n_pages: number; n_chunks: number };

export async function askBackend(body: {
  question: string; provider: string; model: string; api_key: string;
  session_id?: string; retrieval_mode?: string; verified?: boolean;
}): Promise<AnswerResponse> {
  const r = await fetch("/api/answer", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (r.status === 404) throw new Error("Your session expired — please re-upload the document, then ask again.");
  if (!r.ok) throw new Error(`backend ${r.status}`);
  return r.json();
}

/** Surface the backend's own error detail (e.g. "upload too large",
 * "too many pages") instead of a bare status code when present. */
async function backendErr(r: Response): Promise<string> {
  try {
    const j = await r.json();
    if (j?.detail) return String(j.detail);
  } catch {
    /* non-JSON body */
  }
  return `backend ${r.status}`;
}

/** Ingest one or more files into a NEW combined session. */
export async function ingestDocs(files: File[]): Promise<IngestResponse> {
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  const r = await fetch("/api/ingest", { method: "POST", body: fd });
  if (!r.ok) throw new Error(await backendErr(r));
  return r.json();
}

/** Append files to an existing session (OCRs only the new files). */
export async function addDocs(sessionId: string, files: File[]): Promise<IngestResponse> {
  const fd = new FormData();
  fd.append("session_id", sessionId);
  for (const f of files) fd.append("files", f);
  const r = await fetch("/api/documents", { method: "POST", body: fd });
  if (r.status === 404) throw new Error("Your session expired — please re-upload the documents.");
  if (!r.ok) throw new Error(await backendErr(r));
  return r.json();
}

/** Remove one file from a session by its doc_id (rebuilds index, no re-OCR). */
export async function removeDoc(sessionId: string, docId: number): Promise<IngestResponse> {
  const r = await fetch(
    `/api/documents?session_id=${encodeURIComponent(sessionId)}&doc_id=${docId}`,
    { method: "DELETE" }
  );
  if (r.status === 404) throw new Error("Your session expired — please re-upload the documents.");
  if (!r.ok) throw new Error(await backendErr(r));
  return r.json();
}

/** URL for a session's rendered page image (Task 4.4), served through the
 * server-side proxy so BACKEND_TOKEN never reaches the browser. */
export function pageUrl(sessionId: string, page: number): string {
  return `/api/page/${sessionId}/${page}`;
}

/** Task 5.4: mirrors the report dict backend/eval/run_eval.py writes to
 * backend/eval/report.json -- keep in sync with that schema. */
export type EvalModeMetrics = {
  recall_at_1: number;
  recall_at_5: number;
  mrr: number;
  citation_accuracy: number;
  refusal_accuracy: number;
};
export type EvalFaithfulness = {
  mode: string;
  n_items: number;
  faithfulness_rate: number;
  generation_refusal_accuracy: number;
} | null;
export type EvalReport = {
  dataset: string;
  n_docs: number;
  n_answerable: number;
  n_ood: number;
  generated_at?: string | null;
  modes: Record<string, EvalModeMetrics>;
  faithfulness?: EvalFaithfulness;
  note?: string;
  /** present only on the bundled illustrative fixture, never on a real report.json */
  sample?: boolean;
};

/** Fetches the benchmark report through the server-side proxy. Returns null
 * on a 404 (no report run yet) so callers can fall back to sample data. */
export async function fetchEvalReport(): Promise<EvalReport | null> {
  const r = await fetch("/api/eval-report");
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`backend ${r.status}`);
  return r.json();
}
