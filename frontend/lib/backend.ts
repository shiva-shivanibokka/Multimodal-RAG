export type Citation = { page: number; bbox: number[]; snippet: string };
export type Claim = { text: string; supported: boolean; score: number; citations: Citation[] };
export type AnswerResponse = { answer: string; refused: boolean; claims: Claim[]; citations: Citation[] };
export type IngestResponse = { session_id: string; n_pages: number; n_chunks: number };

export async function askBackend(body: {
  question: string; provider: string; model: string; api_key: string;
  session_id?: string; retrieval_mode?: string; verified?: boolean;
}): Promise<AnswerResponse> {
  const r = await fetch("/api/answer", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`backend ${r.status}`);
  return r.json();
}

export async function ingestDoc(file: File): Promise<IngestResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const r = await fetch("/api/ingest", { method: "POST", body: formData });
  if (!r.ok) throw new Error(`backend ${r.status}`);
  return r.json();
}

/** URL for a session's rendered page image (Task 4.4), served through the
 * server-side proxy so BACKEND_TOKEN never reaches the browser. */
export function pageUrl(sessionId: string, page: number): string {
  return `/api/page/${sessionId}/${page}`;
}
