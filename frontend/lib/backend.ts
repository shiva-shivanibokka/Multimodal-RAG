export type Citation = { page: number; bbox: number[]; snippet: string };
export type Claim = { text: string; supported: boolean; score: number; citations: Citation[] };
export type AnswerResponse = { answer: string; refused: boolean; claims: Claim[]; citations: Citation[] };

export async function askBackend(body: {
  question: string; provider: string; model: string; api_key: string;
  session_id?: string; retrieval_mode?: string; verified?: boolean;
}): Promise<AnswerResponse> {
  const r = await fetch("/api/answer", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`backend ${r.status}`);
  return r.json();
}
