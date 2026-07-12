"use client";

import { useState } from "react";
import Link from "next/link";
import { askBackend, ingestDoc, type AnswerResponse, type Citation, type Claim } from "@/lib/backend";
import { CitationViewer } from "@/components/CitationViewer";
import { Dropdown, type Opt } from "@/components/Dropdown";

const PROVIDERS: Opt[] = [
  { value: "groq", label: "Groq", note: "free tier" },
  { value: "gemini", label: "Google Gemini", note: "free tier" },
  { value: "openai", label: "OpenAI" },
  { value: "anthropic", label: "Anthropic" },
];

// Curated per-provider models. Vision-capable models are needed for image /
// cross-modal questions. Availability depends on your key + the provider.
const MODELS: Record<string, Opt[]> = {
  groq: [
    { value: "llama-3.3-70b-versatile", label: "Llama 3.3 70B", note: "text · versatile" },
    { value: "llama-3.1-8b-instant", label: "Llama 3.1 8B", note: "text · fast" },
    { value: "meta-llama/llama-4-scout-17b-16e-instruct", label: "Llama 4 Scout", note: "vision" },
    { value: "meta-llama/llama-4-maverick-17b-128e-instruct", label: "Llama 4 Maverick", note: "vision" },
  ],
  gemini: [
    { value: "gemini-2.0-flash", label: "Gemini 2.0 Flash", note: "vision · fast" },
    { value: "gemini-2.5-flash", label: "Gemini 2.5 Flash", note: "vision" },
    { value: "gemini-2.5-pro", label: "Gemini 2.5 Pro", note: "vision · strong" },
  ],
  openai: [
    { value: "gpt-4o", label: "GPT-4o", note: "vision" },
    { value: "gpt-4o-mini", label: "GPT-4o mini", note: "vision · cheap" },
    { value: "gpt-4.1", label: "GPT-4.1", note: "vision" },
    { value: "gpt-4.1-mini", label: "GPT-4.1 mini", note: "vision · cheap" },
  ],
  anthropic: [
    // Current-generation models first (so a current model is the default);
    // -latest aliases kept below as stable fallbacks. Exact IDs available to
    // a given key/plan can vary — if one 404s, just pick another.
    { value: "claude-sonnet-4-5", label: "Claude Sonnet 4.5", note: "vision" },
    { value: "claude-opus-4-1", label: "Claude Opus 4.1", note: "vision · strong" },
    { value: "claude-3-5-sonnet-latest", label: "Claude 3.5 Sonnet", note: "vision" },
    { value: "claude-3-5-haiku-latest", label: "Claude 3.5 Haiku", note: "text · fast" },
    { value: "claude-3-opus-latest", label: "Claude 3 Opus", note: "vision" },
  ],
};

const MODE_TIPS: Record<string, string> = {
  hybrid: "Dense semantic search + keyword (BM25), fused and reranked. Best default for most questions.",
  dense: "Pure semantic vector search over the text. Good for paraphrased or conceptual questions.",
  cross_modal:
    "Retrieves whole page images with CLIP — finds visually-relevant pages even when they have little text.",
  caption_baseline:
    "Retrieves pages via their OCR'd text embedded like a caption. The baseline that true cross-modal is measured against.",
};
const RETRIEVAL_MODES = Object.keys(MODE_TIPS);

function Claims({ claims, onSelect }: { claims: Claim[]; onSelect: (c: Citation) => void }) {
  return (
    <ul className="claims">
      {claims.map((claim, i) => (
        <li key={i} className={`claim ${claim.supported ? "sup" : "uns"}`}>
          <span className="mark" aria-hidden="true">{claim.supported ? "✓" : "!"}</span>
          <div>
            <div className="txt">{claim.text}</div>
            <div className="meta">
              <span className="verdict">
                {claim.supported ? "grounded" : "unsupported"} · {claim.score.toFixed(2)}
              </span>
              {claim.citations.length > 0 && (
                <button type="button" className="src" onClick={() => onSelect(claim.citations[0])}>
                  show source · p{claim.citations[0].page + 1}
                </button>
              )}
            </div>
          </div>
        </li>
      ))}
    </ul>
  );
}

export default function Home() {
  const [provider, setProvider] = useState<string>("groq");
  const [model, setModel] = useState(MODELS.groq[0].value);
  // ponytail: lazy initializer only runs client-side (typeof window guard), so
  // this is SSR-safe and matches the UI's "stays in this tab" sessionStorage claim.
  const [apiKey, setApiKey] = useState(() =>
    typeof window !== "undefined" ? (sessionStorage.getItem("byok_api_key") ?? "") : ""
  );
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<AnswerResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [verified, setVerified] = useState(true);
  const [retrievalMode, setRetrievalMode] = useState<string>("hybrid");

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [nChunks, setNChunks] = useState<number | null>(null);
  const [ingesting, setIngesting] = useState(false);
  const [ingestError, setIngestError] = useState<string | null>(null);

  const [fileName, setFileName] = useState<string | null>(null);
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);

  function handleProviderChange(p: string) {
    setProvider(p);
    setModel(MODELS[p][0].value); // keep the model valid for the chosen provider
  }

  function handleApiKeyChange(value: string) {
    setApiKey(value);
    // ponytail: BYOK key kept only in tab-lifetime sessionStorage, never localStorage/disk.
    sessionStorage.setItem("byok_api_key", value);
  }

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setIngesting(true);
    setIngestError(null);
    setSelectedCitation(null);
    setResult(null);
    setFileName(file.name);
    try {
      const res = await ingestDoc(file);
      setSessionId(res.session_id);
      setNChunks(res.n_chunks);
    } catch (err) {
      setIngestError(err instanceof Error ? err.message : "Upload failed. Try a smaller PDF or image.");
      setSessionId(null);
    } finally {
      setIngesting(false);
      e.target.value = "";
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    // Validate on submit (button stays clickable) so the reason is always clear.
    if (ingesting) return setError("Still indexing your document — hang on a second, then ask.");
    if (ingestError) return setError(`That upload didn't finish: ${ingestError}`);
    if (!sessionId) return setError("Upload a document first, then ask.");
    if (!apiKey) return setError("Paste your provider API key to run the model.");
    if (!question.trim()) return setError("Type a question.");

    setLoading(true);
    setError(null);
    setResult(null);
    setSelectedCitation(null);
    try {
      const res = await askBackend({
        question,
        provider,
        model,
        api_key: apiKey,
        session_id: sessionId,
        retrieval_mode: retrievalMode,
        verified,
      });
      setResult(res);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Request failed. Check your API key and model.";
      if (msg.includes("session expired")) setSessionId(null); // reset UI so they re-upload
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="shell">
      <div className="topbar">
        <span className="eyebrow">Multimodal RAG · Trust Layer</span>
        <span className="topbar-right">
          <span className="live">
            <span className="dot" /> live
          </span>
          <Link className="navlink" href="/eval">
            benchmark →
          </Link>
        </span>
      </div>

      <header className="hero">
        <h1>
          <span className="line">Don&apos;t trust the model.</span>{" "}
          <span className="grad line">Verify it.</span>
        </h1>
        <p className="lede">
          Ask questions across scanned PDFs, images, and tables. Every claim is checked against the source —{" "}
          <b>grounded ones turn green, unsupported ones get flagged</b> — and when the answer isn&apos;t in your
          documents, it says so instead of guessing.
        </p>
      </header>

      {/* ---- workspace ---- */}
      <section className="panel">
        <div className="ph">
          <span className="n">01</span>
          <h2>Workspace</h2>
          <span className="chip">BYOK · key never stored</span>
        </div>

        <div className="rowbar">
          <div className="field grow" data-tip="Upload a scanned PDF or image. It's OCR'd, tables are extracted, and everything is indexed so you can ask about it.">
            <label>Document <i className="ti">i</i></label>
            <input type="file" accept=".pdf,.png,.jpg,.jpeg" onChange={handleFileChange} disabled={ingesting} />
          </div>

          <div className="field" data-tip="Which LLM provider to use with your own API key. Groq and Gemini have free tiers. Your key is never stored.">
            <label>Provider <i className="ti">i</i></label>
            <Dropdown value={provider} options={PROVIDERS} onChange={handleProviderChange} ariaLabel="Provider" />
          </div>

          <div className="field" data-tip="The specific model. Vision-capable models (marked 'vision') are needed for image and cross-modal questions.">
            <label>Model <i className="ti">i</i></label>
            <Dropdown value={model} options={MODELS[provider]} onChange={setModel} ariaLabel="Model" />
          </div>

          <div className="field" data-tip="Your provider API key. It stays in this browser tab only — never sent to our servers, never stored.">
            <label>API key <i className="ti">i</i></label>
            <input
              type="password"
              placeholder="paste — stays in this tab"
              value={apiKey}
              onChange={(e) => handleApiKeyChange(e.target.value)}
              autoComplete="off"
            />
          </div>
        </div>

        <div className="rowbar" style={{ marginTop: "1.05rem" }}>
          <div className="field grow">
            <label>Retrieval <i className="ti">i</i></label>
            <div className="seg" role="group" aria-label="Retrieval mode">
              {RETRIEVAL_MODES.map((m) => (
                <button
                  key={m}
                  type="button"
                  data-tip={MODE_TIPS[m]}
                  aria-pressed={retrievalMode === m}
                  onClick={() => setRetrievalMode(m)}
                >
                  {m}
                </button>
              ))}
            </div>
          </div>

          <div
            className="field"
            data-tip="The NLI faithfulness firewall. Each claim is checked against the retrieved source; if nothing is grounded, the answer is refused. Slower, but trustworthy."
          >
            <label>Verify <i className="ti">i</i></label>
            <label className="toggle">
              <input type="checkbox" checked={verified} onChange={(e) => setVerified(e.target.checked)} />
              <span className="track" />
              <span className="lbl">
                {verified ? "on" : "off"}
                <small>NLI firewall</small>
              </span>
            </label>
          </div>
        </div>

        <form className="ask-row" onSubmit={handleSubmit} style={{ marginTop: "1.3rem" }}>
          <div className="field grow" data-tip="Ask in plain language, e.g. 'What was the total on the invoice?'">
            <label>Question <i className="ti">i</i></label>
            <textarea
              placeholder="What was the total on the invoice?"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              rows={1}
            />
          </div>
          <button className="btn" type="submit" disabled={loading}>
            {loading ? "Verifying…" : "Ask"}
          </button>
        </form>

        <div style={{ marginTop: ".9rem" }}>
          {ingesting && <p className="status">Reading and indexing {fileName}…</p>}
          {ingestError && <p className="status err">{ingestError}</p>}
          {sessionId && !ingesting && (
            <p className="status ok">
              <b>{nChunks}</b> passages indexed from {fileName} · session {sessionId.slice(0, 8)}
            </p>
          )}
          {!sessionId && !ingesting && !ingestError && (
            <p className="status">Upload a document to begin. Then pick a model, paste a free Groq or Gemini key, and ask.</p>
          )}
          {error && <p className="status err" style={{ marginTop: ".4rem" }}>{error}</p>}
        </div>
      </section>

      {/* ---- evidence bench ---- */}
      {result && (
        <section className="panel">
          <div className="ph">
            <span className="n">02</span>
            <h2>Evidence bench</h2>
            {verified && !result.refused && <span className="chip">claims NLI-verified</span>}
          </div>

          <div className={`bench ${selectedCitation ? "" : "single"}`}>
            <div>
              {result.refused ? (
                <>
                  <div className="refused-head">
                    <span className="stamp">not in the record</span>
                    <h2>{result.answer ? "Drafted, but ungrounded" : "No supporting evidence"}</h2>
                  </div>
                  <p className="empty">
                    {result.answer
                      ? "The model produced an answer, but none of its claims are supported by your documents — so it's withheld as untrusted."
                      : "The retrieved passages were too weak to answer this. Try rephrasing, or upload a document that covers it."}
                  </p>
                  {result.answer && <p className="draft">{result.answer}</p>}
                  {result.claims.length > 0 && <Claims claims={result.claims} onSelect={setSelectedCitation} />}
                </>
              ) : (
                <>
                  <div className="answer-body">{result.answer}</div>
                  {result.claims.length > 0 && <Claims claims={result.claims} onSelect={setSelectedCitation} />}
                  {result.claims.length === 0 && result.citations.length > 0 && (
                    <button className="src" style={{ marginTop: "1rem" }} onClick={() => setSelectedCitation(result.citations[0])}>
                      show source · p{result.citations[0].page + 1}
                    </button>
                  )}
                </>
              )}
            </div>

            {selectedCitation && sessionId && (
              <div>
                <div className="ph" style={{ marginBottom: ".7rem" }}>
                  <span className="n">source</span>
                  <h2>Page {selectedCitation.page + 1}</h2>
                </div>
                <CitationViewer sessionId={sessionId} citation={selectedCitation} />
                {selectedCitation.snippet && <p className="source-cap">{selectedCitation.snippet}</p>}
              </div>
            )}
          </div>
        </section>
      )}

      <footer className="footer">
        Multimodal RAG Trust Layer · Next.js on Vercel + FastAPI/CPU-ML on Google Cloud Run ·{" "}
        <Link href="/eval">see the benchmark</Link>
      </footer>
    </main>
  );
}
