"use client";

import { useState } from "react";
import Link from "next/link";
import { askBackend, ingestDoc, type AnswerResponse, type Citation, type Claim } from "@/lib/backend";
import { CitationViewer } from "@/components/CitationViewer";

const PROVIDERS = ["groq", "gemini", "openai", "anthropic"] as const;
const RETRIEVAL_MODES = ["hybrid", "dense", "cross_modal", "caption_baseline"] as const;

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
  const [model, setModel] = useState("");
  const [apiKey, setApiKey] = useState("");
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
        session_id: sessionId ?? undefined,
        retrieval_mode: retrievalMode,
        verified,
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed. Check your API key and model name.");
    } finally {
      setLoading(false);
    }
  }

  const canAsk = !loading && !!question && !!apiKey && !!model && !!sessionId;

  return (
    <main className="shell">
      <div className="topbar">
        <span className="eyebrow">Multimodal RAG · Trust Layer</span>
        <Link className="navlink" href="/eval">
          benchmark →
        </Link>
      </div>

      <header className="hero">
        <div>
          <h1>
            Don&apos;t trust the model.<br />
            <span className="grad">Verify it.</span>
          </h1>
          <p className="lede">
            Ask questions across scanned PDFs, images, and tables. Every claim is checked against the source —{" "}
            <b>grounded ones turn green, unsupported ones get flagged</b>, and when the answer isn&apos;t in your
            documents, it says so instead of guessing.
          </p>
        </div>
        <span className="live">
          <span className="dot" /> live · <Link href="/eval">benchmark</Link>
        </span>
      </header>

      {/* ---- workspace: one row of controls ---- */}
      <section className="panel">
        <div className="ph">
          <span className="n">01</span>
          <h2>Workspace</h2>
          <span className="chip">BYOK · key never stored</span>
        </div>

        <div className="rowbar">
          <div className="field grow">
            <label>Document</label>
            <input type="file" accept=".pdf,.png,.jpg,.jpeg" onChange={handleFileChange} disabled={ingesting} />
          </div>

          <div className="field">
            <label htmlFor="provider">Provider</label>
            <select id="provider" value={provider} onChange={(e) => setProvider(e.target.value)}>
              {PROVIDERS.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>

          <div className="field">
            <label htmlFor="model">Model</label>
            <input
              id="model"
              placeholder="llama-3.1-8b-instant"
              value={model}
              onChange={(e) => setModel(e.target.value)}
            />
          </div>

          <div className="field">
            <label htmlFor="apikey">API key</label>
            <input
              id="apikey"
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
            <label>Retrieval</label>
            <div className="seg" role="group" aria-label="Retrieval mode">
              {RETRIEVAL_MODES.map((m) => (
                <button
                  key={m}
                  type="button"
                  aria-pressed={retrievalMode === m}
                  onClick={() => setRetrievalMode(m)}
                >
                  {m}
                </button>
              ))}
            </div>
          </div>

          <div className="field">
            <label>Verify</label>
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
          <div className="field grow">
            <label htmlFor="q">Question</label>
            <textarea
              id="q"
              placeholder="What was the total on the invoice?"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              rows={1}
            />
          </div>
          <button className="btn" type="submit" disabled={!canAsk}>
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
            <p className="status">Upload a document to begin. Then paste a free Groq or Gemini key and ask.</p>
          )}
          {error && <p className="status err" style={{ marginTop: ".4rem" }}>{error}</p>}
        </div>
      </section>

      {/* ---- evidence bench: answer | source, row-wise ---- */}
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
