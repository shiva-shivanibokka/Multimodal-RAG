"use client";

import { useState } from "react";
import Link from "next/link";
import { CheckCircle2, Flag } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { askBackend, ingestDoc, type AnswerResponse, type Citation, type Claim } from "@/lib/backend";
import { CitationViewer } from "@/components/CitationViewer";

const PROVIDERS = ["openai", "groq", "gemini", "anthropic"] as const;
const RETRIEVAL_MODES = ["hybrid", "dense", "cross_modal", "caption_baseline"] as const;

function ClaimsList({ claims, onSelectCitation }: { claims: Claim[]; onSelectCitation: (c: Citation) => void }) {
  return (
    <ul className="flex flex-col gap-3">
      {claims.map((claim, i) => (
        <li key={i} className="flex items-start gap-2 text-sm">
          {claim.supported ? (
            <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-green-600" />
          ) : (
            <Flag className="mt-0.5 size-4 shrink-0 text-destructive" />
          )}
          <div className="flex flex-col gap-1">
            <span>{claim.text}</span>
            <span className="text-xs text-muted-foreground">
              {claim.supported ? "supported" : "unsupported"} · {claim.score.toFixed(2)}
            </span>
            {claim.citations.length > 0 && (
              <button
                type="button"
                onClick={() => onSelectCitation(claim.citations[0])}
                className="w-fit text-xs text-primary underline underline-offset-2"
              >
                view source (page {claim.citations[0].page + 1})
              </button>
            )}
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
    try {
      const res = await ingestDoc(file);
      setSessionId(res.session_id);
      setNChunks(res.n_chunks);
    } catch (err) {
      setIngestError(err instanceof Error ? err.message : "upload failed");
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
      setError(err instanceof Error ? err.message : "request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-2xl flex-col gap-6 p-8">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Multimodal RAG — Trust Layer</h1>
        <Link href="/eval" className="text-sm text-primary underline underline-offset-2">
          eval dashboard
        </Link>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Document</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          <input
            type="file"
            accept=".pdf,.png,.jpg,.jpeg"
            onChange={handleFileChange}
            disabled={ingesting}
            className="text-sm file:mr-3 file:rounded-md file:border-0 file:bg-secondary file:px-3 file:py-1.5 file:text-sm file:font-medium"
          />
          {ingesting && <p className="text-sm text-muted-foreground">Indexing document...</p>}
          {ingestError && <p className="text-sm text-destructive">{ingestError}</p>}
          {sessionId && !ingesting && (
            <p className="text-sm text-muted-foreground">
              {nChunks} chunks indexed · session {sessionId.slice(0, 8)}
            </p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Settings</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <label className="text-sm font-medium">Provider</label>
            <Select value={provider} onValueChange={setProvider}>
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select a provider" />
              </SelectTrigger>
              <SelectContent>
                {PROVIDERS.map((p) => (
                  <SelectItem key={p} value={p}>
                    {p}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-sm font-medium">Model</label>
            <Input
              placeholder="e.g. llama-3.1-8b-instant"
              value={model}
              onChange={(e) => setModel(e.target.value)}
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-sm font-medium">API key (BYOK)</label>
            <Input
              type="password"
              placeholder="never stored on disk"
              value={apiKey}
              onChange={(e) => handleApiKeyChange(e.target.value)}
              autoComplete="off"
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-sm font-medium">Retrieval mode</label>
            <Select value={retrievalMode} onValueChange={setRetrievalMode}>
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {RETRIEVAL_MODES.map((m) => (
                  <SelectItem key={m} value={m}>
                    {m}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center justify-between">
            <label className="text-sm font-medium">Verified mode</label>
            <Switch checked={verified} onCheckedChange={setVerified} />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Ask a question</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <Textarea
              placeholder="What do the documents say about..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              rows={4}
            />
            <Button type="submit" disabled={loading || !question || !apiKey || !model || !sessionId}>
              {loading ? "Asking..." : "Ask"}
            </Button>
          </form>
        </CardContent>
      </Card>

      {error && (
        <Card>
          <CardContent className="text-sm text-destructive">{error}</CardContent>
        </Card>
      )}

      {result &&
        (result.refused ? (
          <Card className="border-destructive">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-destructive">
                Not supported by your documents
                <Badge variant="destructive">refused</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-3">
              <p className="text-sm text-muted-foreground">
                {result.answer
                  ? "The model drafted an answer, but none of its claims were grounded in your documents:"
                  : "The retrieved evidence was too weak to answer this question."}
              </p>
              {result.answer && (
                <p className="whitespace-pre-wrap text-sm italic text-muted-foreground">{result.answer}</p>
              )}
              {result.claims.length > 0 && (
                <ClaimsList claims={result.claims} onSelectCitation={setSelectedCitation} />
              )}
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardHeader>
              <CardTitle>Answer</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <p className="whitespace-pre-wrap text-sm">{result.answer}</p>
              {result.claims.length > 0 && (
                <ClaimsList claims={result.claims} onSelectCitation={setSelectedCitation} />
              )}
            </CardContent>
          </Card>
        ))}

      {selectedCitation && sessionId && (
        <Card>
          <CardHeader>
            <CardTitle>Source — page {selectedCitation.page + 1}</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-2">
            <CitationViewer sessionId={sessionId} citation={selectedCitation} />
            <p className="text-xs text-muted-foreground">{selectedCitation.snippet}</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
