"use client";

import { useState } from "react";
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
import { askBackend, type AnswerResponse } from "@/lib/backend";

const PROVIDERS = ["openai", "groq", "gemini", "anthropic"] as const;

export default function Home() {
  const [provider, setProvider] = useState<string>("groq");
  const [model, setModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<AnswerResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function handleApiKeyChange(value: string) {
    setApiKey(value);
    // ponytail: BYOK key kept only in tab-lifetime sessionStorage, never localStorage/disk.
    sessionStorage.setItem("byok_api_key", value);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await askBackend({ question, provider, model, api_key: apiKey });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-2xl flex-col gap-6 p-8">
      <h1 className="text-2xl font-semibold">Multimodal RAG — Trust Layer</h1>

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
            <Button type="submit" disabled={loading || !question || !apiKey || !model}>
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

      {result && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Answer
              {result.refused && <Badge variant="destructive">Not in your documents</Badge>}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="whitespace-pre-wrap text-sm">
              {result.refused ? "The documents do not support an answer to this question." : result.answer}
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
