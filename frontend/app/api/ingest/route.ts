export async function POST(req: Request) {
  const formData = await req.formData();
  try {
    const r = await fetch(`${process.env.BACKEND_URL}/ingest`, {
      method: "POST",
      headers: { authorization: `Bearer ${process.env.BACKEND_TOKEN}` },
      body: formData,
      signal: AbortSignal.timeout(120000),
    });
    return new Response(await r.text(), { status: r.status, headers: { "content-type": "application/json" } });
  } catch (err) {
    if (err instanceof Error && err.name === "TimeoutError") {
      return Response.json({ error: "backend timed out" }, { status: 504 });
    }
    return Response.json({ error: "backend unavailable" }, { status: 502 });
  }
}
