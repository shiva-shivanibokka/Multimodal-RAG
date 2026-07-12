export async function POST(req: Request) {
  const body = await req.text();
  try {
    const r = await fetch(`${process.env.BACKEND_URL}/answer`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${process.env.BACKEND_TOKEN}` },
      body,
      signal: AbortSignal.timeout(60000),
    });
    return new Response(await r.text(), { status: r.status, headers: { "content-type": "application/json" } });
  } catch (err) {
    if (err instanceof Error && err.name === "TimeoutError") {
      return Response.json({ error: "backend timed out" }, { status: 504 });
    }
    return Response.json({ error: "backend unavailable" }, { status: 502 });
  }
}
