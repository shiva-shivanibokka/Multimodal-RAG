export async function GET() {
  try {
    const r = await fetch(`${process.env.BACKEND_URL}/eval/report`, {
      headers: { authorization: `Bearer ${process.env.BACKEND_TOKEN}` },
      signal: AbortSignal.timeout(10000),
    });
    return new Response(await r.text(), { status: r.status, headers: { "content-type": "application/json" } });
  } catch (err) {
    if (err instanceof Error && err.name === "TimeoutError") {
      return Response.json({ error: "backend timed out" }, { status: 504 });
    }
    return Response.json({ error: "backend unavailable" }, { status: 502 });
  }
}
