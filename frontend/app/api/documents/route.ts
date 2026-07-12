// Proxy for adding/removing files in an existing session. Keeps BACKEND_TOKEN
// server-side (never reaches the browser), mirroring the other /api proxies.

export async function POST(req: Request) {
  const formData = await req.formData();
  try {
    const r = await fetch(`${process.env.BACKEND_URL}/documents`, {
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

export async function DELETE(req: Request) {
  const qs = new URL(req.url).searchParams.toString();
  try {
    const r = await fetch(`${process.env.BACKEND_URL}/documents?${qs}`, {
      method: "DELETE",
      headers: { authorization: `Bearer ${process.env.BACKEND_TOKEN}` },
      signal: AbortSignal.timeout(30000),
    });
    return new Response(await r.text(), { status: r.status, headers: { "content-type": "application/json" } });
  } catch (err) {
    if (err instanceof Error && err.name === "TimeoutError") {
      return Response.json({ error: "backend timed out" }, { status: 504 });
    }
    return Response.json({ error: "backend unavailable" }, { status: 502 });
  }
}
