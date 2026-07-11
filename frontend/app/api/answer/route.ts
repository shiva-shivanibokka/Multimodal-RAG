export async function POST(req: Request) {
  const body = await req.text();
  try {
    const r = await fetch(`${process.env.BACKEND_URL}/answer`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${process.env.BACKEND_TOKEN}` },
      body,
    });
    return new Response(await r.text(), { status: r.status, headers: { "content-type": "application/json" } });
  } catch {
    return Response.json({ error: "backend unavailable" }, { status: 502 });
  }
}
