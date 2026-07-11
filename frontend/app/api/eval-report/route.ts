export async function GET() {
  try {
    const r = await fetch(`${process.env.BACKEND_URL}/eval/report`, {
      headers: { authorization: `Bearer ${process.env.BACKEND_TOKEN}` },
    });
    return new Response(await r.text(), { status: r.status, headers: { "content-type": "application/json" } });
  } catch {
    return Response.json({ error: "backend unavailable" }, { status: 502 });
  }
}
