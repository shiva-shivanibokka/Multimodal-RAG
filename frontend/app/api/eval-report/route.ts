export async function GET() {
  const r = await fetch(`${process.env.BACKEND_URL}/eval/report`, {
    headers: { authorization: `Bearer ${process.env.BACKEND_TOKEN}` },
  });
  return new Response(await r.text(), { status: r.status, headers: { "content-type": "application/json" } });
}
