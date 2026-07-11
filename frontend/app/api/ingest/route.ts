export async function POST(req: Request) {
  const formData = await req.formData();
  const r = await fetch(`${process.env.BACKEND_URL}/ingest`, {
    method: "POST",
    headers: { authorization: `Bearer ${process.env.BACKEND_TOKEN}` },
    body: formData,
  });
  return new Response(await r.text(), { status: r.status, headers: { "content-type": "application/json" } });
}
