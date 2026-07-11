export async function GET(
  _req: Request,
  { params }: { params: Promise<{ session: string; page: string }> }
) {
  const { session, page } = await params;
  const r = await fetch(`${process.env.BACKEND_URL}/page/${session}/${page}`, {
    headers: { authorization: `Bearer ${process.env.BACKEND_TOKEN}` },
  });
  if (!r.ok) return new Response(null, { status: r.status });
  return new Response(r.body, { status: 200, headers: { "content-type": "image/png" } });
}
