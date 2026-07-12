export async function GET(
  _req: Request,
  { params }: { params: Promise<{ session: string; page: string }> }
) {
  const { session, page } = await params;
  try {
    const r = await fetch(`${process.env.BACKEND_URL}/page/${session}/${page}`, {
      headers: { authorization: `Bearer ${process.env.BACKEND_TOKEN}` },
    });
    if (!r.ok) {
      r.body?.cancel();
      return new Response(null, { status: r.status });
    }
    return new Response(r.body, { status: 200, headers: { "content-type": "image/png" } });
  } catch {
    return new Response("backend unavailable", { status: 502 });
  }
}
