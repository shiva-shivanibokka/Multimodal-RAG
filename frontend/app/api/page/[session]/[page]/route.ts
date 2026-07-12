export async function GET(
  _req: Request,
  { params }: { params: Promise<{ session: string; page: string }> }
) {
  const { session, page } = await params;
  try {
    const r = await fetch(`${process.env.BACKEND_URL}/page/${session}/${page}`, {
      headers: { authorization: `Bearer ${process.env.BACKEND_TOKEN}` },
      signal: AbortSignal.timeout(10000),
    });
    if (!r.ok) {
      r.body?.cancel();
      return new Response(null, { status: r.status });
    }
    return new Response(r.body, { status: 200, headers: { "content-type": "image/png" } });
  } catch (err) {
    if (err instanceof Error && err.name === "TimeoutError") {
      return new Response("backend timed out", { status: 504 });
    }
    return new Response("backend unavailable", { status: 502 });
  }
}
