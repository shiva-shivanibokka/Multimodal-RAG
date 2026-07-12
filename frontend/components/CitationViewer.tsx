"use client";

import { useEffect, useRef, useState } from "react";
import { pageUrl, type Citation } from "@/lib/backend";
import { bboxToOverlayRect, type Size } from "@/lib/bbox";

/**
 * Renders the cited page image with a highlighted overlay box over the
 * citation's bbox. The bbox is in page-image pixel space (see
 * backend/app/ingest/loader.py); the <img> is displayed scaled to fit its
 * container, so the overlay rect is recomputed from the image's natural vs.
 * rendered size (bboxToOverlayRect, lib/bbox.ts) on load and on resize.
 */
export function CitationViewer({ sessionId, citation }: { sessionId: string; citation: Citation }) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [natural, setNatural] = useState<Size | null>(null);
  const [rendered, setRendered] = useState<Size | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Reset when the citation points at a different page, so the overlay never
  // renders against the previous page's dimensions while the new image loads.
  useEffect(() => {
    setNatural(null);
    setRendered(null);
    setError(null);
  }, [citation.page]);

  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    const observer = new ResizeObserver(() => {
      setRendered({ width: img.clientWidth, height: img.clientHeight });
    });
    observer.observe(img);
    return () => observer.disconnect();
  }, []);

  function handleLoad() {
    const img = imgRef.current;
    if (!img) return;
    setNatural({ width: img.naturalWidth, height: img.naturalHeight });
    setRendered({ width: img.clientWidth, height: img.clientHeight });
  }

  const rect = natural && rendered ? bboxToOverlayRect(citation.bbox, natural, rendered) : null;

  return (
    <div className="source-frame" style={{ maxHeight: "70vh", overflow: "auto" }}>
      <div style={{ position: "relative", display: "inline-block" }}>
        <img
          ref={imgRef}
          src={pageUrl(sessionId, citation.page)}
          alt={`Source page ${citation.page + 1}`}
          onLoad={handleLoad}
          onError={() => setError("Couldn't load the source page.")}
          style={{ display: "block", maxWidth: "100%" }}
        />
        {error ? (
          <div className="viewer-loading">{error}</div>
        ) : (
          !natural && <div className="viewer-loading">Loading page image…</div>
        )}
        {rect && (
          <div
            className="cite-box"
            style={{ left: rect.left, top: rect.top, width: rect.width, height: rect.height }}
          />
        )}
      </div>
    </div>
  );
}
