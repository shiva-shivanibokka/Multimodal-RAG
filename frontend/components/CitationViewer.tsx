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
    <div className="max-h-[70vh] w-full overflow-auto rounded-md border">
      <div className="relative inline-block">
        <img
          ref={imgRef}
          src={pageUrl(sessionId, citation.page)}
          alt={`Source page ${citation.page + 1}`}
          onLoad={handleLoad}
          className="block max-w-full"
        />
        {!natural && (
          <div className="absolute inset-0 flex items-center justify-center bg-background/70 text-sm text-muted-foreground">
            Loading page image...
          </div>
        )}
        {rect && (
          <div
            className="pointer-events-none absolute border-2 border-destructive bg-destructive/25"
            style={{ left: rect.left, top: rect.top, width: rect.width, height: rect.height }}
          />
        )}
      </div>
    </div>
  );
}
