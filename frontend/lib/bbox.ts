export type Size = { width: number; height: number };
export type Rect = { left: number; top: number; width: number; height: number };

/**
 * Scales a citation bbox (`[x0, y0, x1, y1]`, in page-image pixel space — see
 * backend/app/ingest/loader.py) into CSS pixels for an overlay positioned
 * over a rendered `<img>`, given the image's natural (source) size and its
 * current on-screen (rendered) size.
 */
export function bboxToOverlayRect(bbox: number[], natural: Size, rendered: Size): Rect {
  const [x0, y0, x1, y1] = bbox;
  const scaleX = natural.width > 0 ? rendered.width / natural.width : 0;
  const scaleY = natural.height > 0 ? rendered.height / natural.height : 0;
  return {
    left: x0 * scaleX,
    top: y0 * scaleY,
    width: (x1 - x0) * scaleX,
    height: (y1 - y0) * scaleY,
  };
}
