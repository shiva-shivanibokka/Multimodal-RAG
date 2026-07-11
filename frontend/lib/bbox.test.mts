import test from "node:test";
import assert from "node:assert/strict";
import { bboxToOverlayRect } from "./bbox.ts";

test("scales bbox proportionally when image is displayed smaller than natural size", () => {
  const rect = bboxToOverlayRect([100, 200, 300, 400], { width: 1000, height: 2000 }, { width: 500, height: 1000 });
  assert.deepEqual(rect, { left: 50, top: 100, width: 100, height: 100 });
});

test("is a no-op scale when rendered size equals natural size", () => {
  const rect = bboxToOverlayRect([10, 20, 30, 40], { width: 612, height: 792 }, { width: 612, height: 792 });
  assert.deepEqual(rect, { left: 10, top: 20, width: 20, height: 20 });
});

test("returns a zero rect instead of NaN/Infinity while the image hasn't loaded", () => {
  const rect = bboxToOverlayRect([10, 20, 30, 40], { width: 0, height: 0 }, { width: 0, height: 0 });
  assert.deepEqual(rect, { left: 0, top: 0, width: 0, height: 0 });
});
