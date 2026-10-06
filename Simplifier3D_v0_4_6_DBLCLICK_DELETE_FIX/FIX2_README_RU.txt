Simplifier3D v0.1 CORRECT FIX2
================================

No RDP logic was changed.

This fix only makes the original v26 Target-points behavior easier to test
and understand in 3D.

Important:
- MODEL — 3D LINE is an exact straight line, so pure RDP can only keep 2 points.
- MODEL — 3 SEGMENTS is three exact straight segments, so pure RDP naturally
  keeps 4 vertices. Asking for 10 points cannot produce 10 without inventing
  redundant points, which v26 never did.
- MODEL — 3D CURVE is added specifically to test Target points.
  With Target points = 10, the unchanged v26 target search returns 10 points.

When Target points > 0, Max XYZ error becomes the epsilon found by the
automatic search. This is the same behavior as v26.

Use:
1. Open Geometry 3D.
2. Click MODEL — 3D CURVE (target test).
3. Set Target points = 10.
4. Click SIMPLIFY XYZ.
Expected result: 10 points.
