Simplifier3D v0.3.1 — ARC DISPLAY DIAGNOSTICS
================================================

No arc geometry or RDP logic changed.

Display fixes:
1. 3D view now preserves equal coordinate-unit scale by setting the 3D box
   aspect from the actual X/Y/Z data ranges.
2. New tab: Arc plane UV
   - automatically opens after a successful 3-point arc
   - looks perpendicular to the arc plane
   - uses equal U/V scale
   - marks START, VIA, END and CENTER
   - shows the fitted radius

For the built-in tilted test:
    MODEL — ARC
    ARC — 3 POINTS
Expected diagnostic:
    R = 100
    sweep = 180 deg
    UV view = an ordinary semicircle

XY/XZ/YZ projections of a tilted circle are generally ellipses and therefore
are not expected to look circular.
