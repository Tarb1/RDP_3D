Simplifier3D v0.4.4 — EXACT V26 MOUSE ARCHITECTURE
========================================================

This version corrects the previous misunderstanding.

The actual working Axis Simplifier v26 DOES NOT use mpl_connect for
point edit / double-click add / pan.

Its exact architecture is:

    EditableFigureCanvas(FigureCanvas)
        mousePressEvent   -> editor direct handler
        mouseMoveEvent    -> editor direct handler
        mouseReleaseEvent -> editor direct handler
        mouseDoubleClickEvent -> editor direct handler

Only WHEEL ZOOM stays a matplotlib scroll_event.

v0.4.4 ports that architecture literally to XY / XZ / YZ.

Exact v26 procedures copied/adapted:
------------------------------------
    canvas.mouseEventCoords(event)
    _find point in SCREEN PIXELS
    _find segment in SCREEN PIXELS
    frozen-pixel right-button pan
    wheel zoom around cursor
    one drag snapshot = one Undo
    double-click existing point = select only
    double-click segment = add one point
    artist refresh during drag, no full plot reconstruction

Only necessary 3D adaptation:
-----------------------------
    XY drag -> X,Y
    XZ drag -> X,Z
    YZ drag -> Y,Z

The third coordinate is untouched.

3D perspective:
---------------
Native Matplotlib rotate remains.
ARC selection is intercepted only while ARC mode is active.
Wheel also scales the 3D limits.

Automatic tab switching:
------------------------
Removed after ARC.
Removed after GENERATE AXES.

Recommended validation:
-----------------------
MODEL — CURVE
SIMPLIFY XYZ

In XY:
    wheel -> zoom
    right drag -> pan
    left drag orange point -> move
    double-click orange segment -> add
    select point + DELETE POINT -> delete

Then repeat drag in XZ and YZ.

This is the first 3D version that uses the ACTUAL v26 event ownership model,
not an approximation of it.
