Simplifier3D v0.2 — 3D ARC stage
====================================

Base:
    working Simplifier v26
    + accepted 3D RDP stage

What is new:
    ARC — 3 POINTS in Geometry 3D.

Important:
    The circle algorithm is NOT rewritten.

Implementation:
    1. Operator selects three XYZ vertices.
    2. Their plane is calculated.
    3. A local orthonormal (u,v) basis is created in that plane.
    4. The exact working v26 function:
           circle_arc_through_three_points()
       is called unchanged in local 2D coordinates.
    5. Generated arc points are transformed back to XYZ.

v26 rules preserved:
    - exactly three distinct selected vertices
    - click order irrelevant; trajectory order gives START/VIA/END
    - point selection locks immediately after point 3
    - protected markers between START and END are not removed
    - START/VIA/END time, marker and comment are preserved
    - arc tolerance has the same meaning as v26
    - Undo / Redo available for the 3D arc operation
    - selection cleared after success/failure/reset/undo/redo

Test:
    Geometry 3D
    -> MODEL — TILTED 3-POINT ARC
    -> ARC — 3 POINTS
    -> click all three orange vertices in any order

You may select the vertices in 3D, XY, XZ or YZ view.
For the 3D model, left rotation is temporarily disabled only while ARC selection
mode is active, so single clicks select points reliably.

Build:
    build_exe.bat

This build script no longer creates .venv_build and does not reinstall PySide6.
It uses the already installed Python environment.
