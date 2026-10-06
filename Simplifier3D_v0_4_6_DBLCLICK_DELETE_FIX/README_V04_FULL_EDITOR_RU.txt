Simplifier3D v0.4 — FULL GENERIC 3D EDITOR
================================================

Base:
    accepted Simplifier3D v0.3.1
    working Simplifier v26 behavior

New generic v26 functionality on Geometry 3D:
------------------------------------------------
1. Edit vertices in projections:
       XY drag -> changes X,Y; Z stays unchanged
       XZ drag -> changes X,Z; Y stays unchanged
       YZ drag -> changes Y,Z; X stays unchanged

   Direct drag editing in the 3D perspective view is intentionally NOT used,
   because a 2D mouse displacement does not uniquely define a 3D displacement.

2. Add point:
       double-click a segment in XY / XZ / YZ
       -> inserts one new XYZ point
       -> time and all three coordinates are interpolated from the same segment

3. Delete point:
       select orange vertex in XY / XZ / YZ
       -> DELETE POINT

   Protection is the same as v26:
       first point protected
       last point protected
       marker points protected

4. MARKER SKELETON:
       keeps first + last + all marker points
       removes all other edited points

5. Undo / Redo:
       covers simplification
       manual drag
       added points
       deleted points
       marker skeleton
       3-point arc
       reset

6. Mouse navigation in XY / XZ / YZ:
       left drag point = edit
       double-click segment = add point
       right drag = pan
       wheel = zoom

7. 3D view:
       rotate / inspect
       equal coordinate scale
       ARC point selection works

Already retained:
       3D RDP
       target-points mode
       3-point arc in arbitrary 3D plane
       Arc plane UV diagnostic
       save XYZ
       model Axis1=X, Axis2=Y, Axis3=Z generation
       export axis1.txt / axis2.txt / axis3.txt
       Original timing / Constant path speed

Marker test:
       model_data/marker3d.csv
Use OPEN XYZ to load it and test:
       simplify -> marker preservation
       delete marker -> must be blocked
       MARKER SKELETON -> first + M1 + X1 + last remain

Not ported into Geometry 3D:
----------------------------
The AB–BC-specific MACHINE CONFIG / calibration / inverse kinematics.
Those functions remain unchanged in the Geometria AB–BC tab because a real
3-axis mechanism and its inverse kinematics have not yet been defined.

Build:
       build_exe.bat
The build remains lightweight and uses the installed Python environment.
