Simplifier3D v0.3 — FIX ARC / UNDO-REDO / AXIS GENERATION
================================================================

This version fixes the three issues reported after v0.2.

1. ARC — 3 POINTS
-----------------
The previous 3D-specific mouse detector was unreliable.

Now every orange edited point is a Matplotlib picker target.
ARC selection uses the picker event first, with the previous screen-distance
search kept as a fallback.

The v26 ARC logic itself is unchanged:
    - exactly 3 points
    - click order irrelevant
    - trajectory order gives START / VIA / END
    - original v26 2D circle routine is used in a local plane
    - selection locks after point 3
    - protected markers are preserved
    - START / VIA / END times are preserved

2. UNDO / REDO
--------------
v0.2 incorrectly cleared history after simplification.

Now the behavior follows v26:
    load:
        edited XYZ = recorded XYZ

    SIMPLIFY:
        current edited XYZ is pushed to Undo
        simplified result becomes edited XYZ

    ARC:
        current edited XYZ is pushed to Undo

    RESET:
        current edited XYZ is pushed to Undo
        recorded trajectory is restored

Undo / Redo work across simplification, arc and reset.

3. GENERATE AXES
----------------
Generation is restored on the Geometry 3D page.

For the CURRENT MODEL STAGE ONLY:
    X -> Axis1
    Y -> Axis2
    Z -> Axis3

No real 3D robot inverse kinematics is assumed yet.

Controls copied conceptually from v26:
    Output step, ms
    Timing:
        Original timing
        Constant path speed
    Path speed

Buttons:
    GENERATE AXES 1 / 2 / 3
    EXPORT AXIS1 / AXIS2 / AXIS3

Generated files use the MotionController trajectory format:
    axis1.txt
    axis2.txt
    axis3.txt

Build
-----
build_exe.bat is still the lightweight build:
it uses the Python/PySide6 already installed on the PC and creates no venv.
