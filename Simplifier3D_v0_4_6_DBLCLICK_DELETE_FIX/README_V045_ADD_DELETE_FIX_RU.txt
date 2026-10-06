Simplifier3D v0.4.5

Confirmed working from v0.4.4:
- wheel zoom
- pan
- 3D rotation
- point drag

Those procedures were not changed.

This fix moves Add/Delete into the active projection canvas itself,
matching the working v26 TrajectoryPlot architecture.

Add:
- double-click orange segment in XY/XZ/YZ
- new point is inserted directly into the shared edited XYZ list
- time is a legal integer ms strictly between neighbors
- XYZ lies exactly on the current 3D segment
- operation is one Undo action

Delete:
- select orange point
- press DELETE POINT, Delete, or Backspace
- first/last and marker points are protected
- operation is one Undo action

No automatic tab switching after ARC or Generate Axes.
