Simplifier3D v0.4.6 — DOUBLE-CLICK / DELETE FIX
==================================================

Confirmed working and unchanged:
    zoom
    pan
    3D rotation
    vertex drag

Double-click diagnosis
----------------------
The Add algorithm already matched v26 closely, but on the test PC the Qt
mouseDoubleClickEvent path apparently did not reach the handler reliably.

v0.4.6 keeps the native Qt double-click handler AND adds a robust fallback:
ordinary left mouse press events are timed and compared in screen pixels.

A double click is recognized when:
    interval <= 0.45 s
    pointer displacement <= 8 px

This uses the same mousePressEvent path that is already proven by working drag.

After recognition the existing v26-style Add procedure is used:
    hit-test orange segment in screen pixels
    insert one point strictly between neighbor times
    interpolate X/Y/Z on the current 3D segment
    select the new point
    one Undo action

Delete
------
DELETE POINT no longer depends on asking the active canvas to perform deletion.

The page directly uses the selected point index and:
    protects first/last
    protects marker points
    saves Undo state
    deletes point
    synchronizes all views

Automatic tab switching remains disabled.
