"""Run with QT_QPA_PLATFORM=offscreen python -m unittest -v test_geometry3d_regressions."""
import math
import random
import unittest
from types import SimpleNamespace

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from geometry_module import GeometryPoint, simplify_xy, simplify_xy_to_target
from geometry3d_core import GeometryPoint3D, _clone_geom3d, simplify_xyz, simplify_xyz_to_target
from geometry3d_module import Geometry3DPage, circle_arc_through_three_points_3d

APP = QApplication.instance() or QApplication([])


class Geometry3DTests(unittest.TestCase):
    def setUp(self):
        self.page = Geometry3DPage()

    def tearDown(self):
        self.page.close()
        self.page.deleteLater()
        APP.processEvents()

    def load(self, points):
        p = self.page
        p.original = _clone_geom3d(points)
        p.simplified = _clone_geom3d(points)
        p._update_plot()
        return p.canvas_xy

    def test_short_segment_insertion_and_history(self):
        c = self.load([GeometryPoint3D(0,0,0,5), GeometryPoint3D(.5,10,10,15)])
        self.assertEqual(c._add_point_on_segment(0,.3), 1)
        p = self.page.simplified[1]
        self.assertEqual((p.time,p.x,p.y,p.z), (.15,3,3,8))
        self.assertEqual(len(self.page.undo_stack),1)
        self.assertTrue(all(v.edited is self.page.simplified for v in
                            [self.page.canvas_3d,c,self.page.canvas_xz,self.page.canvas_yz]))
        self.page.undo_3d()
        self.assertEqual(len(self.page.simplified),2)
        self.page.redo_3d()
        self.assertEqual(len(self.page.simplified),3)
        self.page.delete_selected_3d()  # no selection after redo
        c._select_point(1)
        self.page.delete_selected_3d()
        self.assertEqual(len(self.page.simplified),2)
        self.page.undo_3d()
        self.assertEqual(len(self.page.simplified),3)

    def test_real_qt_double_click_inserts_once(self):
        c = self.load([GeometryPoint3D(0,0,0,5),GeometryPoint3D(.5,10,10,15)])
        c.resize(640,480)
        c.show()
        APP.processEvents()
        c.canvas.draw()
        x,y = c.axes.transData.transform((5,5))
        ratio = c.canvas.device_pixel_ratio
        pos = QPoint(round(x/ratio), round((c.figure.bbox.height-y)/ratio))
        QTest.mouseClick(c.canvas,Qt.MouseButton.LeftButton,pos=pos)
        QTest.mouseDClick(c.canvas,Qt.MouseButton.LeftButton,pos=pos)
        QTest.mouseRelease(c.canvas,Qt.MouseButton.LeftButton,pos=pos)
        self.assertEqual(len(self.page.simplified),3)
        self.assertEqual(len(self.page.undo_stack),1)
        self.assertFalse(c._dragging_point)

    def test_double_click_existing_vertex_stops_drag(self):
        c = self.load([GeometryPoint3D(0,0,0,0),GeometryPoint3D(1,5,5,1),GeometryPoint3D(2,10,10,2)])
        c.canvas.draw()
        c._dragging_point = True
        c._drag_snapshot = _clone_geom3d(c.edited)
        x,y = c.axes.transData.transform((5,5))
        c._handle_double_click_xy(x,y)
        self.assertFalse(c._dragging_point)
        self.assertIsNone(c._drag_snapshot)
        self.assertEqual(c.selected_index,1)
        self.assertEqual(len(c.edited),3)

    def test_arc_highlights_current_selection_and_preserves_view(self):
        self.load([GeometryPoint3D(0,0,0,0),GeometryPoint3D(1,1,1,1),GeometryPoint3D(2,2,0,2)])
        p=self.page
        p.canvas_3d.axes.set_xlim(-10,10)
        p.canvas_3d.axes.set_ylim(-11,11)
        p.canvas_3d.axes.set_zlim(-12,12)
        p.arc_indices=[1]
        p.arc_select_mode=True
        p._update_plot(fit=False)
        self.assertIn(' ARC 1',[t.get_text() for t in p.canvas_3d.axes.texts])
        self.assertEqual(p.canvas_3d.axes.get_xlim(),(-10,10))
        self.assertEqual(p.canvas_3d.axes.get_ylim(),(-11,11))
        self.assertEqual(p.canvas_3d.axes.get_zlim(),(-12,12))
        p.arc_indices=[]
        p.arc_select_mode=False
        p._update_plot(fit=False)
        self.assertNotIn(' ARC 1',[t.get_text() for t in p.canvas_3d.axes.texts])

    def test_drag_each_projection_and_undo(self):
        for name in ['XY','XZ','YZ']:
            with self.subTest(projection=name):
                self.load([GeometryPoint3D(0,0,0,0),GeometryPoint3D(1,5,6,7),GeometryPoint3D(2,10,10,10)])
                c=getattr(self.page,'canvas_'+name.lower())
                c.canvas.draw()
                self.page.undo_stack=[]
                c._select_point(1)
                c._dragging_point=True
                c._point_moved=False
                self.page._canvas_edit_event('drag_begin',1,name,None,None,None)
                x,y=c.axes.transData.transform((3,4))
                c._canvas_xy_from_qt_event=lambda event: (x,y)
                event=SimpleNamespace(buttons=lambda: Qt.MouseButton.LeftButton)
                c._direct_mouse_move(event)
                c._direct_mouse_release(SimpleNamespace(button=lambda: Qt.MouseButton.LeftButton))
                p=self.page.simplified[1]
                expected={'XY':(3,4,7),'XZ':(3,6,4),'YZ':(5,3,4)}[name]
                for actual,want in zip((p.x,p.y,p.z),expected):
                    self.assertAlmostEqual(actual,want)
                self.assertEqual(p.time,1)
                self.assertEqual(len(self.page.undo_stack),1)
                self.page.undo_3d()
                p=self.page.simplified[1]
                self.assertEqual((p.x,p.y,p.z),(5,6,7))

    def test_marker_and_endpoint_protection(self):
        c=self.load([GeometryPoint3D(0,0,0,0),GeometryPoint3D(1,1,1,1,'M1'),GeometryPoint3D(2,2,0,2)])
        for i in range(3):
            c._select_point(i)
            self.assertFalse(c.delete_selected_point()[0])
        self.assertEqual(len(c.edited),3)
        self.assertEqual(len(self.page.undo_stack),0)

    def test_planar_rdp_matches_2d(self):
        rng=random.Random(19)
        points=[GeometryPoint(i,i,rng.uniform(-4,4),'M1' if i==8 else '') for i in range(20)]
        xyz=[GeometryPoint3D(p.time,p.x,p.y,7,p.marker,p.comment) for p in points]
        for eps in [0,.1,1,3,100]:
            self.assertEqual([p.time for p in simplify_xy(points,eps)],
                             [p.time for p in simplify_xyz(xyz,eps)])
        for target in [2,3,5,10,20]:
            a,ae,am=simplify_xy_to_target(points,target)
            b,be,bm=simplify_xyz_to_target(xyz,target)
            self.assertEqual(([p.time for p in a],ae,am),([p.time for p in b],be,bm))

    def test_tilted_arc_and_axis_generation(self):
        points=[GeometryPoint3D(0,1,0,1),GeometryPoint3D(10,0,1,0,'M1','keep'),GeometryPoint3D(20,-1,0,-1)]
        arc,center,radius,sweep=circle_arc_through_three_points_3d(*points,.01)
        self.assertTrue(all(abs(p.x-p.z)<1e-12 for p in arc))
        self.assertTrue(any(p.marker=='M1' and p.time==10 for p in arc))
        self.load(points)
        ok,msg=self.page.apply_three_point_arc_3d([2,0,1],.01)
        self.assertTrue(ok,msg)
        self.page.generate_axes_3d()
        axes=self.page.generated
        self.assertEqual(len(axes.axis1),len(axes.axis2))
        self.assertEqual(len(axes.axis1),len(axes.axis3))
        self.assertGreater(len(axes.axis1),0)


if __name__=='__main__':
    unittest.main()
