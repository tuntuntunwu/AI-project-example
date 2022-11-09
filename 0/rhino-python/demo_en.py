import rhinoscriptsyntax as rs


if __name__ == '__main__':
    # print
    rs.MessageBox("This is a MessageBox message")
    print("This is an output message")

    # point
    point_id = rs.AddPoint((10, 0, 0))
    rs.ObjectColor(point_id, (255, 0, 0))
    # curve
    curve_id0 = rs.AddCurve([(0, 0, 0), (0, 5, 5), (0, 10, 0)])
    rs.ObjectColor(curve_id0, (0, 255, 0))
    curve_id1 = rs.AddCurve([(0, 0, 0), (5, 5, 0), (0, 10, 0)])
    rs.ObjectColor(curve_id1, (0, 255, 0))
    # surface
    surface_id0 = rs.AddSrfPt([(0, 0, 0), (10, 0, 0), (10, -10, 0), (5, -5, -5)])
    rs.ObjectColor(surface_id0, (0, 0, 255))
    surface_id1 = rs.AddEdgeSrf([curve_id0, curve_id1])
    rs.ObjectColor(surface_id1, (0, 0, 255))
    
    # select objects and move them
    object_ids = rs.GetObjects("Pick some curves", rs.filter.curve)
    rs.MoveObjects(object_ids, (10, 0, 0))

    # export 2D image
    a = rs.Command("-_ViewCaptureToFile hehe.jpg _Enter")
    print(a)

