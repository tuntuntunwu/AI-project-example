import rhinoscriptsyntax as rs
import Rhino
import scriptcontext
import System.Drawing as dw
import os
import random
from collections import OrderedDict as odict
import json


def NormalizeLocation():
    ''' normalize the entire model location '''
    rs.CurrentView('Perspective')
    MoveAllObjectsToCentre()
    rs.Command("-_Camera _Show _Enter")
    # using camera location point and target point to set camera orientation
    # http://docs.mcneel.com/rhino/mac/help/en-us/commands/camera.htm
    rs.ViewCameraTarget(camera=[random.random(), 0, 0], target=[0, 0, 0])
    # ZoomExtents() finally determines camera location, but target location is still (0, 0, 0)
    rs.ZoomExtents(all=True)

def MoveAllObjectsToCentre():
    ''' move all objects' centre as a whole to origin (0, 0, 0) '''
    objs = rs.AllObjects()  # select all objects
    if objs:
        bbox = ObjectsBoundingBox(objs)
        # points returned in counter-clockwise order starting with the bottom rectangle of the box
        centre = [(x + y) / 2 for x, y in zip(bbox[0], bbox[6])]
        # rs.AddPoint(centre)
        for obj in objs:
            origin = rs.coerce3dpoint([0, 0, 0])  # !!! convert list to Point3d !!!
            centre = rs.coerce3dpoint(centre)
            rs.MoveObject(obj, origin - centre)

def ObjectsBoundingBox(objs):
    ''' calculate the bounding box of all these input objects '''
    max_x, max_y, max_z = min_x, min_y, min_z = rs.BoundingBox(objs[0])[0]  # initialize
    for obj in objs:
        points = rs.BoundingBox(obj)
        for point in points:
            max_x = point[0] if point[0] > max_x else max_x
            max_y = point[1] if point[1] > max_y else max_y
            max_z = point[2] if point[2] > max_z else max_z
            min_x = point[0] if point[0] < min_x else min_x
            min_y = point[1] if point[1] < min_y else min_y
            min_z = point[2] if point[2] < min_z else min_z
    
    bbox = rs.PointArrayBoundingBox([[max_x, max_y, max_z], [min_x, min_y, min_z]])
    # for bbox_point in bbox:
    #     rs.AddPoint(bbox_point)
    return bbox

def Set2DImage(display_mode=None, zoom_factor=1.0, rotate_up_right_angle=[0.0, 0.0], pan_up_right_coordinates=[0.0, 0.0], size=[512, 512], save_path=''):
    ''' set display mode and a basic visual angle for 2D image, then export this image '''
    # fix viewport size for correct export and rendering
    activeViewport = scriptcontext.doc.Views.ActiveView.ActiveViewport
    activeViewport.Size = dw.Size(size[0], size[1])
    # set display mode
    rs.ViewDisplayMode(view='Perspective', mode=display_mode)
    # set a basic visual angle
    # adjust parameters to get your expected output image 
    AdjustView(zoom_factor=zoom_factor, rotate_up_right_angle=rotate_up_right_angle, pan_up_right_coordinates=pan_up_right_coordinates)
    # file name must be the last parameter
    command = "-_ViewCaptureToFile _Width=" + str(size[0]) + " _Height=" + str(size[1]) + " " + os.path.join(save_path, "output_image.jpg") + " _Enter"
    rs.Command(command)

def AdjustView(zoom_factor=1.0, rotate_up_right_angle=[0.0, 0.0], pan_up_right_coordinates=[0.0, 0.0]):
    # Zoom in / out
    # using camera location point and target point to implement this function
    camera, target = rs.ViewCameraTarget()
    camera_location = [(camera[i] - target[i]) * zoom_factor + target[i] for i in range(3)]
    rs.ViewCameraTarget(camera=camera_location, target=target)
    # Rotate up / down
    rotate_up_angle = rotate_up_right_angle[0]
    if rotate_up_angle > 0:
        rs.RotateView(direction=2, angle=rotate_up_angle)  # direction 0=right 1=left 2=down 3=up
    else:
        rs.RotateView(direction=3, angle=-rotate_up_angle) # use 3 rather than 2 because it is 'view' rather than 'camera'
    # Rotate right / left
    rotate_right_angle = rotate_up_right_angle[1]
    if rotate_right_angle > 0:
        rs.RotateView(direction=1, angle=rotate_right_angle)
    else:
        rs.RotateView(direction=0, angle=-rotate_right_angle)
    # Pan
    # using construction plane to implement this function
    src_location = rs.ViewCamera()
    rs.Command("-_CPlane _View _Enter")
    cplane = rs.ViewCPlane()
    cplane_location = rs.XformWorldToCPlane(src_location, cplane)
    cplane_location[1] = cplane_location[1] + pan_up_right_coordinates[0]  # up / down
    cplane_location[0] = cplane_location[0] + pan_up_right_coordinates[1]  # right / left
    dst_location = rs.XformCPlaneToWorld(cplane_location, cplane)
    rs.ViewCamera(camera_location=dst_location)
    rs.Command("-_CPlane _World _Top _Enter")
        
def ExportImagesAndLabels(disturbed_rotate_up_right_angle_range=[0.0, 0.0], disturbed_pan_up_right_coordinates_range=[0.0, 0.0], number=1, size=[512, 512], save_path=''):
    ''' export thousands of images and labels with disturbance '''
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # quexian
    quexian_list = list()
    quexian_id0 = rs.AddPoint([0, 0, 110])
    rs.ObjectColor(quexian_id0, (0, 0, 255))
    quexian_id1 = rs.AddPoint([0, -50, 110])
    rs.ObjectColor(quexian_id1, (0, 255, 0))
    quexian_list.append(quexian_id0)
    quexian_list.append(quexian_id1)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    
    width, height = size
    # disturbance
    for i in range(number):
        rua = random.uniform(0, 1) * disturbed_rotate_up_right_angle_range[0]
        rra = random.uniform(0, 1) * disturbed_rotate_up_right_angle_range[1]
        puc = random.uniform(0, 1) * disturbed_pan_up_right_coordinates_range[0]
        prc = random.uniform(0, 1) * disturbed_pan_up_right_coordinates_range[1]
        AdjustView(rotate_up_right_angle=[rua, rra], pan_up_right_coordinates=[puc, prc])

        # export image
        image_rel = "image" + str(i) + ".jpg"
        command = "-_ViewCaptureToFile _Width=" + str(width) + " _Height=" + str(height) + " " + os.path.join(save_path, image_rel) + " _Enter"
        rs.Command(command)
        # export label - labelme json
        labelme = odict()
        labelme['version'] = "4.5.6"
        labelme['flags'] = odict()
        shapes = list()
        for quexian_id in quexian_list:
            shape = odict()
            shape['label'] = "quexian"
            
            # Rhino exports image in this way: camera target point is regarded as image centre, then it extends pixels according to width and height
            # so we must know: 1.the distance of quexian point world coordinates to camera target point. 2.the scale of world coordinates to pixel coordinates
            
            # 1.the distance of quexian point world coordinates to camera target point.
            # space straight-line equation: (x-x1) / (x2-x1) = (y-y1) / (y2-y1) = (z-z1) / (z2-z1)
            rs.Command("-_CPlane _View _Enter")
            cplane = rs.ViewCPlane()
            point1 = rs.XformWorldToCPlane(rs.PointCoordinates(quexian_id), cplane)
            point2 = rs.XformWorldToCPlane(rs.ViewCamera(), cplane)
            z = 0
            # y = (z-z1) / (z2-z1) * (y2-y1) + y1
            y = (z-point1[2]) / (point2[2]-point1[2]) * (point2[1]-point1[1]) + point1[1]
            # x = (z-z1) / (z2-z1) * (x2-x1) + x1
            x = (z-point1[2]) / (point2[2]-point1[2]) * (point2[0]-point1[0]) + point1[0]
            rs.Command("-_CPlane _World _Top _Enter")
            
            # 2.the scale of world coordinates to pixel coordinates
            activeViewport = scriptcontext.doc.Views.ActiveView.ActiveViewport
            origin = Rhino.Geometry.Point3d.Origin
            _, scale = activeViewport.GetWorldToScreenScale(origin)
            # convert world coordinates to screen pixel coordinates
            pixel_x = x * scale
            pixel_y = y * scale
            label_x = width / 2 + pixel_x
            label_y = height / 2 - pixel_y
            # rs.MessageBox((scale, x, y, pixel_x, pixel_y, label_x, label_y))
            
            shape['points'] = [[label_x - 2, label_y - 2], [label_x + 2, label_y + 2]]
            shape['group_id'] = None
            shape['shape_type'] = "rectangle"
            shape['flags'] = odict()
            shapes.append(shape)
        
        labelme['shapes'] = shapes
        labelme['imagePath'] = image_rel
        labelme['imageData'] = None
        labelme['imageHeight'] = height
        labelme['imageWidth'] = width
        label_abs = os.path.join(save_path, os.path.splitext(image_rel)[0] + ".json")
        # with open(label_abs, 'w') as f:
        with open(os.path.splitext(image_rel)[0] + ".json", 'w') as f:
            json.dump(labelme, f)


if __name__ == '__main__':
    size = [520, 1308]
    save_path = "/Users/tuntuntunwu/Desktop"
    # we stipulate that we must use Perspective view to normalize model and export images+labels
    # we stipulate that we must use Perspective view to normalize model and export images+labels
    # we stipulate that we must use Perspective view to normalize model and export images+labels
    view = rs.CurrentView('Perspective')
    # normalize model location
    NormalizeLocation()
    # determine 2D image setting
    Set2DImage(display_mode='Rendered', zoom_factor=0.5, rotate_up_right_angle=[10.0, 0.0], pan_up_right_coordinates=[70.0, 0.0], size=size, save_path=save_path)
    # export images and labels with disturbance
    ExportImagesAndLabels([-3.0, 3.0], [-3.0, 3.0], number=3, size=size, save_path=save_path)
