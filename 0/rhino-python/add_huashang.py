import rhinoscriptsyntax as rs
import random


# number_error = 1e-09
# def PointsEqual(pointA, pointB, rel_tol=number_error):
#     return abs(pointA[0] - pointB[0]) < rel_tol and abs(pointA[1] - pointB[1]) < rel_tol and abs(pointA[2] - pointB[2]) < rel_tol

def RandomSplitter(amount, number, min_val, max_val=None):
    if not max_val:
        max_val = amount - (number - 1) * min_val
    
    val_list = list()
    for i in range(number):
        if i != number - 1:
            portion = random.uniform(min_val, max_val)
            val_list.append(portion)
            max_val = max_val - portion + min_val
        else:
            val_list.append(max_val)
    
    return val_list

# TODO: 1.design parameters 2.implemented in inclined surfaces
def AddHuashang():
    y = -8.6
    # first-curve
    start_point = [random.uniform(-100, 100), y, random.uniform(-100, 100)]
    end_point = [random.uniform(-100, 100), y, random.uniform(-100, 100)]
    while True:
        point_list = list()
        point_list.append(start_point)
        point_number = random.randint(3, 6)
        # x randomized, z increase progressively
        amount = end_point[2] - start_point[2]
        z_increment = RandomSplitter(amount, point_number + 1, amount / 10)
        z = start_point[2]
        for i in range(point_number):
            z += z_increment[i]
            point = [random.uniform(-100, 100), y, z]
            point_list.append(point)
        point_list.append(end_point)
        a_id = rs.AddCurve(point_list)
        # no self-intersection
        if rs.CurveCurveIntersection(a_id) is None:
            break
        else:
            rs.DeleteObject(a_id)
    
    # second-curve
    n = 0
    while True:
        point_list = list()
        for point in rs.CurvePoints(a_id):
            point[0] += 5 + random.uniform(-2, 2)
            # point[2] += 3 + random.uniform(-1, 1)
            point_list.append(point)
        b_id = rs.AddCurve(point_list)
        if n > 1000:
            rs.MessageBox("We can't create second curve!!!")
            break
        else:
            n += 1
        # no self-intersection, no intersection with first curve
        if (rs.CurveCurveIntersection(b_id) is None) and (rs.CurveCurveIntersection(a_id, b_id) is None):
            break
        else:
            rs.DeleteObject(b_id)
    
    # third-curve
    while True:
        start_point = rs.CurveStartPoint(a_id)
        end_point = rs.CurveStartPoint(b_id)
        point_list = list()
        point_list.append(start_point)
        for i in range(random.randint(1, 1)):
            rate = random.uniform(0, 1)
            point = [rate * start_point[0] + (1 - rate) * end_point[0], y + random.uniform(3, 6), rate * start_point[2] + (1 - rate) * end_point[2]]
            point_list.append(point)
        point_list.append(end_point)
        c_id = rs.AddCurve(point_list)
        if rs.CurveCurveIntersection(c_id) is None:
            break
        else:
            rs.DeleteObject(c_id)
    
    # sweep 2
    rs.AddSweep2([a_id, b_id], [c_id])
    

if __name__ == '__main__':
    for i in range(3):
        AddHuashang()
