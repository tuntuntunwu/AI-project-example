# 运行时请删除中文字符  # 运行时请删除中文字符  # 运行时请删除中文字符  # 运行时请删除中文字符  # 运行时请删除中文字符  # 运行时请删除中文字符  # 运行时请删除中文字符 
# 运行时请删除中文字符  # 运行时请删除中文字符  # 运行时请删除中文字符  # 运行时请删除中文字符  # 运行时请删除中文字符  # 运行时请删除中文字符  # 运行时请删除中文字符 
# 运行时请删除中文字符  # 运行时请删除中文字符  # 运行时请删除中文字符  # 运行时请删除中文字符  # 运行时请删除中文字符  # 运行时请删除中文字符  # 运行时请删除中文字符 
import rhinoscriptsyntax as rs


if __name__ == '__main__':
    # 弹出一个窗口输出信息
    rs.MessageBox("This is a MessageBox message")
    # 代码错误Traceback的信息会以此种 MessageBox弹出窗口的形式 输出

    # 在Rhino提供的output面板内输出信息
    print("This is an output message")

    # Rhino是??-less的，只使用坐标完成所有操作
    # 点
    point_id = rs.AddPoint((10, 0, 0))
    rs.ObjectColor(point_id, (255, 0, 0))
    # 线
    # 创建线见: https://docs.mcneel.com/rhino/mac/help/en-us/index.htm#commands/curve.htm
    curve_id0 = rs.AddCurve([(0, 0, 0), (0, 5, 5), (0, 10, 0)])
    rs.ObjectColor(curve_id0, (0, 255, 0))
    curve_id1 = rs.AddCurve([(0, 0, 0), (5, 5, 0), (0, 10, 0)])
    rs.ObjectColor(curve_id1, (0, 255, 0))
    # 面
    # 创建面见: https://docs.mcneel.com/rhino/mac/help/en-us/index.htm#seealso/sak_surface.htm
    surface_id0 = rs.AddSrfPt([(0, 0, 0), (10, 0, 0), (10, -10, 0), (5, -5, -5)])
    rs.ObjectColor(surface_id0, (0, 0, 255))
    surface_id1 = rs.AddEdgeSrf([curve_id0, curve_id1])
    rs.ObjectColor(surface_id1, (0, 0, 255))
    
    # 人机交互
    # 复制选中的线
    object_ids = rs.GetObjects("Pick some curves", rs.filter.curve)
    rs.MoveObjects(object_ids, (10, 0, 0))
    
    # 导出2D图片
    # 更多CreatePreviewImage()参数见: https://developer.rhino3d.com/api/RhinoScriptSyntax/#document-CreatePreviewImage
    # a = rs.CreatePreviewImage("/Users/tuntuntunwu/Desktop/hehe.jpg")  # it doesn't work
    
    # Rhino在软件内部实现了自己的指令集Command，其可完成很多有用的操作，有哪些Command见: https://docs.mcneel.com/rhino/mac/help/en-us/commandlist/command_list.htm
    # Command使用见: http://docs.mcneel.com/rhino/mac/help/en-us/index.htm#information/rhinoscripting.htm
    # 更多ViewCaptureToFile命令参数见: http://docs.mcneel.com/rhino/mac/help/en-us/commands/viewcapture.htm#ViewCaptureToFile
    a = rs.Command("-_ViewCaptureToFile hehe.jpg _Enter")  # 计算机直接执行不弹出设置框
    print(a)

