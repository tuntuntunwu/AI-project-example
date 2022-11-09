简介：
1、RhinoCommon是跨平台的，基于.NET框架，使用C#语言实现的Rhino官方提供的SDK。
关于Rhino所有的二次开发：一、插件开发。二、代码/脚本开发 都使用RhinoCommon实现。其强大的图形化算法编辑器插件Grasshopper也同样完全由RhinoCommon实现。
RhinoCommon这一SDK支持使用C#、VB和python代码调用。由此可见，我们可使用python语言调用所有官方提供的API来实现我们想要的功能。如果不能，只能换软件了。
2、RhinoCommon其是使用C#语言实现的，因而其提供的API的名称、调用逻辑都是符合C#风格的。这之后RhinoCommon才提供的python调用也沿用了一开始的C#风格。
python调用RhinoCommon的API的例子：https://developer.rhino3d.com/guides/rhinopython/using-rhinocommon-from-python/
为了更便捷的python调用，官方将许多实用的功能封装在了rhinoscriptsyntax包，我们import rhinoscriptsyntax，便可使用许许多多实用的功能。
python使用rhinoscriptsyntax包的例子：https://developer.rhino3d.com/guides/rhinopython/python-rhinoscriptsyntax-introduction/

Rhino的python开发：
Rhino ----> RhinoCommon ----> RhinoCommon的API的python调用  ----> rhinoscriptsyntax
                 |                  (all functions)         (some easy-to-use functions)
                 |  A plugin
                 |
            Grasshopper ----> GhPython
                           (all functions)

二次开发纲领：
如果3D建模工程师直接使用Rhino建模、渲染：
rhinoscriptsyntax逻辑清晰、使用方便，但有些功能不能实现；RhinoCommon则提供了底层的，全部的API，可实现一切功能，但调用复杂。
因而我们使用的开发方法应当是：能使用rhinoscriptsyntax就使用rhinoscriptsyntax，不能实现的再使用python调用RhinoCommon提供API实现。
如果3D建模工程师使用Grasshopper建模渲染：
我们使用GhPython实现二次开发。

Appendix：
Rhino在软件内部实现了自己的实用的指令集Command，其可完成很多有用的操作。python可完整地调用这些Command。
https://docs.mcneel.com/rhino/mac/help/en-us/index.htm#commandlist/command_list.htm
使用RhinoCommon的API调用：Rhino.RhinoApp.RunScript("xxx")；使用rhinoscriptsyntax调用：rs.Command("xxx")

我的学习过程：
1、https://developer.rhino3d.com/guides/rhinocommon/
2、https://developer.rhino3d.com/guides/rhinopython/
3、任一Rhino软件学习课程
4、https://developer.rhino3d.com/api/RhinoScriptSyntax/
5、https://developer.rhino3d.com/api/RhinoCommon/html/R_Project_RhinoCommon.htm

开发时你必须要的References：
https://developer.rhino3d.com/api/RhinoScriptSyntax/
https://developer.rhino3d.com/api/RhinoCommon/html/R_Project_RhinoCommon.htm
https://docs.mcneel.com/rhino/mac/help/en-us/index.htm
https://docs.mcneel.com/rhino/mac/help/en-us/index.htm#commandlist/command_list.htm
https://docs.mcneel.com/rhino/mac/help/en-us/index.htm#information/rhinoscripting.htm
https://discourse.mcneel.com/

---------------------------------------------------------------------------------------------------------------------------------------

实现二次开发的过程：
1. 将Rhino软件设置为英文界面，搜索相关内容应以英文为主。 
2. 有关模型修改部分，要观看大秉操作流程，尝试用代码复现。
3. 相关概念一定要了解，询问大秉，在 https://docs.mcneel.com/rhino/mac/help/en-us/index.htm 中搜索。
4. 在提供的API文档中按英文直接搜索函数方法实现 https://developer.rhino3d.com/api/RhinoScriptSyntax/。
5. 不知如何实现的，去官方论坛 https://discourse.mcneel.com/ 或 谷歌 搜索。

