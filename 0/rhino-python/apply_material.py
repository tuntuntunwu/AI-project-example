# https://docs.mcneel.com/rhino/mac/help/en-us/commands/materials.htm
# rhinoscriptsyntax can't create a new material, so we have to use RhinoCommon
# https://developer.rhino3d.com/samples/rhinocommon/add-material/
# https://developer.rhino3d.com/api/RhinoCommon/html/T_Rhino_DocObjects_Material.htm
import Rhino
import scriptcontext
import System.Drawing
import rhinoscriptsyntax as rs
import os


IMG_TYPE = ['.jpg', '.jpeg', '.png', '.gif']

def CreateMaterial(name="my_material", image_path="my_image.jpg"):
    # Create a Rhino material.
    rhino_material = Rhino.DocObjects.Material()
    rhino_material.Name = name
    # diffuse color = color
    rhino_material.DiffuseColor = System.Drawing.Color.Chocolate
    rhino_material.SpecularColor = System.Drawing.Color.CadetBlue

    texture = Rhino.DocObjects.Texture()
    texture.FileName = image_path
    rhino_material.SetTexture(texture, Rhino.DocObjects.TextureType.Bitmap)

    # Use the Rhino material to create a Render material.
    render_material = Rhino.Render.RenderMaterial.CreateBasicMaterial(rhino_material, scriptcontext.doc)
    scriptcontext.doc.RenderMaterials.Add(render_material)
    return render_material

def RemoveMaterial(render_material):
    return scriptcontext.doc.RenderMaterials.Remove(render_material)

def ApplyMaterial(object_id, render_material):
    obj = scriptcontext.doc.Objects.FindId(object_id)
    if obj:
      # Assign the material to the object.
      obj.RenderMaterial = render_material
      obj.CommitChanges()

    scriptcontext.doc.Views.Redraw()
    return Rhino.Commands.Result.Success


if __name__ == '__main__':
    obj = rs.GetObject("Select an object")
    if obj:
        images_path = os.path.join(os.getcwd(), 'images')
        for file in os.listdir(images_path):
            if os.path.splitext(file)[1].lower() in IMG_TYPE:
                render_material = CreateMaterial(name="my_material", image_path=os.path.join(images_path, file))
                ApplyMaterial(obj, render_material)
                rs.CurrentView('Perspective')
                rs.Command("-_ViewCaptureToFile " + file + " _Enter")
                RemoveMaterial(render_material)
