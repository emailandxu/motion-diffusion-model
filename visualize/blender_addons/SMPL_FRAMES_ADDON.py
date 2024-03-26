bl_info = {
    "name": "MY_SMPL_FRAMES_ANIMATION",
    "blender": (4, 0, 0),
    "category": "Object",
}


import bpy
import os

def delete_objects_in_collection(collection_name):
    collection = bpy.data.collections.get(collection_name)
    if collection:
        for obj in collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

def load_objs_into_collection(path, collection_name):
    # Check if the collection already exists
    if collection_name not in bpy.data.collections:
        # Create a new collection
        new_collection = bpy.data.collections.new(collection_name)

        # Link the new collection to the scene's collection
        bpy.context.scene.collection.children.link(new_collection)

        # Set the newly created collection as the active one
        bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children[collection_name]


    # Check if the directory exists
    if not os.path.exists(path):
        print("Directory does not exist:", path)
    else:
        # Loop through all files in the directory
        for filename in sorted(os.listdir(path)):
            if filename.lower().endswith(".obj"):
                # Construct the full file path
                file_path = os.path.join(path, filename)

                # Load the OBJ file
                bpy.ops.wm.obj_import(filepath=file_path)
                print("Loaded:", filename)


def turn_visibility_in_order():
    # Set the name of the collection containing the meshes
    collection_name = "Frames"

    # Get the collection by name
    collection = bpy.data.collections.get(collection_name)

    if collection:
        # Get all objects in the collection
        objects = collection.objects
        
        # Set the start frame of the animation
        start_frame = 0
        
        # Set initial frame
        frame = start_frame
        
        # Iterate over each object in the collection
        for obj in objects:
            print(obj.name)
            # Set both viewport and render visibility of all objects in collection to False
            for other_obj in objects:
                if other_obj != obj:
                    other_obj.hide_viewport = True
                    other_obj.hide_render = True
                    other_obj.keyframe_insert(data_path="hide_viewport", frame=frame)
                    other_obj.keyframe_insert(data_path="hide_render", frame=frame)
            
            # Set both viewport and render visibility of the current object to True at the current frame
            obj.hide_viewport = False
            obj.hide_render = False
            obj.keyframe_insert(data_path="hide_viewport", frame=frame)
            obj.keyframe_insert(data_path="hide_render", frame=frame)
            
            # Increment the frame for the next object
            frame += 1

    # Set the end frame of the animation
    end_frame = frame - 1

    # Set the end frame of the timeline
    bpy.context.scene.frame_end = end_frame


class TurnVisibilityInOrderOperator(bpy.types.Operator):
    bl_idname = "object.turn_visibility_in_order"
    bl_label = "Turn Visibility In Order"

    def execute(self, context):
        turn_visibility_in_order()
        return {'FINISHED'}

class CollectionManagementPanel(bpy.types.Panel):
    bl_label = "Collection Management"
    bl_idname = "OBJECT_PT_collection_management"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tools'

    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene, "obj_path")
        layout.operator("object.load_objs_into_frames", text="Load OBJs into 'Frames'")
        layout.operator("object.delete_objects_in_frames", text="Delete Objects in 'Frames'")
        layout.operator("object.turn_visibility_in_order", text="Turn Visibility In Order")  # New button for turning visibility in order

class LoadObjsOperator(bpy.types.Operator):
    bl_idname = "object.load_objs_into_frames"
    bl_label = "Load OBJs Operator"

    def execute(self, context):
        load_objs_into_collection(context.scene.obj_path, "Frames")
        return {'FINISHED'}

class DeleteObjectsOperator(bpy.types.Operator):
    bl_idname = "object.delete_objects_in_frames"
    bl_label = "Delete Objects Operator"

    def execute(self, context):
        delete_objects_in_collection("Frames")
        return {'FINISHED'}

def register():
    # Register the new operator along with the existing ones
    bpy.utils.register_class(TurnVisibilityInOrderOperator)
    bpy.utils.register_class(CollectionManagementPanel)
    bpy.utils.register_class(LoadObjsOperator)
    bpy.utils.register_class(DeleteObjectsOperator)
    bpy.types.Scene.obj_path = bpy.props.StringProperty(name="OBJ Path")

def unregister():
    # Unregister the new operator along with the existing ones
    bpy.utils.unregister_class(TurnVisibilityInOrderOperator)
    bpy.utils.unregister_class(CollectionManagementPanel)
    bpy.utils.unregister_class(LoadObjsOperator)
    bpy.utils.unregister_class(DeleteObjectsOperator)
    del bpy.types.Scene.obj_path

if __name__ == "__main__":
    register()
