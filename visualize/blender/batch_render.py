import bpy
from pathlib import Path
from bpy.app.handlers import persistent

@persistent
def frame_change_handler(scene, depsgraph):
    current_frame = scene.frame_current
    total_frames = scene.frame_end - scene.frame_start + 1
    progress = current_frame / total_frames
    bpy.context.window_manager.progress_update(progress)

def render_animation(directory, fps=20):
    # Assuming you have a setup where you determine the total number of items to process
    total_items = sum(1 for _ in Path(directory).glob("*") if _.is_dir())
    processed_items = 0

    for p in sorted(Path(directory).glob("*")):
        if not p.is_dir():
            continue
        
        print(f"render {p}") 

        bpy.context.scene.obj_path = p.absolute().as_posix()
        bpy.ops.object.delete_objects_in_frames()
        bpy.ops.object.load_objs_into_frames()
        bpy.ops.object.turn_visibility_in_order()

        bpy.context.scene.render.filepath = p.with_suffix(".mp4").as_posix()
        bpy.context.scene.render.fps = fps

        # Before rendering, add the handler
        bpy.app.handlers.frame_change_pre.append(frame_change_handler)

        # Render the animation
        bpy.ops.render.render(animation=True)

        # After rendering, remove the handler to avoid it being called outside of rendering
        bpy.app.handlers.frame_change_pre.remove(frame_change_handler)

        processed_items += 1
        bpy.context.window_manager.progress_update(processed_items / total_items)

    bpy.context.window_manager.progress_end()

if __name__ == "__main__":
    # Start the process
    # blender -b SMPL_FRAMES.blend -P batch_render.py
    import sys
    assert "--" in sys.argv
    meshsroot = sys.argv[sys.argv.index("--") + 1]
    print(f"rendering smpl meshs at {meshsroot}.", sys.argv)
    render_animation(meshsroot)
