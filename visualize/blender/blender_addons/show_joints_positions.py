import bpy
import numpy as np

# Example NumPy array (time, joint, 3)
# Replace with your actual motion 

#motion_data = np.load("/home/tony/local-git-repo/HumanML3D/joints/000020.npy")
motion_data = np.load("/home/tony/local-git-repo/ProPose/testing_data/indian/joints/indian_joints.npy")
motion_data = motion_data[:, :22]


# Create a cube for each joint
cubes = []
for joint_index in range(22):  # Assuming 22 joints
    bpy.ops.mesh.primitive_cube_add(size=0.1)  # Adjust cube size as needed
    cube = bpy.context.object
    cubes.append(cube)

# Animate each cube
for frame_number, frame_data in enumerate(motion_data):
    for joint_index, joint_pos in enumerate(frame_data):
        cube = cubes[joint_index]
        cube.location = joint_pos  # Set cube location
        cube.keyframe_insert(data_path="location", frame=frame_number)

# Set animation range
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = len(motion_data)