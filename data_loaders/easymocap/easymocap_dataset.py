import torch
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path
from data_loaders.easymocap.postprocess import EasyMocapOutputPostProcess
from data_loaders.humanml.scripts import motion_process

class EasyMocapDataset():
    @staticmethod
    def align(joints_position):
        def rotate_x(a):
            s, c = np.sin(a), np.cos(a)
            return np.array([[1,  0, 0, 0], 
                            [0,  c, s, 0], 
                            [0, -s, c, 0], 
                            [0,  0, 0, 1]]).astype(np.float32)
        time, joint, _ = joints_position.shape
        joints_position = joints_position.reshape(-1, 3)
        joints_position = (rotate_x(np.pi)[:3,:3] @ joints_position.T).T
        return joints_position.reshape(time, joint, 3)

    @staticmethod
    def downsample_array(arr, original_fps, target_fps):
        """
        Downsamples a numpy array from original_fps to target_fps.

        :param arr: Numpy array with shape (time, channel)
        :param original_fps: Original frames per second (e.g., 30)
        :param target_fps: Target frames per second (e.g., 20)
        :return: Downsampled numpy array
        """
        frame_ratio = original_fps / target_fps
        total_frames = arr.shape[0]
        selected_frames = np.arange(0, total_frames, frame_ratio).astype(int)
        return arr[selected_frames]

    def __init__(self, input_dirs, max_frame=196) -> None:
        self.max_frame = max_frame
        if type(input_dirs) is str:
            input_dirs = [input_dirs]

        self.features = []
        for thedir in input_dirs:
            poses = []
            for p in sorted(map(lambda p: p.as_posix(), Path(thedir).glob("*.json"))):
                with open(p, "rb") as f:
                    pose_output = json.load(f)
                    # pose_output['transl'][..., 1] = 0 # make it on ground
                    poses.append(pose_output)


            joints_position = np.concatenate(
                [
                    EasyMocapOutputPostProcess(pose).to_smpl_output().joints.cpu().numpy()
                    for pose in poses
                ],
                axis=0,
            )

            joints_position = EasyMocapDataset.align(joints_position)
            # joints_position = EasyMocapDataset.downsample_array(joints_position, original_fps=30, target_fps=20)
            feature = motion_process.tofeature(joints_position)
            self.features.append(feature)
    
    def __len__(self):
        return self.features
    
    def __getitem__(self, idx) -> np.ndarray:
        return self.features[idx]

if __name__ == "__main__":
    path = "/home/tony/local-git-repo/EasyMocap/output/yuki/vitpose/smpl"
    feat = EasyMocapDataset(path)[0]
    print(feat)
    print(feat.shape)