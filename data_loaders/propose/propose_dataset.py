import torch
import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path
from data_loaders.propose.postpropose import ProPoseOutputPostProcess
from data_loaders.humanml.scripts import motion_process

class ProposeDataset():
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


    def __init__(self, input_dirs, max_frame=196) -> None:
        self.max_frame = max_frame
        if type(input_dirs) is str:
            input_dirs = [input_dirs]

        self.features = []
        for thedir in input_dirs:
            poses = []
            for p in sorted(map(lambda p: p.as_posix(), Path(thedir).glob("*.pkl"))):
                with open(p, "rb") as f:
                    pose_output = pickle.load(f)
                    # pose_output['transl'][..., 1] = 0 # make it on ground
                    poses.append(pose_output)


            joints_position = np.concatenate(
                [
                    ProPoseOutputPostProcess(pose).to_smpl_output().joints.cpu().numpy()
                    for pose in poses
                ],
                axis=0,
            )

            joints_position = ProposeDataset.align(joints_position)
            feature = motion_process.tofeature(joints_position)
            self.features.append(feature)
    
    def __len__(self):
        return self.features
    
    def __getitem__(self, idx) -> np.ndarray:
        return self.features[idx]