import json
import torch
from trimesh import Trimesh
from smplx import SMPL
from functools import lru_cache
from scipy.spatial.transform import Rotation as R

device = "cuda"

@lru_cache(maxsize=1)
def load_smpl_model(smpl_model_path="body_models/smpl/SMPL_NEUTRAL.pkl") -> SMPL:
    return SMPL(smpl_model_path).to(device)


class EasyMocapOutputPostProcess:
    def __init__(self, json_obj) -> None:
        self.smpl = load_smpl_model()
        self.json_obj = json_obj[0] if isinstance(json_obj, list) else json_obj # ['id', 'Rh', 'Th', 'poses', 'shapes']

    def to_smpl_pose_params(
        self,
    ):
        pred_betas = self.json_obj['shapes']
        transl = self.json_obj['Th']
        global_orient = self.json_obj['Rh']
        body_pose = self.json_obj['poses']

        paramters = {
            "betas": pred_betas,
            "transl": transl,
            "global_orient": global_orient,
            "body_pose": body_pose,
        }
        
        for key, value in paramters.items():
            paramters[key] = torch.Tensor(value).to(device)
        
        return paramters
    @torch.no_grad()
    def to_smpl_output(
        self,
    ):
        smpl_pose = self.to_smpl_pose_params()
        smpl_output = self.smpl.forward(**smpl_pose
        )
        return smpl_output

    def to_trimesh(
        self,
    ) -> Trimesh:
        smpl_output = self.to_smpl_output()
        # torch.Tensor: batchsize x 6890 x 3
        verts = smpl_output.vertices
        verts = verts.squeeze(0).cpu().numpy()
        # numpy.ndarray: 13776 x3
        faces = self.smpl.faces
        return Trimesh(vertices=verts, faces=faces)

    def to_obj(self, path: str):
        assert path.endswith(".obj")
        self.to_trimesh().export(path)


if __name__ == "__main__":
    path = "/home/tony/local-git-repo/motion-diffusion-model/data_loaders/easymocap/example/000000.json"
    pose_output = json.load(open(path, "rb"))[0] # this is an one element list of dict

    for key, value in pose_output.items():
        print(key, type(value))

    EasyMocapOutputPostProcess(pose_output).to_obj("temp.obj")
