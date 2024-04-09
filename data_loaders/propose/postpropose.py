import torch
import pickle
from trimesh import Trimesh
from smplx import SMPL
from functools import lru_cache
from scipy.spatial.transform import Rotation as R


@lru_cache(maxsize=1)
def load_smpl_model(smpl_model_path="body_models/smpl/SMPL_NEUTRAL.pkl") -> SMPL:
    return SMPL(smpl_model_path).cuda()


class ProPoseOutputPostProcess:
    def __init__(self, pose_output) -> None:
        self.smpl = load_smpl_model()
        self.pose_output = pose_output

    def to_smpl_pose_params(
        self,
    ):
        # squeeze the batch
        transl = self.pose_output["transl"].squeeze(0)
        pred_betas = self.pose_output["pred_shape"]  # 10 beta smpl paramters for shape

        pred_mats = self.pose_output["pred_theta_mats"].squeeze(0)  # is 24 x rotmat
        pred_mats = R.from_matrix(pred_mats.cpu().reshape(-1, 3, 3)).as_rotvec()
        pred_mats = torch.from_numpy(pred_mats).to(transl)
        global_orient = pred_mats[0].reshape(1, -1)
        body_pose = pred_mats[1:].reshape(1, -1)
        return {
            "betas": pred_betas,
            "transl": transl,
            "global_orient": global_orient,
            "body_pose": body_pose,
        }

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
    pose_output = pickle.load(open("pose_output.pkl", "rb"))

    for key, value in pose_output.items():
        print(key, type(value))

        if value is not None:
            print(key, value.shape)

    ProPoseOutputPostProcess(pose_output).to_obj("temp.obj")
