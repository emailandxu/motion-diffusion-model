#%%
import os
import sys
import numpy as np
#%%
os.chdir("/home/tony/local-git-repo/motion-diffusion-model/")
# sys.path.append("/home/tony/local-git-repo/motion-diffusion-model/")

from matplotlib import pyplot as plt
from data_loaders.propose.propose_dataset import ProposeDataset
#%%

dataset = ProposeDataset([
    "/home/tony/local-git-repo/motion-diffusion-model/dataset/ProPose/daiqin_full/output",
    "/home/tony/local-git-repo/motion-diffusion-model/dataset/ProPose/mifei/output"
])
#%%
daiqin, mifei = dataset
#%%

motion = daiqin
# time, 263-66+5
time, channel = motion.shape

max_frames = (time // 196) * 196
motion = motion[:max_frames]
motion = motion.reshape(-1, 196, 263-66+5)
outlier = motion[4]

plt.imshow((outlier[1:]-outlier[:-1])[..., np.newaxis])
plt.show()
# %%
