# for videoname in mifei yuki honey;
for videoname in honey;
do 
    echo $videoname;
    #cd ProPose
    #make run videoname=$videoname
    #cd ../
    #make \
    #evaluate \
    #ckptpath=/home/tony/local-git-repo/motion-diffusion-model/save/denoise_variance_noise_condition/model000600000.pt \
    #input_motion=/home/tony/local-git-repo/motion-diffusion-model/ProPose/testing_data/output/"$videoname"_feat.npy
    # make \
    # to_mesh \
    # videodir=/home/tony/local-git-repo/motion-diffusion-model/save/denoise_variance_noise_condition/denoise_000600000_"$videoname"_feat
    make \
    to_mesh_render \
    videodir=/home/tony/local-git-repo/motion-diffusion-model/save/denoise_variance_noise_condition/denoise_000600000_"$videoname"_feat
done
