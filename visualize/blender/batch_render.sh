
my_list=(
    "/home/tony/local-git-repo/motion-diffusion-model/save/denoise_fix_embeding_overwirte/edit_denoise_fix_embeding_overwirte_000195000_in_between_seed10_cfg"
    "/home/tony/local-git-repo/motion-diffusion-model/save/denoise_more_condition_token/denoise_000330000_in_between_seed10"
    "/home/tony/local-git-repo/motion-diffusion-model/save/denoise_variance_noise_condition/denoise_000230000_in_between_seed10")

# Iterate over the list
for videodir in "${my_list[@]}"
do
    echo "$videodir"

    stem=$(basename "$videodir")
    mkdir -p $HOME/Videos/$stem
    cp $videodir/sample0?.mp4 $HOME/Videos/$stem

    make to_mesh videodir=$videodir
    make to_mesh_render videodir=$videodir
    cp $videodir/*obj.mp4 $HOME/Videos/$stem

done



