# -----------------------------GLOBAL-----------------------------------
name ?= denoise_variance_noise_condition
devices ?= 0
debug ?= 0

ifeq ($(debug), 0)
	python=python
else
	python=python -m debugpy --wait-for-client --listen 5678
endif

run=CUDA_VISIBLE_DEVICES=$(devices) $(python)

# -----------------------------TRAIN-----------------------------------

save_dir=save/$(name)
mini_step= --num_steps 1000	--save_interval 500	--log_interval 100
# super slow
eval_during_training = --eval_during_training 

fit:
    # https://github.com/GuyTevet/motion-diffusion-model?tab=readme-ov-file#train-your-own-mdm
	$(run) \
	-m train.train_mdm \
	--save_dir $(save_dir) \
	--dataset humanml \
	--cond_noise_motion \
	--train_platform_type TensorboardPlatform \
	--save_interval 10000

tensorboard:
	tensorboard --logdir save/tensorboard --port 7777

clean:
	rm -rf $(save_dir)

# ----------------------------EVALUATE-----------------------------------

ckptpath ?= default
full_mask= --prefix_end 0.0 --suffix_start 1.0
almost_mask = --prefix_end 0.05 --suffix_start 0.95
# assign ckptpath
evaluate:
	$(run) \
	-m sample.cond_noise_motion \
	--model_path $(ckptpath) \
	--edit_mode in_between \
	--num_samples 10 \
	--num_repetitions 3 \
	$(full_mask)


# ----------------------------VISUALIZE---------------------------------
videopath ?= default
videodir ?= default
#assign videopath
to_mesh_single:
	$(run) \
	-m visualize.render_mesh \
	--input_path $(videopath)

# assign videodir
to_mesh:
	ls $(videodir)/sample??_rep00.mp4 | xargs -P 4 -I {} make videopath={} to_mesh_single

# assign videodir
to_mesh_render:
	blender -b visualize/blender/SMPL_FRAMES.blend -P visualize/blender/batch_render.py -- $(videodir)
