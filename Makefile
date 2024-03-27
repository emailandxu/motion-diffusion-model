name ?= denoise_more_condition_token
devices ?= 0
debug ?= 0

ifeq ($(debug), 0)
	python=python
else
	python=python -m debugpy --wait-for-client --listen 5678
endif

run=CUDA_VISIBLE_DEVICES=$(devices) $(python)

save_dir=save/$(name)

mini_step= --num_steps 1000	--save_interval 500	--log_interval 100
full_mask= --prefix_end 0.0 --suffix_start 1.0
almost_mask = --prefix_end 0.05 --suffix_start 0.95
# super slow
eval_during_training = --eval_during_training 

ckptpath ?= default
videopath ?= default

fit:
    # https://github.com/GuyTevet/motion-diffusion-model?tab=readme-ov-file#train-your-own-mdm
	$(run) \
	-m train.train_mdm \
	--save_dir $(save_dir) \
	--dataset humanml \
	--cond_noise_motion \
	--train_platform_type TensorboardPlatform \
	--save_interval 10000

generate:
	$(run) \
	-m sample.generate \
	--model_path $(ckptpath)  \
	--num_samples 10 \
	--num_repetitions 3

cond_noise_motion:
	$(run) \
	-m sample.cond_noise_motion \
	--model_path $(ckptpath) \
	--edit_mode in_between \
	$(full_mask)

to_mesh:
	$(run) \
	-m visualize.render_mesh \
	--input_path $(videopath)

tensorboard:
	tensorboard --logdir save/tensorboard --port 7777

clean:
	rm -rf $(save_dir)