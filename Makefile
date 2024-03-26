name ?= denoise_fix_embeding_overwirte
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
	--model_path $(save_dir)/model000005000.pt  \
	--num_samples 10 \
	--num_repetitions 3

cond_noise_motion:
	$(run) \
	-m sample.cond_noise_motion \
	--model_path $(save_dir)/model000195000.pt \
	--edit_mode in_between \
	$(full_mask)

convert_to_mesh:
	$(run) \
	-m visualize.render_mesh \
	--input_path $(save_dir)/edit_denoise_fix_embeding_overwirte_000195000_in_between_seed10/sample00_rep00.mp4

tensorboard:
	tensorboard --logdir save/tensorboard --port 7777

clean:
	rm -rf $(save_dir)