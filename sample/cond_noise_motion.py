# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import sys
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import edit_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel, NoClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders import humanml_utils
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from data_loaders.tensors import collate
from pathlib import Path
from model.rotation2xyz import Rotation2TheMotion
from visualize.simplify_loc2rot import joints2smpl

NOISE_LEVEL_MIN = 0.0
NOISE_LEVEL_MAX = 0.0

to_the_motion = Rotation2TheMotion("cuda")
j2s = joints2smpl(num_frames=196, device_id=0, cuda=True)

def rand_like_from(arr, low, high):
    low, high = (high, low) if low < high else (low, high)
    return (high - low) * torch.rand_like(arr) + low


def main():
    args = edit_args()
    fixseed(args.seed)

    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60

    if args.input_motion:
        # from data_loaders.propose.propose_dataset import ProposeDataset
        # full_motion = ProposeDataset([args.input_motion])[0] # time, channel=253
        from data_loaders.easymocap.easymocap_dataset import EasyMocapDataset
        full_motion = EasyMocapDataset([args.input_motion])[0]
        print(full_motion.shape)
        chunks = (len(full_motion) // max_frames) + 1
        args.num_samples = chunks


    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    fps = 12.5 if args.dataset == 'kit' else 20
    dist_util.setup_dist(args.device)
    if out_path == '':
        if args.input_motion:
            out_path = os.path.join(os.path.dirname(args.model_path),
                                    'denoise_{}_{}_{}'.format(niter, Path(args.input_motion).parent.stem, f"{NOISE_LEVEL_MIN}_{NOISE_LEVEL_MAX}"))
        else:
            out_path = os.path.join(os.path.dirname(args.model_path),
                                    'denoise_{}_{}_seed{}'.format(niter, args.edit_mode, args.seed))

        if args.text_condition != '':
            out_path += '_' + args.text_condition.replace(' ', '_').replace('.', '')

    print('Loading dataset...')
    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='quick',
                              hml_mode='train')  # in train mode, you get both text and motion.
    # data.fixed_length = n_frames
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    model = NoClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    if not args.input_motion:
        iterator = iter(data)
        input_motions, model_kwargs = next(iterator)
    else:
        m_length = max_frames if len(full_motion) > max_frames else len(full_motion)
        assert chunks == args.num_samples, f"{chunks} must equal args.num_samples"

        collate_args = []
        for i in range(chunks):
            motion = full_motion[m_length*i:m_length*(i+1)]
            #normalization
            motion = (motion - data.dataset.mean) / data.dataset.std

            #padding
            if m_length < max_frames:
                motion = np.concatenate([motion,np.zeros((max_frames-m_length, motion.shape[1]))], axis=0)

            motion = motion.transpose()[:, np.newaxis] # channel, 1, time
            motion = torch.from_numpy(motion).float()
            collate_args.append({'inp': motion, 'tokens': None, 'lengths': max_frames})

        input_motions, model_kwargs = collate(collate_args)

        noise_level = rand_like_from(input_motions, NOISE_LEVEL_MIN, NOISE_LEVEL_MAX)
        # noise_level[:, :9] = rand_like_from(noise_level[:, :9], 1.0, 1.0)


        # batch, channel, _, time = input_motions.shape
        input_motions


        # root trans
        # input_motions[:, :4, :, :] = torch.randn(batch, 4, 1, time).to(input_motions)
        # noise_level[:, :4, :, :] = 1.
        # input_motions[:, -4:, :, :] = torch.randn(batch, 4, 1, time).to(input_motions)
        # noise_level[:, -4:, :, :] = 1.
        
        # model_kwargs['y']['noise_motion'] = ( 1- noise_level) * input_motions + noise_level * torch.randn_like(input_motions)
        # model_kwargs['y']['noise_level'] = torch.zeros_like(noise_level)
        # model_kwargs['y']['noise_level'][:, humanml_utils.HML_UPPER_BODY_MASK]  = noise_level[:, humanml_utils.HML_UPPER_BODY_MASK] 

        # model_kwargs['y']['noise_motion'][:, humanml_utils.HML_UPPER_BODY_MASK] = \
        #         ( 1- noise_level[:, humanml_utils.HML_UPPER_BODY_MASK] ) * input_motions[:, humanml_utils.HML_UPPER_BODY_MASK] + \
        #         noise_level[:, humanml_utils.HML_UPPER_BODY_MASK]  * torch.randn_like(input_motions)[:, humanml_utils.HML_UPPER_BODY_MASK] 
        
        # model_kwargs['y']['noise_motion'] = ( 1- noise_level) * input_motions + noise_level * torch.randn_like(input_motions)
        
        model_kwargs['y']['noise_motion'] = input_motions
        model_kwargs['y']['noise_level'] = noise_level

        print(input_motions.shape)
   
    input_motions = input_motions.to(dist_util.dev())
    model_kwargs['y']['noise_motion'] = model_kwargs['y']['noise_motion'].to(dist_util.dev())  
    model_kwargs['y']['noise_level'] = model_kwargs['y']['noise_level'].to(dist_util.dev())               
             
    texts = [args.text_condition] * args.num_samples
    model_kwargs['y']['text'] = texts
    # if args.text_condition == '':
    #     args.guidance_param = 0.  # Force unconditioned generation

    # add inpainting mask according to args
    assert max_frames == input_motions.shape[-1]
    gt_frames_per_sample = {}
    model_kwargs['y']['inpainted_motion'] = input_motions
    if args.edit_mode == 'in_between':
        model_kwargs['y']['inpainting_mask'] = torch.ones_like(input_motions, dtype=torch.bool,
                                                               device=input_motions.device)  # True means use gt motion
        for i, length in enumerate(model_kwargs['y']['lengths'].cpu().numpy()):
            start_idx, end_idx = int(args.prefix_end * length), int(args.suffix_start * length)
            gt_frames_per_sample[i] = list(range(0, start_idx)) + list(range(end_idx, max_frames))
            model_kwargs['y']['inpainting_mask'][i, :, :,
            start_idx: end_idx] = False  # do inpainting in those frames
    elif args.edit_mode == 'upper_body':
        model_kwargs['y']['inpainting_mask'] = torch.tensor(humanml_utils.HML_LOWER_BODY_MASK, dtype=torch.bool,
                                                            device=input_motions.device)  # True is lower body data
        model_kwargs['y']['inpainting_mask'] = model_kwargs['y']['inpainting_mask'].unsqueeze(0).unsqueeze(
            -1).unsqueeze(-1).repeat(input_motions.shape[0], 1, input_motions.shape[2], input_motions.shape[3])

    elif args.edit_mode == 'lower_body':
        model_kwargs['y']['inpainting_mask'] = torch.tensor(humanml_utils.HML_UPPER_BODY_MASK, dtype=torch.bool,
                                                            device=input_motions.device)  # True is lower body data
        model_kwargs['y']['inpainting_mask'] = model_kwargs['y']['inpainting_mask'].unsqueeze(0).unsqueeze(
            -1).unsqueeze(-1).repeat(input_motions.shape[0], 1, input_motions.shape[2], input_motions.shape[3])

    all_motions = []
    all_lengths = []
    all_text = []

    samples = []
    model_inputs = []

    for rep_i in range(args.num_repetitions):
        print(f'### Start sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
        model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        
        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, max_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )


        # Recover XYZ *positions* from HumanML3D vector representation
        if model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263-66+5 else 21

            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            model_input = data.dataset.t2m_dataset.inv_transform(input_motions.cpu().permute(0, 2, 3, 1)).float()
            
            samples.append(sample)
            model_inputs.append(model_input)

            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        all_text += model_kwargs['y']['text']
        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")

    samples = np.concatenate(samples, axis=0)
    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions,
             'samples':samples, "model_inputs":model_inputs})

    if args.to_the_motion:
        the_motion = all_motions
        print(the_motion.shape)
        the_motion, _ = j2s.joint2smpl(the_motion[0].transpose(2, 0, 1))
        print(the_motion.shape)
        the_motion = the_motion.detach().cpu().numpy()
        the_motion = to_the_motion(torch.tensor(the_motion), mask=None,
                                        pose_rep='rot6d', translation=True, glob=True,
                                        jointstype='vertices',
                                        # jointstype='smpl',  # for joint locations
                                        vertstrans=True)
        np.save(args.input_motion + "/../" + "the_motion", the_motion)
        print("save the motion at", args.input_motion + "/../" + "the_motion")
        sys.exit(0)
    
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

    # Recover XYZ *positions* from HumanML3D vector representation
    
    if model.data_rep == 'hml_vec':
        def recover_from_humanml3d(input_motions):
            input_motions = data.dataset.t2m_dataset.inv_transform(input_motions.cpu().permute(0, 2, 3, 1)).float()
            input_motions = recover_from_ric(input_motions, n_joints)
            input_motions = input_motions.view(-1, *input_motions.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()
            return input_motions
        
        input_motions = recover_from_humanml3d(input_motions)
        noise_motion =  recover_from_humanml3d(model_kwargs['y']['noise_motion'])

    for sample_i in range(args.num_samples):
        caption = 'Input Motion'
        length = model_kwargs['y']['lengths'][sample_i]
        motion = input_motions[sample_i].transpose(2, 0, 1)[:length]
        the_noise_motion = noise_motion[sample_i].transpose(2, 0, 1)[:length]

        save_file = 'input_motion{:02d}.mp4'.format(sample_i)
        animation_save_path = os.path.join(out_path, save_file)
        rep_files = [animation_save_path]
        print(f'[({sample_i}) "{caption}" | -> {save_file}]')
        plot_3d_motion(animation_save_path, skeleton, the_noise_motion, title=caption,
                       dataset=args.dataset, fps=fps, vis_mode='gt',
                       gt_frames=gt_frames_per_sample.get(sample_i, []))
        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i*args.batch_size + sample_i]
            if caption == '':
                caption = 'Denoise [{}] unconditioned'.format(args.edit_mode)
            else:
                caption = 'Denoise [{}]: {}'.format(args.edit_mode, caption)
            length = all_lengths[rep_i*args.batch_size + sample_i]
            motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]
            save_file = 'sample{:02d}_rep{:02d}.mp4'.format(sample_i, rep_i)
            animation_save_path = os.path.join(out_path, save_file)
            rep_files.append(animation_save_path)
            print(f'[({sample_i}) "{caption}" | Rep #{rep_i} | -> {save_file}]')
            plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                           dataset=args.dataset, fps=fps, vis_mode=args.edit_mode,
                           gt_frames=gt_frames_per_sample.get(sample_i, []))
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion

        all_rep_save_file = os.path.join(out_path, 'sample{:02d}.mp4'.format(sample_i))
        ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
        hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions+1}'
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_file}'
        os.system(ffmpeg_rep_cmd)
        print(f'[({sample_i}) "{caption}" | all repetitions | -> {all_rep_save_file}]')

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


if __name__ == "__main__":
    main()
