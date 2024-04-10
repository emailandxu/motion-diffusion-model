import numpy as np
import torch
from data_loaders.humanml.scripts import motion_process

def random_start_end(lenght):
    start = np.random.uniform(0, lenght)
    end = np.random.uniform(lenght - start, lenght)
    return int(start*lenght), int(end*lenght)

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    bs, feat, _, t = motion.shape

    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text']for b in notnone_batches]
        cond['y'].update({'action_text': action_text})
    
    if "inp" in notnone_batches[0]:
        batch, channel, _, time = motion.shape

        # range from (0, noise_strength)
        background_noise_max = 1.
        specaug_noise_level = 1.

        noise_level = torch.rand_like(motion) * background_noise_max 

        if np.random.rand() > 0.25:
            c_start, c_end = random_start_end(channel)
            noise_level[:, c_start:c_end, :, :] = specaug_noise_level
        if np.random.rand() > 0.25:
            t_start, t_end = random_start_end(time)
            noise_level[:, :, :, t_start:t_end] = specaug_noise_level

        if np.random.rand() > 0.25:
            c_start, c_end = random_start_end(channel)
            noise_level[:, c_start:c_end, :, :] = 0.
        if np.random.rand() > 0.25:
            t_start, t_end = random_start_end(time)
            noise_level[:, :, :, t_start:t_end] = 0.

        noise_motion = ( 1 - noise_level ) * motion + noise_level * torch.randn_like(motion)

        cond['y'].update({"noise_motion": noise_motion})
        cond['y'].update({"noise_level": noise_level})

    return motion, cond

# an adapter to our collate func
def t2m_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
    } for b in batch]
    return collate(adapted_batch)


