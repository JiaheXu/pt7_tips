import os
from subprocess import call

task_names = [
    'close_pen', 'pick_up_plate', 'pouring_into_bowl',
    'put_block_into_bowl', 'stack_block'
]

for split in ['train', 'eval']:
    for task_name in task_names:
        dirname = os.path.join(split, f'{task_name}+0')
        os.makedirs(dirname, exist_ok=True)
        for ep_ind in range(1, 100):
            if ((split == 'train' and ep_ind <= 38) or
               (split == 'eval' and ep_ind > 38)):
                epname = os.path.abspath(os.path.join(f'{task_name}/ep{ep_ind}.npy'))
                if os.path.exists(epname):
                    new_epname = os.path.abspath(os.path.join(f'{dirname}/ep{ep_ind}.npy'))
                    call(['ln', '-s', epname, new_epname])
