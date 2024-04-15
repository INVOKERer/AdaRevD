import os

import torch
from collections import OrderedDict

state_dict_pth = '/home/ubuntu/RevIR/experiments/GoPro_AdaRevD-B_classifier_hardsimple_naf_e3_classifyLoss_step3_ema_10k_3e-4/models/net_g_10000.pth'
save_dir = '/home/ubuntu/90t/personal_data/mxt/MXT/ckpts/classifier_rebuttal'
save_name = 'step3.pth'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, save_name)

checkpoint = torch.load(state_dict_pth)
state_dict = checkpoint["params"]

print(state_dict.keys())
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    # print(k)

    # v = repeat(v, )
    # if k[:5] == 'flow.':
    #     name = k[5:]  # remove `module.`
    #     new_state_dict[name] = v
    if k[:4] == 'ape.':
        name = k[4:]  # remove `module.`
        new_state_dict[name] = v
# print(new_state_dict)
torch.save(new_state_dict, save_path)