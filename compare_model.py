import torch
import mindspore 
from mindspore import load_checkpoint


def traversal_params(pth_file_path, ckpt_file_path):
    # load pth file as a dictionary
    torch_params_dict = torch.load(pth_file_path,map_location=torch.device('cpu'))
    # traversal a params dictionary
    
    # torch_f = open('torch_key.txt','w')
    
    for k, v in torch_params_dict.items():
        print(v["layers.0.blocks.0.attn.qkv.weight"].shape)
        # for kk,vv in v.items(): 
        #     print(kk , file = torch_f)

    # load mindspore ckpt file as a dictionary
    ms_f = open('ms_key2.txt','w')
    mind_params_dict = load_checkpoint(ckpt_file_path)
    print(mind_params_dict["model.layers.0.blocks.0.attn.q.weight"])
    # for k, v in mind_params_dict.items():
    #     print(k , file = ms_f)
        
traversal_params("swin_tiny_patch4_window7_224.pth","swintransformer_ascend_v150_imagenet2012_research_cv_top1acc80.96_top5acc95.37.ckpt")