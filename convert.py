from mindspore import save_checkpoint, Tensor, load_checkpoint
import torch

def convert_model(pth_file_path, torch_key_list, mind_key_list):
    torch_params_dict = torch.load(pth_file_path,map_location=torch.device('cpu'))
    torch_params_dict = torch_params_dict["model"]
    params_dict_list = load_checkpoint("swintransformer_ascend_v150_imagenet2012_research_cv_top1acc80.96_top5acc95.37.ckpt")
    i = 0
    key_file = open("unknown.txt",'w')
    for key in mind_key_list:
        if torch_key_list[i] in torch_params_dict:
            value = torch_params_dict[torch_key_list[i]]
            # convert a torch tensor into numpy array
            # and then covert it into mindspore tensor
            value = Tensor(value.numpy())
            params_dict_list[key] = value
        else:
            print(key,file = key_file)
        i = i + 1
    # save mindspore checkpoint file
    # save_checkpoint(params_dict_list, "convert_from_torch.ckpt")

torch_key_list = []
mind_key_list = []
with open("ms_key2.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')  #去掉列表中每一个元素的换行符
        torch_key_list.append(line)
with open("ms_key1.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')  #去掉列表中每一个元素的换行符
        mind_key_list.append(line)
convert_model("swin_tiny_patch4_window7_224.pth",torch_key_list, mind_key_list)