from mindspore import save_checkpoint, Tensor, load_checkpoint
import torch

def convert_model(pth_file_path, torch_key_list, mind_key_list,ckpt_file_path):
    torch_params_dict = torch.load(pth_file_path,map_location=torch.device('cpu'))
    torch_params_dict = torch_params_dict["model"]
    params_dict_list = []
    i = 0
    mind_params_dict = load_checkpoint(ckpt_file_path)
    for key in mind_key_list:
        if torch_key_list[i] in torch_params_dict:
            value = torch_params_dict[torch_key_list[i]]
            # convert a torch tensor into numpy array
            # and then covert it into mindspore tensor
            value = Tensor(value.numpy())
            params_dict_list.append({"name": key, "data": value})
        elif "attn.q.bias" in key and i < 242:
            s = torch_key_list[i].replace("attn.q.bias","attn.qkv.bias")
            value = torch_params_dict[s]
            value = Tensor(value.numpy())
            num = value.shape[0]
            value = value[:(int(num/3))]
            params_dict_list.append({"name": key, "data": value})
        elif "attn.k.bias" in key and i < 242:
            s = torch_key_list[i].replace("attn.k.bias","attn.qkv.bias")
            value = torch_params_dict[s]
            value = Tensor(value.numpy())
            num = value.shape[0]
            value = value[int(num/3):int(num/3)*2]
            params_dict_list.append({"name": key, "data": value})
        elif "attn.v.bias" in key and i < 242:
            s = torch_key_list[i].replace("attn.v.bias","attn.qkv.bias")
            value = torch_params_dict[s]
            value = Tensor(value.numpy())
            num = value.shape[0]
            value = value[int(num/3)*2:int(num/3)*3]
            params_dict_list.append({"name": key, "data": value})
        elif "attn.q.weight" in key and i < 242:
            s = torch_key_list[i].replace("attn.q.weight","attn.qkv.weight")
            value = torch_params_dict[s]
            value = Tensor(value.numpy())
            num = value.shape[0]
            value = value[:(int(num/3))]
            params_dict_list.append({"name": key, "data": value})
        elif "attn.k.weight" in key and i < 242:
            s = torch_key_list[i].replace("attn.k.weight","attn.qkv.weight")
            value = torch_params_dict[s]
            value = Tensor(value.numpy())
            num = value.shape[0]
            value = value[int(num/3):int(num/3)*2]
            params_dict_list.append({"name": key, "data": value})
        elif "attn.v.weight" in key and i < 242:
            s = torch_key_list[i].replace("attn.v.weight","attn.qkv.weight")
            value = torch_params_dict[s]
            value = Tensor(value.numpy())
            num = value.shape[0]
            value = value[int(num/3)*2:int(num/3)*3]
            params_dict_list.append({"name": key, "data": value})
        else:
            print(key)
            value = mind_params_dict[key]
            params_dict_list.append({"name": key, "data": value})
        i = i + 1
        print(i)
    # save mindspore checkpoint file
    save_checkpoint(params_dict_list, "convert_from_torch.ckpt")

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
convert_model("swin_tiny_patch4_window7_224.pth",torch_key_list, mind_key_list,"swintransformer_ascend_v150_imagenet2012_research_cv_top1acc80.96_top5acc95.37.ckpt")