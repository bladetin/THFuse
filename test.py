import torch
from torch.autograd import Variable 
import utils
import numpy as np
import time
from fusenet import Fusenet
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def load_model(path):
    if not os.path.exists(path):
        raise ValueError('Invalid path: {}'.format(path))

    fuse_net = Fusenet()
    fuse_net.load_state_dict(torch.load(path))
    para = sum([np.prod(list(p.size())) for p in fuse_net.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(fuse_net._get_name(), para * type_size / 1000 / 1000))

    fuse_net.eval()
    fuse_net.cuda()

    return fuse_net

def generate_fuse_image(model, vi, ir):
    out = model(vi, ir)
    return out

def fuse_test(model, vi_path, ir_path, output_path_root, index):
    if not os.path.exists(vi_path):
        print("Error: {} does not exist".format(vi_path))
        return
    if not os.path.exists(ir_path):
        print("Error: {} does not exist".format(ir_path))
        return
    if not os.path.exists(output_path_root):
        print("Error: {} does not exist".format(output_path_root))
        return

    vi_img = utils.get_test_images(vi_path, height=None, width=None)
    ir_img = utils.get_test_images(ir_path, height=None, width=None)

    out = utils.get_image(vi_path, height=None, width=None)

    vi_img = vi_img.cuda()
    ir_img = ir_img.cuda()
    vi_img = Variable(vi_img, requires_grad=False)
    ir_img = Variable(ir_img, requires_grad=False)

    img_fusion = generate_fuse_image(model, vi_img, ir_img) 

    file_name = 'fusion_' + str(index) + '.png'
    output_path = output_path_root + file_name

    if torch.cuda.is_available():
        img = img_fusion.cpu().clamp(0, 255).numpy()
    else:
        img = img_fusion.clamp(0, 255).numpy()
    img = img.astype('uint8')
    utils.save_images(output_path, img, out)
    print(output_path)


def main():
    vi_path = "/path/to/visible"
    ir_path = "/path/to/infrared"
    output_path = '/path/to/output'
    if os.path.exists(output_path) is False:
        os.mkdir(output_path)
    model_path = "/path/to/model"
    
    with torch.no_grad():
        if not os.path.exists(model_path):
            print("Model file does not exist")
            return
        model = load_model(model_path) 
        for i in range(10):
            index = i + 1
            visible_path = vi_path + str(index) + '.bmp'
            infrared_path = ir_path + str(index) + '.bmp'
            if not os.path.exists(visible_path):
                print("Visible image does not exist")
                return
            if not os.path.exists(infrared_path):
                print("Infrared image does not exist")
                return
            start = time.time()
            fuse_test(model, visible_path, infrared_path, output_path, index)
            end = time.time()
            print('time:', end - start, 'S')
    print('Done......')

if __name__ == "__main__":
    main()
