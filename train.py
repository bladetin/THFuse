import args
import time
import random
import torch
import torch.nn as nn
import utils
import dataset
from fusenet import Fusenet
from tqdm import tqdm, trange
from torch.optim import Adam
from os.path import join
from loss import final_ssim, TV_Loss
from loss_p import VggDeep,VggShallow
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '6'

def train(image_lists):

    image_mode = 'L'
    fusemodel = Fusenet() 
    vgg_ir_model = VggDeep() 
    vgg_vi_model = VggShallow()

    mse_loss = torch.nn.MSELoss() 
    TVLoss = TV_Loss() 
    L1_loss = nn.L1Loss() 

    fusemodel.cuda()
    vgg_ir_model.cuda()
    vgg_vi_model.cuda()

    tbar = trange(args.epochs, ncols=150)
    print('Start training.....')

    Loss_model = []
    Loss_ir_feature = [] 
    Loss_vi_feature = [] 

    all_ssim_loss = 0
    all_model_loss = 0.
    all_ir_feature_loss = 0.
    all_vi_feature_loss = 0. 
    save_num = 0

    for e in tbar:
        print('Epoch %d.....' % e)
        image_set, batches = dataset.load_dataset(image_lists, args.batch_size)
        fusemodel.train()
        count = 0
        for batch in range(batches):
            image_paths = image_set[batch * args.batch_size:(batch * args.batch_size + args.batch_size)]
            dir1 = "/path/to/your/own/dataset/vi" 
            dir2 = "/path/to/your/own/dataset/ir" 
            path1 = []
            path2 = []

            for path in image_paths:
                path1.append(join(dir1, path))
                path2.append(join(dir2, path))

            img_vi = utils.get_train_images_auto(path1, height=args.image_height, width=args.image_width, mode=image_mode)
            img_ir = utils.get_train_images_auto(path2, height=args.image_height, width=args.image_width, mode=image_mode)

            count += 1

            optimizer_model = Adam(fusemodel.parameters(), args.learning_rate)
            optimizer_model.zero_grad()

            optimizer_vgg_ir = Adam(vgg_ir_model.parameters(), args.learning_rate_d)
            optimizer_vgg_ir.zero_grad()

            optimizer_vgg_vi = Adam(vgg_vi_model.parameters(), args.learning_rate_d)
            optimizer_vgg_vi.zero_grad()

            img_vi = img_vi.cuda()
            img_ir = img_ir.cuda()

            outputs = fusemodel(img_vi, img_ir)

            ssim_loss_value = 0
            mse_loss_value = 0
            TV_loss_value = 0

            ssim_loss_temp = 1 - final_ssim(img_ir, img_vi, outputs)
            mse_loss_temp = mse_loss(img_ir,outputs) + mse_loss(img_vi,outputs)
            TVLoss_temp = TVLoss(img_ir,outputs)+TVLoss(img_vi,outputs)
            mse_loss_temp = 0
            TVLoss_temp = 0

            ssim_loss_value += ssim_loss_temp
            mse_loss_value += mse_loss_temp
            TV_loss_value +=TVLoss_temp

            ssim_loss_value /= len(outputs)
            mse_loss_value /= len(outputs)
            TV_loss_value /= len(outputs)

            model_loss = ssim_loss_value + 0.05 * mse_loss_value + 0.05 *  TV_loss_value
            model_loss.backward() 
            optimizer_model.step() 

        
            vgg_ir_fuse_out = vgg_ir_model(outputs.detach())[2]
            vgg_ir_out = vgg_ir_model(img_ir)[2]
            per_loss_ir = L1_loss(vgg_ir_fuse_out, vgg_ir_out)
            per_loss_ir_value = 0
            per_loss_ir_temp = per_loss_ir
            per_loss_ir_value += per_loss_ir_temp
            per_loss_ir_value /= len(outputs)
            per_loss_ir_value.backward()
            optimizer_vgg_ir.step()
        

        
            vgg_vi_fuse_out = vgg_vi_model(outputs.detach())[0]
            vgg_vi_out = vgg_vi_model(img_vi)[0]
            per_loss_vi = L1_loss(vgg_vi_fuse_out, vgg_vi_out)
            per_loss_vi_value = 0
            per_loss_vi_temp = per_loss_vi
            per_loss_vi_value += per_loss_vi_temp
            per_loss_vi_value /= len(outputs)
            per_loss_vi_value.backward()
            optimizer_vgg_vi.step()
       

            all_ssim_loss += ssim_loss_value.item()
            all_model_loss = all_ssim_loss
            all_ir_feature_loss += per_loss_ir_value.item()
            all_vi_feature_loss += per_loss_vi_value.item()

            if (batch + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:[{}/{}] fusemodel loss: {:.5f}".format(
                    time.ctime(), e + 1, count, batches,
                                  all_model_loss / args.log_interval)
                tbar.set_description(mesg)

                Loss_model.append(all_model_loss / args.log_interval)
                Loss_ir_feature.append(all_ir_feature_loss / args.log_interval)
                Loss_vi_feature.append(all_vi_feature_loss / args.log_interval)

                save_num += 1
                all_ssim_loss = 0.
                
            if (batch + 1) % (args.train_num//args.batch_size) == 0:
                fusemodel.eval()
                fusemodel.cpu()
                save_model_filename = "Epoch_" + str(e) + "_iters_" + str(count) + ".model"
                save_model_path = os.path.join(args.save_model_path, save_model_filename)
                torch.save(fusemodel.state_dict(), save_model_path)
                fusemodel.train()
                fusemodel.cuda()
                
    fusemodel.eval()
    fusemodel.cpu()
    save_model_filename = "Final_epoch_" + str(args.epochs) + ".model"
    save_model_path = os.path.join(args.save_model_path, save_model_filename)
    torch.save(fusemodel.state_dict(), save_model_path)

def main():
    images_path = utils.list_images(args.dataset_path)
    train_num = args.train_num 
    images_path = images_path[:train_num]
    random.shuffle(images_path)
    train(images_path)

if __name__ == "__main__":
    main()
