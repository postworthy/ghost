print("started imports")

import sys
import argparse
import time
import cv2
from PIL import Image
import os
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as scheduler
from torch.cuda.amp import GradScaler, autocast
# custom imports

from network.AEI_Net import *
from network.MultiscaleDiscriminator import *
from utils.training.Dataset import FaceEmbedVGG2, FaceEmbed, CelebADataset
from utils.training.image_processing import make_image_list, get_faceswap
from utils.training.detector import detect_landmarks, paint_eyes
from AdaptiveWingLoss.core import models
from arcface_model.iresnet import iresnet100

import insightface
import onnxruntime
from onnx import numpy_helper
import onnx
from insightface.utils import face_align

from utils.training.helpers import get_hsv, stuck_loss_func, batch_edge_loss, emboss_loss_func, structural_loss, compute_eye_loss
from utils.training.losses import compute_discriminator_loss
from utils.training.upsampler  import upscale
from models.MultiScalePerceptualColorLoss import MultiScalePerceptualColorLoss

print("finished imports")

print("started globals")
multiscale_color_loss_func = MultiScalePerceptualColorLoss()
print("finished globals")

def is_any_nan(loss, name):
    if torch.isnan(loss).any():
        print(f"Warning: {name} contains NaN values")
        return True
    else:
        return False

def train_one_epoch(G: 'generator model', 
                    opt_G: "generator opt", 
                    scheduler_G: "scheduler G opt",
                    D: 'discriminator model',
                    opt_D: "generator opt", 
                    scheduler_D: "scheduler G opt",
                    netArc: 'ArcFace model',
                    teacher: 'Teacher model',
                    model_ft: 'Landmark Detector',
                    args: 'Args Namespace',
                    dataloader: torch.utils.data.DataLoader,
                    device: 'torch device',
                    epoch:int,
                    loss_adv_accumulated:int):
    
    universal_multiplier = 100
    verbose_output = args.verbose_output
    total_lossD = torch.tensor(0.0).to(device)
    for iteration, data in enumerate(dataloader):
        start_time = time.time()
        
        Xs_orig, Xs, _Xt_raw, Xt, same_person = data
        
        Xs_orig = Xs_orig.to(device)
        Xs = Xs.to(device)
        Xt = Xt.to(device)
        same_person = same_person.to(device)

        with torch.no_grad():
            netarc_embeds = netArc(F.interpolate(Xs_orig, [112, 112], mode='bilinear', align_corners=False))

        # generator training
        opt_G.zero_grad()
        
        #with autocast():
        if verbose_output:
            print(Xt.shape)
            print(netarc_embeds.shape)
        
        Y, Xt_attr = G(Xt, netarc_embeds)

        tY, _ = teacher(Xt.half(), netarc_embeds.half())
        tY = tY.float()
        with torch.no_grad():
            tY_resized_112 = F.interpolate(tY, [112, 112], mode='bilinear', align_corners=False)
            ZtY = netArc(tY_resized_112)
        netarc_embeds_loss_from_hq = (1 - torch.cosine_similarity(netarc_embeds, ZtY, dim=1)).mean()
        
        Y_attr = G.get_attr(Y)
        L_attr = 0
        for i in range(len(Xt_attr)):
            L_attr += torch.mean(torch.pow(Xt_attr[i] - Y_attr[i], 2).reshape(Xs.size(0), -1), dim=1).mean()
        L_attr /= 2.0

         # adversarial loss
        if D:
            diff_person = torch.ones_like(same_person)
            Di = D(Y)
            L_adv = 0.0
            for di in Di:
                L_adv += torch.relu(1-di[0]).mean(dim=[1, 2, 3])
            L_adv = torch.sum(L_adv * diff_person) / (diff_person.sum() + 1e-4)
     
        with torch.no_grad():
            ZY = netArc(F.interpolate(Y, [112, 112], mode='bilinear', align_corners=False))
            tZY = netArc(F.interpolate(tY, [112, 112], mode='bilinear', align_corners=False))

        netarc_embeds_loss =(1 - torch.cosine_similarity(tZY, ZY, dim=1)).mean()
        teacher_loss = F.mse_loss(tY, Y)

        L_attr_multiplier = 3.5
        L_adv_multiplier = 1.5
        teacher_loss_multiplier = 10000.0

        netarc_embeds_loss_multiplier = 3.5
        #while universal_multiplier*netarc_embeds_loss_multiplier*netarc_embeds_loss.item() < 200:
        #    netarc_embeds_loss_multiplier = netarc_embeds_loss_multiplier * 1.1
        
        if args.without_teacher_loss == True:
            teacher_loss = torch.tensor(0.0).to(device)
            netarc_embeds_loss_from_hq = torch.tensor(0.0).to(device)

        if D:
            total_loss = universal_multiplier * (
                #netarc_embeds_loss_multiplier * netarc_embeds_loss + 
                teacher_loss_multiplier * teacher_loss# + 
                #L_attr_multiplier * L_attr
                + L_adv_multiplier * L_adv
                )
        else:
            total_loss = universal_multiplier * (
                netarc_embeds_loss_multiplier * netarc_embeds_loss + 
                teacher_loss_multiplier * teacher_loss + 
                L_attr_multiplier * L_attr)

        # Backward and optimize
        opt_G.zero_grad()
        total_loss.backward()
        opt_G.step()


        if D:
            diff_person = torch.ones_like(same_person)
            lossD = compute_discriminator_loss(D, Y, Xs, diff_person)

            lossD_multiplier = 0.5
            

            # Backward and optimize
            if iteration % 20 == 0:
                total_lossD = universal_multiplier * lossD_multiplier * lossD
                opt_D.zero_grad()
                total_lossD.backward()
                opt_D.step()
                total_lossD = torch.tensor(0.0).to(device)
            else:
                total_lossD = total_lossD + (universal_multiplier * lossD_multiplier * lossD)


        # Progress Report
        batch_time = time.time() - start_time

        if iteration % args.show_step == 0:
            images = [Xs, Xt, tY, Y]
            image = make_image_list(images, normalize=False)
            os.makedirs('./output/images/', exist_ok=True)
            cv2.imwrite(f'./output/images/generated_image_{args.run_name}_{str(epoch)}_{iteration:06}.jpg', image[:,:,::-1])

        if iteration % 10 == 0:
            print(f'epoch:                      {epoch}    {iteration} / {len(dataloader)}')
            print(f'netarc_embeds_loss:         {universal_multiplier*netarc_embeds_loss_multiplier*netarc_embeds_loss.item()}      (x{netarc_embeds_loss_multiplier})')
            print(f'L_attr:                     {universal_multiplier*L_attr_multiplier*L_attr.item()}')
            print(f'teacher_loss:               {universal_multiplier*teacher_loss_multiplier*teacher_loss.item()}        (x{teacher_loss_multiplier})')
            if D:
                print(f'L_adv:                      {universal_multiplier*L_adv_multiplier*L_adv.item()}')
                print(f'lossD:                      {total_lossD.item()}')
            #print(f'round_trip_loss:            {universal_multiplier*round_trip_loss_multiplier*round_trip_loss.item()}    (x{round_trip_loss_multiplier})')
            print(f'total_loss:                 {total_loss.item()} batch_time: {batch_time}s')
            
            if args.scheduler:
                print(f'scheduler_G lr: {scheduler_G.get_last_lr()}')

        #if iteration % 2500 == 0:
        if iteration % args.save_interval == 0:
            os.makedirs(f'./output/saved_models_{args.run_name}/', exist_ok=True)
            os.makedirs(f'./output/current_models_{args.run_name}/', exist_ok=True)
            torch.save(G.state_dict(), f'./output/saved_models_{args.run_name}/G_latest.pth')
            torch.save(G.state_dict(), f'./output/current_models_{args.run_name}/G_' + str(epoch)+ '_' + f"{iteration:06}" + '.pth')
            if D:
                torch.save(D.state_dict(), f'./output/saved_models_{args.run_name}/D_latest.pth')
                torch.save(D.state_dict(), f'./output/current_models_{args.run_name}/D_' + str(epoch)+ '_' + f"{iteration:06}" + '.pth')
        
        torch.cuda.empty_cache()


def train(args, device):
    # training params
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    
    # initializing main models
    G = AEI_Net(args.backbone, num_blocks=args.num_blocks, c_id=512).to(device)
    G.train()
    
    D = MultiscaleDiscriminator(input_nc=3, n_layers=5, norm_layer=torch.nn.InstanceNorm2d).to(device)
    D.train()
    opt_D = optim.Adam(D.parameters(), lr=args.lr_D, betas=(0, 0.999), weight_decay=1e-4)
    if args.scheduler:
        scheduler_D = scheduler.StepLR(opt_D, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    else:
        scheduler_D = None

    if args.D_path:    
        D.load_state_dict(torch.load(args.D_path, map_location=torch.device('cpu')), strict=False)

    teacher = AEI_Net(args.backbone, num_blocks=args.num_blocks, c_id=512).to(device)
    teacher.train()
    teacher.half()
    
    if args.teacher_path:
        try:
            teacher.load_state_dict(torch.load(args.teacher_path, map_location=torch.device('cpu')), strict=False)
            print("Loaded teacher weights")
        except FileNotFoundError as e:
            print("Not found teacher weights. Teacher must have weights via --teacher_path")
            return

    # initializing model for identity extraction
    netArc = iresnet100(fp16=False)
    netArc.load_state_dict(torch.load('arcface_model/backbone.pth'))
    netArc=netArc.cuda()
    netArc.eval()
        
    if args.eye_detector_loss:
        model_ft = models.FAN(4, "False", "False", 98)
        checkpoint = torch.load('./AdaptiveWingLoss/AWL_detector/WFLW_4HG.pth')
        if 'state_dict' not in checkpoint:
            model_ft.load_state_dict(checkpoint)
        else:
            pretrained_weights = checkpoint['state_dict']
            model_weights = model_ft.state_dict()
            pretrained_weights = {k: v for k, v in pretrained_weights.items() \
                                  if k in model_weights}
            model_weights.update(pretrained_weights)
            model_ft.load_state_dict(model_weights)
        model_ft = model_ft.to(device)
        model_ft.eval()
    else:
        model_ft=None

    opt_G = optim.Adam(G.parameters(), lr=args.lr_G, betas=(0, 0.999), weight_decay=1e-4)
    
    if args.scheduler:
        scheduler_G = scheduler.StepLR(opt_G, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    else:
        scheduler_G = None
        
    if args.pretrained:
        try:
            G.load_state_dict(torch.load(args.G_path, map_location=torch.device('cpu')), strict=False)
            print("Loaded pretrained weights for G")
        except FileNotFoundError as e:
            print("Not found pretrained weights. Continue without any pretrained weights.")
    
    if args.celeb:
        dataset = CelebADataset(args.dataset_path, args.normalize_training_images, args.fine_tune_filter, only_attractive=args.only_attractive, into_data_path=args.into_data_path)
    elif args.vgg:
        dataset = FaceEmbedVGG2(args.dataset_path, same_prob=args.same_person, same_identity=args.same_identity)
    else:
        dataset = FaceEmbed([args.dataset_path], same_prob=args.same_person)
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    loss_adv_accumulated = 20.
    
    for epoch in range(0, max_epoch):
        train_one_epoch(G,
                        opt_G,
                        scheduler_G,
                        D,
                        opt_D,
                        scheduler_D,
                        netArc,
                        teacher,
                        model_ft,
                        args,
                        dataloader,
                        device,
                        epoch,
                        loss_adv_accumulated)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print('cuda is not available. using cpu. check if it\'s ok')
    
    print("Starting training")
    train(args, device=device)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # dataset params
    parser.add_argument('--dataset_path', default='/VggFace2-crop/', help='Path to the dataset. If not VGG2 dataset is used, param --vgg should be set False')
    parser.add_argument('--G_path', default='./saved_models/G.pth', help='Path to pretrained weights for G. Only used if pretrained=True')
    parser.add_argument('--D_path', default=None, help='Path to pretrained weights for D. Only used if pretrained=True')
    parser.add_argument('--vgg', default=True, type=bool, help='When using VGG2 dataset (or any other dataset with several photos for one identity)')
    parser.add_argument('--celeb', default=False, type=bool, help='When using VGG2 dataset (or any other dataset with several photos for one identity)')
    # weights for loss
    parser.add_argument('--weight_adv', default=1, type=float, help='Adversarial Loss weight')
    parser.add_argument('--weight_attr', default=10, type=float, help='Attributes weight')
    parser.add_argument('--weight_id', default=20, type=float, help='Identity Loss weight')
    parser.add_argument('--weight_rec', default=10, type=float, help='Reconstruction Loss weight')
    parser.add_argument('--weight_eyes', default=0., type=float, help='Eyes Loss weight')
    # training params you may want to change
    parser.add_argument('--backbone', default='unet', const='unet', nargs='?', choices=['unet', 'linknet', 'resnet'], help='Backbone for attribute encoder')
    parser.add_argument('--num_blocks', default=2, type=int, help='Numbers of AddBlocks at AddResblock')
    parser.add_argument('--same_person', default=0.05, type=float, help='Probability of using same person identity during training')
    parser.add_argument('--same_identity', default=True, type=bool, help='Using simswap approach, when source_id = target_id. Only possible with vgg=True')
    parser.add_argument('--diff_eq_same', default=False, type=bool, help='Don\'t use info about where is defferent identities')
    parser.add_argument('--pretrained', default=True, type=bool, help='If using the pretrained weights for training or not')
    parser.add_argument('--scheduler', default=False, type=bool, help='If True decreasing LR is used for learning of generator')
    parser.add_argument('--scheduler_step', default=5000, type=int)
    parser.add_argument('--scheduler_gamma', default=0.2, type=float, help='It is value, which shows how many times to decrease LR')
    parser.add_argument('--eye_detector_loss', default=True, type=bool, help='If True eye loss with using AdaptiveWingLoss detector is applied to generator')
    # info about this run
    parser.add_argument('--run_name', required=True, type=str, help='Name of this run. Used to create folders where to save the weights.')
    # training params you probably don't want to change
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr_G', default=4e-4, type=float)
    parser.add_argument('--lr_D', default=4e-4, type=float)
    parser.add_argument('--max_epoch', default=2000, type=int)
    parser.add_argument('--show_step', default=500, type=int)
    parser.add_argument('--save_epoch', default=1, type=int)
    parser.add_argument('--optim_level', default='O2', type=str)
    
    #Extra
    parser.add_argument('--save_interval', default=2500, type=int)
    parser.add_argument('--teacher_fine_tune', default=False, type=bool)
    parser.add_argument('--teacher_inner_crop', default=False, type=bool)
    parser.add_argument('--only_attractive', default=False, type=bool)
    parser.add_argument('--normalize_training_images', default=False, type=bool)
    parser.add_argument('--fine_tune_filter', default=None, type=str)
    parser.add_argument('--into_data_path', default=None, type=str)
    parser.add_argument('--without_teacher_loss', default=False, type=bool)
    parser.add_argument('--teacher_path', default=None, help='Path to pretrained weights for Teacher Model')
    parser.add_argument('--verbose_output', default=False, type=bool, help='More print() when training')

    args = parser.parse_args()
    
    if args.vgg==False and args.same_identity==True:
        raise ValueError("Sorry, you can't use some other dataset than VGG2 Faces with param same_identity=True")
    
    if not os.path.exists('./images'):
        os.mkdir('./images')
    
    if not os.path.exists(f'./saved_models_{args.run_name}'):
        os.mkdir(f'./saved_models_{args.run_name}')
        os.mkdir(f'./current_models_{args.run_name}')
    
    main(args)


