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
# custom imports

from network.AEI_Net import *
from utils.training.Dataset import FaceEmbedVGG2, FaceEmbed, CelebADataset
from utils.training.image_processing import make_image_list, get_faceswap
from arcface_model.iresnet import iresnet100

import insightface
import onnxruntime
from onnx import numpy_helper
import onnx
from insightface.utils import face_align

import random
import math

print("finished imports")

print("started globals")

border_size = 100
print("finished globals")

def create_soft_mask_batch(hull_points_batch, H, W):
    B, N, _ = hull_points_batch.shape
    device = hull_points_batch.device

    # Create a grid of the same size as the image
    xs = torch.linspace(0, W - 1, steps=W, device=device)
    ys = torch.linspace(0, H - 1, steps=H, device=device)
    grid = torch.meshgrid(ys, xs)  # This creates a grid for one image
    grid = torch.stack(grid, dim=-1).unsqueeze(0).to(device)  # Add batch dimension and send to device

    # Expand the grid to match the batch size
    grid_batch = grid.repeat(B, 1, 1, 1)  # Now grid_batch is of shape (B, H, W, 2)

    # Calculate distance from each grid point to the nearest hull point for each image in the batch
    distances = torch.cdist(grid_batch.view(B, -1, 2), hull_points_batch).min(dim=-1).values.view(B, H, W)

    # Convert distances to a soft mask using a sigmoid or other function
    sigma = 10  # Controls how soft the mask edges are
    soft_mask_batch = torch.zeros_like(distances)

    for i in range(B):
        mean_distance = distances[i].mean()
        soft_mask_batch[i] = torch.sigmoid(-sigma * (distances[i] - mean_distance))

    # Add the channel dimension to the mask, resulting in shape (B, 1, H, W)
    soft_mask_batch_unsqueezed = soft_mask_batch.unsqueeze(1)

    return soft_mask_batch_unsqueezed

def calculate_convex_hulls(landmarks_batch):
    hulls = []
    for landmarks in landmarks_batch:
        # Convert landmarks to a format suitable for cv2.convexHull
        landmarks_np = landmarks.cpu().numpy()
        hull = cv2.convexHull(landmarks_np).squeeze()
        hulls.append(hull)
    return hulls

def pad_hulls(hulls, pad_value=0):
    # Find the largest number of points in any hull
    max_points = max(hull.shape[0] for hull in hulls)

    # Pad each hull to have max_points
    padded_hulls = [np.pad(hull, ((0, max_points - hull.shape[0]), (0, 0)), 
                           mode='constant', constant_values=pad_value) for hull in hulls]
    return padded_hulls

def calculate_sine_weight(iteration, amplitude = 0.5, frequency = 0.01, phase_shift = math.pi / 2, base_weight=0.5):
    """
    Calculate a weight that oscillates between 0 and 1 using a sine wave.

    :param iteration: Current iteration or epoch number.
    :param amplitude: Amplitude of the sine wave.
    :param frequency: Frequency of the sine wave.
    :param phase_shift: Phase shift of the sine wave.
    :param base_weight: Base weight around which the oscillation occurs.
    :return: A weight between 0 and 1.
    """
    return base_weight + amplitude * math.sin(frequency * iteration + phase_shift)

def get_hsv(im, eps=1e-7):
    img = im * 0.5 + 0.5
    hue = torch.Tensor(im.shape[0], im.shape[2], im.shape[3]).to(im.device)

    hue[ img[:,2]==img.max(1)[0] ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,2]==img.max(1)[0] ]
    hue[ img[:,1]==img.max(1)[0] ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,1]==img.max(1)[0] ]
    hue[ img[:,0]==img.max(1)[0] ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,0]==img.max(1)[0] ]) % 6

    hue[img.min(1)[0]==img.max(1)[0]] = 0.0
    hue = hue/6

    saturation = ( img.max(1)[0] - img.min(1)[0] ) / ( img.max(1)[0] + eps )
    saturation[ img.max(1)[0]==0 ] = 0

    value = img.max(1)[0]

    return torch.stack([hue, saturation, value], dim=1)

def stuck_loss_func(output, region, region_penalty_weight=0.5):
    x, y, h, w = region


    x, y, h, w = region
    region_output = output[:, :, y:y+h, x:x+w]

    gradient_x = torch.abs(region_output[:, :, :, 1:] - region_output[:, :, :, :-1])
    gradient_y = torch.abs(region_output[:, :, 1:, :] - region_output[:, :, :-1, :])

    # Ensure gradients are the same size
    min_width = min(gradient_x.size(3), gradient_y.size(3))
    min_height = min(gradient_x.size(2), gradient_y.size(2))
    gradient_x = gradient_x[:, :, :min_height, :min_width]
    gradient_y = gradient_y[:, :, :min_height, :min_width]

    # Penalize large gradients (sharp color changes)
    smoothness_loss = torch.mean(gradient_x * gradient_x + gradient_y * gradient_y)


    return region_penalty_weight * smoothness_loss


def train_one_epoch(G: 'generator model', 
                        opt_G: "generator opt", 
                        scheduler_G: "scheduler G opt",
                        netArc: 'ArcFace model',
                        teacher: 'Teacher model',
                        args: 'Args Namespace',
                        dataloader: torch.utils.data.DataLoader,
                        device: 'torch device',
                        epoch:int,
                        loss_adv_accumulated:int):
    
    verbose_output = args.verbose_output
    
    for iteration, data in enumerate(dataloader):
        start_time = time.time()
        
        Xs_orig, Xs, Xt, _same_person = data
        
        Xs_orig = Xs_orig.to(device)
        Xs = Xs.to(device)
        Xt = Xt.to(device)

        with torch.no_grad():
            netarc_embeds = netArc(F.interpolate(Xs_orig, [112, 112], mode='bilinear', align_corners=False))


        # generator training
        opt_G.zero_grad()
        
        #with autocast():
        if verbose_output:
            print(Xt.shape)
            print(netarc_embeds.shape)
        
        #Y, Xt_attr = G(Xt, embeds)
        Y, Xt_attr = G(Xt, netarc_embeds)

        tY, _ = teacher(Xt.half(), netarc_embeds.half())

        tY = tY.float()
        
        ZY = netArc(F.interpolate(Y, [112, 112], mode='bilinear', align_corners=False))
        tZY = netArc(F.interpolate(tY, [112, 112], mode='bilinear', align_corners=False))
        netarc_embeds_loss =(1 - torch.cosine_similarity(tZY, ZY, dim=1)).mean()
        
        if args.teacher_inner_crop == True:
            #Crops the inner 56x56 which is the part of the face we care most about
            crop_start = 64
            crop_end = 192
            Xt_cropped = Xt[:, :, crop_start:crop_end, crop_start:crop_end]
            tY_cropped = tY[:, :, crop_start:crop_end, crop_start:crop_end]
            Y_cropped = Y[:, :, crop_start:crop_end, crop_start:crop_end]
            cropped_loss = torch.norm((Xt_cropped - tY_cropped) - (Xt_cropped - Y_cropped), p=2)
            teacher_loss = 10*cropped_loss + torch.norm((Xt - tY) - (Xt - Y), p=2)
        else:
            #Not cropped
            teacher_loss = torch.norm((Xt - tY) - (Xt - Y), p=2)

        Y_attr = G.get_attr(Y)
        L_attr = 0
        for i in range(len(Xt_attr)):
            L_attr += torch.mean(torch.pow(Xt_attr[i] - Y_attr[i], 2).reshape(Xs_orig.size(0), -1), dim=1).mean()
        L_attr /= 2.0

        Y_hsv = get_hsv(Y)
        Xt_hsv = get_hsv(Xt)
        hsv_loss = torch.mean(torch.abs(Xt_hsv - Y_hsv))

        universal_multiplier = 1
        netarc_embeds_loss_miltiplier = 3.5
        L_attr_miltiplier = 2.0

        if teacher_loss.item() > 50:
            teacher_loss_miltiplier = 3.0
        else:
            teacher_loss_miltiplier = 0.5
        
        hsv_loss_multiplier = 10.0

        if args.teacher_fine_tune == False:    
            total_loss = universal_multiplier * ( 
                            netarc_embeds_loss_miltiplier * netarc_embeds_loss + 
                            L_attr_miltiplier * L_attr + 
                            hsv_loss_multiplier + hsv_loss +
                            teacher_loss_miltiplier * teacher_loss 
                        )
        else:
            teacher_loss_miltiplier = teacher_loss_miltiplier * 1000
            total_loss = teacher_loss_miltiplier * teacher_loss

        # Backward and optimize
        opt_G.zero_grad()
        total_loss.backward()
        opt_G.step()


        # Progress Report
        batch_time = time.time() - start_time

        if iteration % args.show_step == 0:
            images = [Xs, Xt, tY, Y]
            image = make_image_list(images, normalize=False)
            os.makedirs('./output/images/', exist_ok=True)
            cv2.imwrite(f'./output/images/generated_image_{args.run_name}_{str(epoch)}_{iteration:06}.jpg', image[:,:,::-1])

        if iteration % 10 == 0:
            print(f'epoch:                      {epoch}    {iteration} / {len(dataloader)}')
            print(f'netarc_embeds_loss:         {universal_multiplier*netarc_embeds_loss_miltiplier*netarc_embeds_loss.item()}')
            print(f'L_attr:                     {universal_multiplier*L_attr_miltiplier*L_attr.item()}')
            print(f'hsv_loss:                   {universal_multiplier*hsv_loss_multiplier*hsv_loss.item()}')
            print(f'teacher_loss:               {universal_multiplier*teacher_loss_miltiplier*teacher_loss.item()}')
            print(f'total_loss:                 {total_loss.item()} batch_time: {batch_time}s')
            
            if args.scheduler:
                print(f'scheduler_G lr: {scheduler_G.get_last_lr()}')

        #if iteration % 2500 == 0:
        if iteration % args.save_interval == 0:
            os.makedirs(f'./output/saved_models_{args.run_name}/', exist_ok=True)
            os.makedirs(f'./output/current_models_{args.run_name}/', exist_ok=True)
            torch.save(G.state_dict(), f'./output/saved_models_{args.run_name}/G_latest.pth')
            torch.save(G.state_dict(), f'./output/current_models_{args.run_name}/G_' + str(epoch)+ '_' + f"{iteration:06}" + '.pth')

        
        torch.cuda.empty_cache()


def train(args, device):
    # training params
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    
    # initializing main models
    G = AEI_Net(args.backbone, num_blocks=args.num_blocks, c_id=512).to(device)
    G.train()

    teacher = AEI_Net(args.backbone, num_blocks=args.num_blocks, c_id=512).to(device)
    teacher.train()
    teacher.half()
    
    # initializing model for identity extraction
    netArc = iresnet100(fp16=False)
    netArc.load_state_dict(torch.load('arcface_model/backbone.pth'))
    netArc=netArc.cuda()
    netArc.eval()
        
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

    if args.teacher_path:
        try:
            teacher.load_state_dict(torch.load(args.teacher_path, map_location=torch.device('cpu')), strict=False)
            print("Loaded teacher weights")
        except FileNotFoundError as e:
            print("Not found teacher weights. Teacher must have weights via --teacher_path")
            return
    
    if args.celeb:
        dataset = CelebADataset(args.dataset_path, args.normalize_training_images)        
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
                        netArc,
                        teacher,
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
    parser.add_argument('--eye_detector_loss', default=False, type=bool, help='If True eye loss with using AdaptiveWingLoss detector is applied to generator')
    # info about this run
    parser.add_argument('--run_name', required=True, type=str, help='Name of this run. Used to create folders where to save the weights.')
    # training params you probably don't want to change
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr_G', default=4e-4, type=float)
    parser.add_argument('--max_epoch', default=2000, type=int)
    parser.add_argument('--show_step', default=500, type=int)
    parser.add_argument('--save_epoch', default=1, type=int)
    parser.add_argument('--optim_level', default='O2', type=str)
    
    #Extra
    parser.add_argument('--save_interval', default=2500, type=int)
    parser.add_argument('--teacher_fine_tune', default=False, type=bool)
    parser.add_argument('--teacher_inner_crop', default=False, type=bool)
    parser.add_argument('--normalize_training_images', default=False, type=bool)
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


