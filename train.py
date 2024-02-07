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

print("finished imports")

print("started globals")
l2_loss = torch.nn.MSELoss()
face_analyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_analyser.prepare(ctx_id=0, det_size=(640, 640))
high_quality_teacher_model_file = '/root/.insightface/models/inswapper_128.onnx'
model = onnx.load(high_quality_teacher_model_file)
high_quality_teacher_session = onnxruntime.InferenceSession(high_quality_teacher_model_file, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
#low_quality_teacher_session = onnxruntime.InferenceSession(model_file, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
emap = numpy_helper.to_array(model.graph.initializer[-1])

# Get input and output names
high_quality_output_names = high_quality_teacher_session.get_outputs()[0].name
#low_quality_output_names = low_quality_teacher_session.get_outputs()[0].name

high_quality_input_names = [input.name for input in high_quality_teacher_session.get_inputs()]
#low_quality_input_names = [input.name for input in low_quality_teacher_session.get_inputs()]

scaler = GradScaler()

border_size = 100
print("finished globals")

def train_one_epoch(G: 'generator model', 
                    opt_G: "generator opt", 
                    scheduler_G: "scheduler G opt",
                    D: 'discriminator model',
                    opt_D: "generator opt", 
                    scheduler_D: "scheduler G opt",
                    netArc: 'ArcFace model',
                    model_ft: 'Landmark Detector',
                    args: 'Args Namespace',
                    dataloader: torch.utils.data.DataLoader,
                    device: 'torch device',
                    epoch:int,
                    loss_adv_accumulated:int):
    
    verbose_output = args.verbose_output
    total_lossD = torch.tensor(0.0).to(device)
    for iteration, data in enumerate(dataloader):
        start_time = time.time()
        
        _Xs_orig, _Xs, _Xt_raw, _Xt, _same_person = data
        
        _Xs_orig_on_device = _Xs_orig.to(device)

        Xs_orig = []
        Xs = [] 
        Xt = [] 
        same_person = []
        embeds = []
        high_quality_results = []
        high_quality_results_netarc = []
        high_quality_results_lmks = []
        can_run = False
        with torch.no_grad():
            embed_old_way = netArc(F.interpolate(_Xs_orig_on_device, [112, 112], mode='area'))
            netarc_embeds = []
            for i, _  in enumerate(_Xs_orig):
                blob_128 = None
                blob_256 = None
                embed = None
                try:
                    img = F.interpolate(_Xs_orig[i:i+1], size=(128, 128), mode='area')
                    img = img.detach().cpu().numpy()
                    img = img.transpose((0,2,3,1))[0]
                    img = np.clip(255 * img, 0, 255).astype(np.uint8)[:,:,::-1]
                    img = cv2.copyMakeBorder(img, top=border_size, bottom=border_size, left=border_size, right=border_size, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    face = face_analyser.get(img)[0]
                    embed = face.normed_embedding.reshape((1,-1))
                    embed = np.dot(embed, emap)
                    embed /= np.linalg.norm(embed)
                except Exception as e:
                    if verbose_output:
                        print(f'Embed Failed: Skipped {str(epoch)}:{iteration:06}:{i}')
                        print(e)
                    continue

                try:
                    img = F.interpolate(_Xt_raw[i:i+1], size=(128, 128), mode='area')
                    img = img.detach().cpu().numpy()
                    img = img.transpose((0,2,3,1))[0]
                    img = np.clip(255 * img, 0, 255).astype(np.uint8)[:,:,::-1]
                    warped_img_128 = cv2.resize(img, (128, 128))
                    blob_128 = warped_img_128.astype('float32') / 255.0
                    blob_128 = blob_128[:, :, [2, 1, 0]]
                    blob_128 = blob_128.transpose((2,0,1))
                    blob_128 = np.expand_dims(blob_128, axis=0)
                    if verbose_output:
                        img_fake = blob_128.transpose((0,2,3,1))[0]
                        img_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:,:,::-1]
                        Image.fromarray(cv2.cvtColor(img_fake,cv2.COLOR_RGBA2BGR)).save(f'./output/images/blob_128_{i}.jpg')
                    blob_256 = _Xt[i:i+1]
                except Exception as e:
                    if verbose_output:
                        print(f'Blobbing Failed: Skipped {str(epoch)}:{iteration:06}:{i}')
                        print(e)
                    continue

                try:
                    high_quality_pred = high_quality_teacher_session.run(
                        [high_quality_output_names],
                        {
                            high_quality_input_names[0]: blob_128, 
                            high_quality_input_names[1]: embed
                        }
                    )[0]
                    img_fake = high_quality_pred.transpose((0,2,3,1))[0]
                    bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:,:,::-1]
                    if verbose_output:
                        Image.fromarray(cv2.cvtColor(bgr_fake,cv2.COLOR_RGBA2BGR)).save(f'./output/images/generated_image_HQ_OUT_{i}.jpg')
                    bgr_fake = cv2.copyMakeBorder(bgr_fake, top=border_size, bottom=border_size, left=border_size, right=border_size, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    face = face_analyser.get(bgr_fake)[0]
                    high_quality_lmks_tensor = torch.from_numpy(face.landmark_2d_106).unsqueeze(0).to(device)
                    high_quality_results_lmks.append(high_quality_lmks_tensor)

                    high_quality_pred_tensor = torch.from_numpy(high_quality_pred).to(device)
                    HQ_Resized = F.interpolate(high_quality_pred_tensor, [112, 112], mode='area')
                    PRED = netArc(HQ_Resized)
                    high_quality_results_netarc.append(PRED)
                    high_quality_results.append(high_quality_pred_tensor)
                except Exception as e:
                    if verbose_output:
                        print(f'Teacher Failed: Skipped {str(epoch)}:{iteration:06}:{i}')
                        print(e)
                    continue
                
                embeds.append(embed)
                Xt.append(blob_256)
                Xs_orig.append(_Xs_orig[i:i+1])
                Xs.append(_Xs[i:i+1])
                same_person.append(_same_person[i:i+1])
                netarc_embeds.append(embed_old_way[i:i+1])
                can_run = True
                
            
        if not can_run:
            if verbose_output:
                print("Nothing to Process This Iteration!")
            continue

        Xs_orig = torch.cat(Xs_orig , dim=0).to(device)
        Xs = torch.cat(Xs , dim=0).to(device)
        Xt = torch.cat([torch.tensor(x) for x in Xt] , dim=0).to(device)
        same_person = torch.cat(same_person, dim=0).to(device)
        embeds = torch.cat([torch.tensor(x) for x in embeds] , dim=0).to(device)
        netarc_embeds = torch.cat(netarc_embeds, dim=0).to(device)
        Xt_embeds = netArc(F.interpolate(Xt, [112, 112], mode='area'))

        # generator training
        opt_G.zero_grad()
        
        #with autocast():
        if verbose_output:
            print(_Xt.shape)
            print(Xt.shape)
            print(embed_old_way.shape)
            print(embeds.shape)
        
        #Y, Xt_attr = G(Xt, embeds)
        Y, Xt_attr = G(Xt, netarc_embeds)

        #The model should handle inputs that are simply mirrored identically
        Xt_mirrored = torch.flip(Xt, dims=[3])
        Y_mirrored, Xt_attr_mirrored = G(Xt_mirrored, netarc_embeds)
        Y_unmirrored = torch.flip(Y_mirrored, dims=[3])
        mirror_loss = F.mse_loss(Y, Y_unmirrored)

        #The model should be able to take the outputs and get back to the original inputs
        Y_round_trip, Xt_attr_round_trip = G(Y, Xt_embeds)
        round_trip_loss = F.mse_loss(Xt, Y_round_trip)

        # adversarial loss
        if D:
            diff_person = torch.ones_like(same_person)
            Di = D(Y)
            L_adv = 0.0
            for di in Di:
                L_adv += torch.relu(1-di[0]).mean(dim=[1, 2, 3])
            L_adv = torch.sum(L_adv * diff_person) / (diff_person.sum() + 1e-4)

        high_quality_results_netarc_combined = torch.cat(high_quality_results_netarc, dim=0)
        high_quality_results_combined = torch.cat(high_quality_results, dim=0)
        high_quality_results_lmks_combined = torch.cat(high_quality_results_lmks, dim=0)


        Y_resized = F.interpolate(Xt, size=(128, 128), mode='area')
        Y_resized_112 = F.interpolate(Y, [112, 112], mode='area')
        ZY = netArc(Y_resized_112)

        if args.eye_detector_loss:
            high_quality_results_resized = F.interpolate(high_quality_results_combined, size=(256, 256), mode='bicubic', align_corners=False)
            Xt_eyes, Xt_heatmap_left, Xt_heatmap_right = detect_landmarks(high_quality_results_resized, model_ft)
            Y_eyes, Y_heatmap_left, Y_heatmap_right = detect_landmarks(Y, model_ft)
            eye_heatmaps = [Xt_heatmap_left, Xt_heatmap_right, Y_heatmap_left, Y_heatmap_right]
            L_l2_eyes = compute_eye_loss(eye_heatmaps)
        else:
            L_l2_eyes = torch.tensor(0.0).to(device)

        try:
            y_lmks = []
            #landmark_2d_106 need to be calculated at 128x128, only because that is how they were calculated with the teacher model
            Y_resized_128 = F.interpolate(Y, [128, 128], mode='area')
            for i, image in enumerate(Y_resized_128):
                img_fake = Y_resized_128.detach().cpu().numpy().transpose((0,2,3,1))[i]
                bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:,:,::-1]
                if verbose_output:
                    Image.fromarray(cv2.cvtColor(bgr_fake,cv2.COLOR_RGBA2BGR)).save(f'./output/images/Y_resized_{i}.jpg')
                bgr_fake = cv2.copyMakeBorder(bgr_fake, top=border_size, bottom=border_size, left=border_size, right=border_size, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
                face = face_analyser.get(bgr_fake)[0]
                y_lmks_tensor = torch.from_numpy(face.landmark_2d_106).unsqueeze(0).to(device)
                y_lmks.append(y_lmks_tensor)
            y_lmks_combined = torch.cat(y_lmks, dim=0)
        except Exception as e:
            # Create a tensor of zeros with the same shape as high_quality_results_lmks_combined
            y_lmks_combined = torch.zeros_like(high_quality_results_lmks_combined)
            if verbose_output:
                print(e)


        #high_quality_pred = torch.from_numpy(high_quality_pred).to(device)
        netarc_embeds_loss_from_hq =(1 - torch.cosine_similarity(high_quality_results_netarc_combined, ZY, dim=1)).mean()
        #teacher_loss = l2_loss(soft_masks*Y_resized, soft_masks*high_quality_results_combined)


        Xt_resized = F.interpolate(Xt, [128, 128], mode='area')
        if args.teacher_inner_crop == True:
            #Crops the inner 56x56 which is the part of the face we care most about
            crop_start = 28
            crop_end = 84
            Xt_cropped = Xt_resized[:, :, crop_start:crop_end, crop_start:crop_end]
            high_quality_cropped = high_quality_results_combined[:, :, crop_start:crop_end, crop_start:crop_end]
            Y_cropped = Y_resized[:, :, crop_start:crop_end, crop_start:crop_end]
            teacher_loss = torch.norm((Xt_cropped - high_quality_cropped) - (Xt_cropped - Y_cropped), p=2)
        else:
            #Not cropped
            teacher_loss = torch.norm((Xt_resized - high_quality_results_combined) - (Xt_resized - Y_resized), p=2)
            #teacher_loss = structural_loss(high_quality_results_combined, Y_resized)

        Y_attr = G.get_attr(Y)
        L_attr = 0
        for i in range(len(Xt_attr)):
            L_attr += torch.mean(torch.pow(Xt_attr[i] - Y_attr[i], 2).reshape(Xs_orig.size(0), -1), dim=1).mean()
        L_attr /= 2.0
        

        netarc_embeds_loss =(1 - torch.cosine_similarity(netarc_embeds, ZY, dim=1)).mean()
        
        universal_multiplier = 100

        L_attr_multiplier = 3.0
        L_adv_multiplier = 0.5
        teacher_loss_multiplier = 30.0 #25.0 #20.0

        round_trip_loss_multiplier = 1.0
        while universal_multiplier*round_trip_loss_multiplier*round_trip_loss.item() < 100:
            round_trip_loss_multiplier = round_trip_loss_multiplier * 1.1

        mirror_loss_multiplier = 1.0
        while universal_multiplier*mirror_loss_multiplier*mirror_loss.item() < 10:
            mirror_loss_multiplier = mirror_loss_multiplier * 1.1
            
        netarc_embeds_loss_from_hq_multiplier = 3.0
        while universal_multiplier*netarc_embeds_loss_from_hq_multiplier*netarc_embeds_loss_from_hq.item() < 250:
            netarc_embeds_loss_from_hq_multiplier = netarc_embeds_loss_from_hq_multiplier * 1.1

        netarc_embeds_loss_multiplier = 3.5
        while universal_multiplier*netarc_embeds_loss_multiplier*netarc_embeds_loss.item() < 200:
            netarc_embeds_loss_multiplier = netarc_embeds_loss_multiplier * 1.1
        
        L_l2_eyes_multiplier = 1.0
        while universal_multiplier*L_l2_eyes_multiplier*L_l2_eyes.item() < 100:
            L_l2_eyes_multiplier = L_l2_eyes_multiplier * 1.1
        

        if args.teacher_fine_tune == False:   
            if D:
                total_loss = universal_multiplier * ( 
                                netarc_embeds_loss_multiplier * netarc_embeds_loss + 
                                netarc_embeds_loss_from_hq_multiplier * netarc_embeds_loss_from_hq +
                                L_attr_multiplier * L_attr + 
                                teacher_loss_multiplier * teacher_loss +
                                L_adv_multiplier * L_adv  + 
                                L_l2_eyes_multiplier * L_l2_eyes +
                                mirror_loss_multiplier * mirror_loss +
                                round_trip_loss_multiplier * round_trip_loss
                            )
            else: 
                total_loss = universal_multiplier * ( 
                                netarc_embeds_loss_multiplier * netarc_embeds_loss + 
                                netarc_embeds_loss_from_hq_multiplier * netarc_embeds_loss_from_hq +
                                L_attr_multiplier * L_attr + 
                                teacher_loss_multiplier * teacher_loss +
                                L_l2_eyes_multiplier * L_l2_eyes +
                                mirror_loss_multiplier * mirror_loss + 
                                round_trip_loss_multiplier * round_trip_loss
                            )
        else:
            teacher_loss_multiplier = teacher_loss_multiplier * 1000
            total_loss = teacher_loss_multiplier * teacher_loss

        # Backward and optimize
        opt_G.zero_grad()
        total_loss.backward()
        opt_G.step()


        if D:
            #high_quality_results_combined = torch.cat(high_quality_results, dim=0)
            #high_quality_results_resized = F.interpolate(high_quality_results_combined, size=(256, 256), mode='bicubic', align_corners=False)

            diff_person = torch.ones_like(same_person)
            #lossD = compute_discriminator_loss(D, high_quality_results_resized, Xs, diff_person)
            lossD = compute_discriminator_loss(D, Y, Xs, diff_person)

            #universal_multiplier = 1.0
            lossD_multiplier = 0.5
            

            # Backward and optimize
            if iteration % 10 == 0:
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
            high_quality_results_resized = F.interpolate(high_quality_results_combined, size=(256, 256), mode='bicubic', align_corners=False)
            images = [Xs, Xt, high_quality_results_resized, Y]
            image = make_image_list(images, normalize=False)
            os.makedirs('./output/images/', exist_ok=True)
            cv2.imwrite(f'./output/images/generated_image_{args.run_name}_{str(epoch)}_{iteration:06}.jpg', image[:,:,::-1])

        if iteration % 10 == 0:
            print(f'epoch:                      {epoch}    {iteration} / {len(dataloader)}')
            #print(f'tiny_original_loss:         {universal_multiplier*tiny_original_loss_multiplier*tiny_original_loss.item()}')
            print(f'netarc_embeds_loss:         {universal_multiplier*netarc_embeds_loss_multiplier*netarc_embeds_loss.item()}      (x{netarc_embeds_loss_multiplier})')
            print(f'netarc_embeds_loss_from_hq: {universal_multiplier*netarc_embeds_loss_from_hq_multiplier*netarc_embeds_loss_from_hq.item()}      (x{netarc_embeds_loss_from_hq_multiplier})')
            #print(f'lmks_loss:                  {universal_multiplier*lmks_loss_multiplier*lmks_loss.item()}')
            print(f'L_attr:                     {universal_multiplier*L_attr_multiplier*L_attr.item()}')
            print(f'L_l2_eyes:                  {universal_multiplier*L_l2_eyes_multiplier*L_l2_eyes.item()}')
            print(f'mirror_loss:                {universal_multiplier*mirror_loss_multiplier*mirror_loss.item()}        (x{mirror_loss_multiplier})')
            print(f'round_trip_loss:            {universal_multiplier*round_trip_loss_multiplier*round_trip_loss.item()}        (x{round_trip_loss_multiplier})')
            #print(f'emboss_loss:                {universal_multiplier*emboss_loss_multiplier*emboss_loss.item()}')
            #print(f'hsv_loss:                   {universal_multiplier*hsv_loss_multiplier*hsv_loss.item()}')
            #print(f'edge_loss:                  {universal_multiplier*edge_loss_multiplier*edge_loss.item()}')
            print(f'teacher_loss:               {universal_multiplier*teacher_loss_multiplier*teacher_loss.item()}')
            #print(f'color_loss:                 {universal_multiplier*color_loss_multiplier*color_loss.item()}')
            if D:
                print(f'L_adv:                      {universal_multiplier*L_adv_multiplier*L_adv.item()}')
                print(f'lossD:                      {total_lossD.item()}')
            #print(f'stuck_loss:                 {stuck_loss.item()}')
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
        dataset = CelebADataset(args.dataset_path, args.normalize_training_images, args.fine_tune_filter)        
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
    parser.add_argument('--normalize_training_images', default=False, type=bool)
    parser.add_argument('--fine_tune_filter', default=None, type=str)
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


