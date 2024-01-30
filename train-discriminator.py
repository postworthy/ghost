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
from arcface_model.iresnet import iresnet100

import insightface
import onnxruntime
from onnx import numpy_helper
import onnx
from insightface.utils import face_align

from utils.training.helpers import get_hsv, stuck_loss_func, batch_edge_loss
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


def train_one_epoch(D: 'discriminator model',
                        opt_D: "generator opt", 
                        scheduler_D: "scheduler G opt",
                        netArc: 'ArcFace model',
                        args: 'Args Namespace',
                        dataloader: torch.utils.data.DataLoader,
                        device: 'torch device',
                        epoch:int,
                        loss_adv_accumulated:int):
    
    verbose_output = args.verbose_output
    
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
        can_run = False
        with torch.no_grad():
            for i, _  in enumerate(_Xs_orig):
                blob_128 = None
                blob_256 = None
                embed = None
                try:
                    img = F.interpolate(_Xs_orig[i:i+1], size=(128, 128), mode='bilinear', align_corners=False)
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
                    img = F.interpolate(_Xt_raw[i:i+1], size=(128, 128), mode='bilinear', align_corners=False)
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

                    high_quality_pred_tensor = torch.from_numpy(high_quality_pred).to(device)
                    HQ_Resized = F.interpolate(high_quality_pred_tensor, [112, 112], mode='bilinear', align_corners=False)
                    high_quality_results.append(HQ_Resized)
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

        # generator training
        opt_D.zero_grad()
        
        #with autocast():
        if verbose_output:
            print(_Xt.shape)
            print(Xt.shape)
            print(embeds.shape)
        
        high_quality_results_combined = torch.cat(high_quality_results, dim=0)
        high_quality_results_resized = F.interpolate(high_quality_results_combined, size=(256, 256), mode='bilinear', align_corners=False)

        diff_person = torch.ones_like(same_person)
        Y = high_quality_results_resized
        lossD = compute_discriminator_loss(D, Y, Xs, diff_person)

        universal_multiplier = 1.0
        lossD_multiplier = 1000.0
        total_lossD = universal_multiplier * lossD_multiplier * lossD
        # Backward and optimize
        opt_D.zero_grad()
        total_lossD.backward()
        opt_D.step()

        # Progress Report
        batch_time = time.time() - start_time

        if iteration % args.show_step == 0:
            images = [Xs, Xt, Y]
            image = make_image_list(images, normalize=False)
            os.makedirs('./output/images/', exist_ok=True)
            cv2.imwrite(f'./output/images/generated_image_{args.run_name}_{str(epoch)}_{iteration:06}.jpg', image[:,:,::-1])

        if iteration % 10 == 0:
            print(f'epoch:              {epoch}    {iteration} / {len(dataloader)}')
            print(f'total_loss:         {total_lossD.item()} batch_time: {batch_time}s')
            
            if args.scheduler:
                print(f'scheduler_G lr: {scheduler_D.get_last_lr()}')

        #if iteration % 2500 == 0:
        if iteration % args.save_interval == 0:
            os.makedirs(f'./output/saved_models_{args.run_name}/', exist_ok=True)
            os.makedirs(f'./output/current_models_{args.run_name}/', exist_ok=True)
            torch.save(D.state_dict(), f'./output/saved_models_{args.run_name}/D_latest.pth')
            torch.save(D.state_dict(), f'./output/current_models_{args.run_name}/D_' + str(epoch)+ '_' + f"{iteration:06}" + '.pth')

        
        torch.cuda.empty_cache()


def train(args, device):
    # training params
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    
    # initializing main models
    D = MultiscaleDiscriminator(input_nc=3, n_layers=5, norm_layer=torch.nn.InstanceNorm2d).to(device)
    D.train()
    
    # initializing model for identity extraction
    netArc = iresnet100(fp16=False)
    netArc.load_state_dict(torch.load('arcface_model/backbone.pth'))
    netArc=netArc.cuda()
    netArc.eval()
        
    opt_D = optim.Adam(D.parameters(), lr=args.lr_D, betas=(0, 0.999), weight_decay=1e-4)
    
    if args.scheduler:
        scheduler_D = scheduler.StepLR(opt_D, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    else:
        scheduler_D = None
        
    if args.pretrained:
        try:
            D.load_state_dict(torch.load(args.D_path, map_location=torch.device('cpu')), strict=False)
            print("Loaded pretrained weights for G")
        except FileNotFoundError as e:
            print("Not found pretrained weights. Continue without any pretrained weights.")
    
    if args.celeb:
        dataset = CelebADataset(args.dataset_path, args.normalize_training_images)        
    elif args.vgg:
        dataset = FaceEmbedVGG2(args.dataset_path, same_prob=args.same_person, same_identity=args.same_identity)
    else:
        dataset = FaceEmbed([args.dataset_path], same_prob=args.same_person)
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    loss_adv_accumulated = 20.
    
    for epoch in range(0, max_epoch):
        train_one_epoch(D,
                        opt_D,
                        scheduler_D,
                        netArc,
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
    parser.add_argument('--D_path', default='./saved_models/D.pth', help='Path to pretrained weights for D. Only used if pretrained=True')
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
    parser.add_argument('--lr_D', default=4e-4, type=float)
    parser.add_argument('--max_epoch', default=2000, type=int)
    parser.add_argument('--show_step', default=500, type=int)
    parser.add_argument('--save_epoch', default=1, type=int)
    parser.add_argument('--optim_level', default='O2', type=str)
    
    #Extra
    parser.add_argument('--save_interval', default=2500, type=int)
    parser.add_argument('--teacher_fine_tune', default=False, type=bool)
    parser.add_argument('--teacher_inner_crop', default=True, type=bool)
    parser.add_argument('--normalize_training_images', default=False, type=bool)
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


