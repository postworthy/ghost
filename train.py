print("started imports")

import sys
import argparse
import time
import cv2
from PIL import Image
import os

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as scheduler
from torch.cuda.amp import GradScaler, autocast
# custom imports

from network.AEI_Net import *
from utils.training.Dataset import FaceEmbedVGG2, FaceEmbed
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


def train_one_epoch_new(G: 'generator model', 
                        opt_G: "generator opt", 
                        scheduler_G: "scheduler G opt",
                        netArc: 'ArcFace model',
                        args: 'Args Namespace',
                        dataloader: torch.utils.data.DataLoader,
                        device: 'torch device',
                        epoch:int,
                        loss_adv_accumulated:int):
    
    verbose_output = args.verbose_output
    
    for iteration, data in enumerate(dataloader):
        start_time = time.time()
        
        _Xs_orig, _Xs, _Xt, _same_person = data
        
        _Xs_orig_on_device = _Xs_orig.to(device)

        Xs_orig = []
        Xs = [] 
        Xt = [] 
        same_person = []
        embeds = []
        high_quality_results = []
        high_quality_results_pred = []
        high_quality_results_lmks = []
        with torch.no_grad():
            embed_old_way = netArc(F.interpolate(_Xs_orig_on_device, [112, 112], mode='bilinear', align_corners=False))
            netarc_embeds = []
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
                    img = F.interpolate(_Xt[i:i+1], size=(256, 256), mode='bilinear', align_corners=False)
                    img = img.detach().cpu().numpy()
                    img = img.transpose((0,2,3,1))[0]
                    img = np.clip(255 * img, 0, 255).astype(np.uint8)[:,:,::-1]
                    #img = cv2.copyMakeBorder(img, top=border_size, bottom=border_size, left=border_size, right=border_size, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    #face = face_analyser.get(img)[0]
                    #warped_img_128, _ = face_align.norm_crop2(img, face.kps, 128)
                    warped_img_128 = cv2.resize(img, (128, 128))
                    blob_128 = warped_img_128.astype('float32') / 255.0
                    blob_128 = blob_128[:, :, [2, 1, 0]]
                    blob_128 = blob_128.transpose((2,0,1))
                    blob_128 = np.expand_dims(blob_128, axis=0)
                    #warped_img_256, _ = face_align.norm_crop2(img, face.kps, 256)
                    warped_img_256 = img
                    blob_256 = warped_img_256.astype('float32') / 255.0
                    blob_256 = blob_256[:, :, [2, 1, 0]]
                    blob_256 = blob_256.transpose((2,0,1))
                    blob_256 = np.expand_dims(blob_256, axis=0)
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
                    HQ_Resized = F.interpolate(high_quality_pred_tensor, [112, 112], mode='bilinear', align_corners=False)
                    PRED = netArc(HQ_Resized)
                    high_quality_results_pred.append(PRED)
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
                netarc_embeds.append(embed_old_way[i:i+1])
                
            

        Xs_orig = torch.cat(Xs_orig , dim=0).to(device)
        Xs = torch.cat(Xs , dim=0).to(device)
        Xt = torch.cat([torch.tensor(x) for x in Xt] , dim=0).to(device)
        same_person = torch.cat(same_person, dim=0).to(device)
        embeds = torch.cat([torch.tensor(x) for x in embeds] , dim=0).to(device)
        netarc_embeds = torch.cat(netarc_embeds, dim=0).to(device)

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

        high_quality_results_pred_combined = torch.cat(high_quality_results_pred, dim=0)
        high_quality_results_combined = torch.cat(high_quality_results, dim=0)
        high_quality_results_lmks_combined = torch.cat(high_quality_results_lmks, dim=0)

        #Y_resized = F.interpolate(Xt, size=(128, 128), mode='bilinear', align_corners=False)
        Y_resized = F.interpolate(Y, [112, 112], mode='bilinear', align_corners=False)
        ZY = netArc(Y_resized)

        try:
            y_lmks = []
            #landmark_2d_106 need to be calculated at 128x128, only because that is how they were calculated with the teacher model
            Y_resized_128 = F.interpolate(Y, [128, 128], mode='bilinear', align_corners=False)
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


        # Assuming high_quality_results_lmks_combined is a batch of landmarks
        # Convert landmarks to convex hulls and pad hulls so they all have the same number of points
        #hulls = pad_hulls(calculate_convex_hulls(high_quality_results_lmks_combined))
        # Convert hulls to a tensor and pass to the mask creation function
        #hulls_tensor = [torch.tensor(hull, device=device, dtype=torch.float32) for hull in hulls]
        #hulls_batch = torch.stack(hulls_tensor)
        #Focuses the importance on the area closest to facial landmarks
        #soft_masks = create_soft_mask_batch(high_quality_results_lmks_combined, 112, 112)
        #soft_masks = create_soft_mask_batch(hulls_batch, 112, 112)

        #high_quality_pred = torch.from_numpy(high_quality_pred).to(device)
        #positive_distillation_loss_a = l2_loss(ZY, high_quality_results_pred_combined)
        #teacher_loss = l2_loss(soft_masks*Y_resized, soft_masks*high_quality_results_combined)
        Xt_resized = F.interpolate(Xt, [112, 112], mode='bilinear', align_corners=False)
        teacher_loss = torch.norm((Xt_resized - high_quality_results_combined) - (Xt_resized - Y_resized), p=2)


        netarc_embeds_loss =(1 - torch.cosine_similarity(netarc_embeds, ZY, dim=1)).mean()
        lmks_loss = torch.norm(high_quality_results_lmks_combined - y_lmks_combined, dim=2).mean()
        #weight_a = random.uniform(0.3, 0.7)  # generates a random number between 0.3 and 0.7
        #weight_b = 1 - weight_a  # ensures the total weight sums up to 1
        #weight_a = calculate_sine_weight(iteration)
        #weight_b = 1 - weight_a  # Complementary weight
        #total_loss = (weight_a) * positive_distillation_loss_a3 + (weight_b) * positive_distillation_loss_b 
        total_loss = (35) * netarc_embeds_loss + (20) * lmks_loss  + (2) * teacher_loss 
        #total_loss = positive_distillation_loss_b

        # Backward and optimize
        opt_G.zero_grad()
        total_loss.backward()
        opt_G.step()


        # Progress Report
        batch_time = time.time() - start_time

        if iteration % args.show_step == 0:
            high_quality_results_resized = F.interpolate(high_quality_results_combined, size=(256, 256), mode='bilinear', align_corners=False)
            images = [Xs, Xt, high_quality_results_resized, Y]
            image = make_image_list(images)
            os.makedirs('./output/images/', exist_ok=True)
            cv2.imwrite(f'./output/images/generated_image_{args.run_name}_{str(epoch)}_{iteration:06}.jpg', image[:,:,::-1])

        if iteration % 10 == 0:
            print(f'epoch: {epoch}    {iteration} / {len(dataloader)}')
            print(f'total_loss: {total_loss.item()} batch_time: {batch_time}s')
            
            if args.scheduler:
                print(f'scheduler_G lr: {scheduler_G.get_last_lr()}')

        if iteration % 5000 == 0:
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
    
    if args.vgg:
        dataset = FaceEmbedVGG2(args.dataset_path, same_prob=args.same_person, same_identity=args.same_identity)
    else:
        dataset = FaceEmbed([args.dataset_path], same_prob=args.same_person)
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    loss_adv_accumulated = 20.
    
    for epoch in range(0, max_epoch):
        train_one_epoch_new(G,
                            opt_G,
                            scheduler_G,
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
    parser.add_argument('--vgg', default=True, type=bool, help='When using VGG2 dataset (or any other dataset with several photos for one identity)')
    # weights for loss
    parser.add_argument('--weight_adv', default=1, type=float, help='Adversarial Loss weight')
    parser.add_argument('--weight_attr', default=10, type=float, help='Attributes weight')
    parser.add_argument('--weight_id', default=20, type=float, help='Identity Loss weight')
    parser.add_argument('--weight_rec', default=10, type=float, help='Reconstruction Loss weight')
    parser.add_argument('--weight_eyes', default=0., type=float, help='Eyes Loss weight')
    # training params you may want to change
    parser.add_argument('--backbone', default='unet', const='unet', nargs='?', choices=['unet', 'linknet', 'resnet'], help='Backbone for attribute encoder')
    parser.add_argument('--num_blocks', default=2, type=int, help='Numbers of AddBlocks at AddResblock')
    parser.add_argument('--same_person', default=0.2, type=float, help='Probability of using same person identity during training')
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


