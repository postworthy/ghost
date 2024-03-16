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

from utils.training.helpers import masked_color_consistency_loss
from utils.training.upsampler  import upscale
from models.MultiScalePerceptualColorLoss import MultiScalePerceptualColorLoss
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

print("finished imports")

print("started globals")
face_analyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_analyser.prepare(ctx_id=0, det_size=(640, 640))
bounds_info  = {
        'lower_sum': np.zeros(3, dtype=np.float64),
        'upper_sum': np.zeros(3, dtype=np.float64),
        'count': 0,
        'average_lower_bound': np.zeros(3, dtype=np.uint8),
        'average_upper_bound': np.zeros(3, dtype=np.uint8)
    }

from utils.training.upsampler  import upscale

print("finished globals")

def gaussian_kernel(size, sigma):
    """
    Creates a Gaussian kernel using the specified size and sigma (standard deviation).
    """
    x = torch.arange(-size // 2 + 1., size // 2 + 1.)
    y = torch.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = torch.meshgrid(x, y)
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()

def apply_Gaussian_blur(batch_images, kernel_size=11, sigma=10.5):
    """
    Applies Gaussian blur to a batch of images.
    
    Parameters:
        batch_images (torch.Tensor): Batch of images with shape [BATCH_SIZE, CHANNELS, HEIGHT, WIDTH].
        kernel_size (int): Size of the Gaussian kernel (must be odd).
        sigma (float): Standard deviation (radius) of the Gaussian kernel.
        
    Returns:
        blurred_batch (torch.Tensor): Batch of blurred images.
    """
    # Ensure input is a PyTorch tensor
    if not isinstance(batch_images, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    
    # Generate Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma).to(batch_images.device)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(batch_images.shape[1], 1, 1, 1)
    
    # Apply Gaussian blur
    padding = kernel_size // 2
    blurred_batch = F.conv2d(batch_images, kernel, padding=padding, groups=batch_images.shape[1])
    
    return blurred_batch


def draw_hull_around_objects_in_batch_old(image_batch_tensor):
    # Assuming image_batch_tensor is of shape [BATCH_SIZE, 1, 256, 256]
    BATCH_SIZE, C, H, W = image_batch_tensor.shape  # Extract shape components
    assert C == 1, "The input tensor should have 1 channel (Grayscale)"
    
    # Convert PyTorch tensor to a NumPy array
    image_batch = image_batch_tensor.cpu().detach().numpy()
    
    # Initialize processed_batch with zeros, to keep the same shape as image_batch
    processed_batch = np.zeros_like(image_batch)  # [BATCH_SIZE, 1, 256, 256]
    
    for i, img_tensor in enumerate(image_batch):
        # Convert from [1, 256, 256] (PyTorch format) to [256, 256] (OpenCV format)
        img = np.squeeze(img_tensor, axis=0).astype(np.uint8)

        # Create a binary thresholded version of the image
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

        # Find contours and the convex hull
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest_contour)

            # Create an empty mask and fill the convex hull with black
            mask = np.ones_like(img) * 255  # Start with a white mask
            cv2.drawContours(mask, [hull], -1, 0, -1)  # Fill the hull with black

            # Apply the mask to the image: black inside the hull, white outside
            img = np.where(mask == 0, 0, 255).astype(np.uint8)

        # Ensure the image maintains its shape [1, 256, 256] when reassigned back
        processed_batch[i] = img[np.newaxis, :, :]  # Add back the channel dimension
    
    # Convert the processed batch back to a PyTorch tensor and transfer to the original device
    return torch.from_numpy(processed_batch).to(image_batch_tensor.device)

def draw_hull_around_objects_in_batch(image_batch_tensor, blur_radius=11, morph_radius=15):
    """
    Draw rounded shapes around objects in a batch of images by applying
    morphological operations and Gaussian blur.
    
    Parameters:
        image_batch_tensor (torch.Tensor): Batch of grayscale images with shape [BATCH_SIZE, 1, H, W].
        blur_radius (int): Radius for Gaussian blur to smooth edges.
        morph_radius (int): Radius for morphological operations to create rounded shapes.
    
    Returns:
        torch.Tensor: Batch of processed images with rounded shapes around objects.
    """
    image_batch_tensor = draw_hull_around_objects_in_batch_old(image_batch_tensor)
    BATCH_SIZE, C, H, W = image_batch_tensor.shape  # Extract shape components
    assert C == 1, "The input tensor should have 1 channel (Grayscale)"
    
    # Convert PyTorch tensor to a NumPy array
    image_batch = image_batch_tensor.cpu().detach().numpy()
    
    # Initialize processed_batch with zeros, to keep the same shape as image_batch
    processed_batch = np.zeros_like(image_batch)
    
    for i, img_tensor in enumerate(image_batch):
        # Convert from [1, H, W] (PyTorch format) to [H, W] (OpenCV format)
        img = np.squeeze(img_tensor, axis=0).astype(np.uint8)

        # Create a binary thresholded version of the image
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

        # Apply Gaussian blur to smooth edges
        blurred = cv2.GaussianBlur(binary, (blur_radius, blur_radius), 0)

        # Use morphological closing to round corners
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_radius, morph_radius))
        rounded = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

        # Reapply threshold to clean up the image after blurring and morphological operations
        _, clean_rounded = cv2.threshold(rounded, 127, 255, cv2.THRESH_BINARY_INV)

        # Ensure the image maintains its shape [1, H, W] when reassigned back
        processed_batch[i] = clean_rounded[np.newaxis, :, :]
    
    # Convert the processed batch back to a PyTorch tensor and transfer to the original device
    return torch.from_numpy(processed_batch).to(image_batch_tensor.device)


def apply_mask(images, masks):
    """
    Apply the mask to all images, ensuring all tensors are on the same device.
    
    Parameters:
    - images (Tensor): A batch of images with shape [BATCH_SIZE, 3, 256, 256].
    - masks (Tensor): A batch of masks with shape [BATCH_SIZE, 3, 256, 256].
    
    Returns:
    - Tensor: A batch of images after applying masks with shape [BATCH_SIZE, 3, 256, 256].
    """
    # Detect the device of the images tensor and ensure all operations are performed on this device.
    device = images.device

    # Ensure the masked areas are turned to white (255,255,255) and keep original image pixels otherwise.
    # Adjust the white_pixels tensor to be on the same device as the images.
    white_pixels = torch.tensor([255, 255, 255], dtype=torch.uint8, device=device).view(1, 3, 1, 1)

    # Apply mask: Use where to select pixels from the white_pixels if mask is white, else from the original images.
    masked_images = torch.where(masks > 0.98, white_pixels, images)
    
    return masked_images

def threshold_mask(mask, threshold=0.8):
    """
    Apply a threshold to the mask. Values above the threshold are set to 1, values below are set to 0.
    
    Parameters:
    - mask (Tensor): A tensor with values between 0 and 1.
    - threshold (float): A threshold value between 0 and 1.
    
    Returns:
    - Tensor: The thresholded mask.
    """
    # Apply threshold: values greater than the threshold are set to 1, otherwise 0.
    thresholded_mask = (mask > threshold).float()
    return thresholded_mask



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
    global bounds_info
    universal_multiplier = 1
    verbose_output = args.verbose_output

    # Initialize SAM for segmentation
    sam = sam_model_registry["vit_h"](checkpoint="/app/sam_vit_h_4b8939.pth")
    sam.cuda()  # If using GPU
    #mask_generator = SamAutomaticMaskGenerator(sam)
    predictor = SamPredictor(sam)
    border_size = 100

    for iteration, data in enumerate(dataloader):
        start_time = time.time()
        
        _, _, _, Xt, _ = data
        
        Xt = Xt.to(device)

        with torch.no_grad():
            netarc_embeds = netArc(F.interpolate(Xt, [112, 112], mode='bilinear', align_corners=False))
        
        _masks = []
        _Xt = []
        _netarc_embeds = []
        
        for i, xt  in enumerate(Xt):
            #img = F.interpolate(xt, size=(128, 128), mode='area')
            img = xt.detach().cpu().numpy()
            img = img.transpose((1,2,0))
            img = np.clip(255 * img, 0, 255).astype(np.uint8)[:,:,::-1]
            img_bordered = cv2.copyMakeBorder(img, top=border_size, bottom=border_size, left=border_size, right=border_size, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
            try:
                face = face_analyser.get(img_bordered)[0]
            except Exception as e_3:
                    print(e_3)
                    face = None
            if face:
                face_points = face.kps - border_size
                predictor.set_image(img)
                masks, scores, logits = predictor.predict(
                    point_coords=face_points,
                    point_labels=np.array([1, 1, 1, 1, 1]),
                    multimask_output=False,
                )
                for _, mask in enumerate(masks):
                    mask = mask * 1.0
                    mask_img = torch.tensor(mask)
                    mask_img = mask_img.float()
                    mask_img = mask_img.unsqueeze(0).repeat(3, 1, 1)
                    _masks.append(mask_img)
                    _Xt.append(Xt[i:i+1])
                    _netarc_embeds.append(netarc_embeds[i:i+1])

        Xt = torch.cat(_Xt , dim=0).to(device)
        Yt = torch.stack(_masks, dim=0).to(device)
        netarc_embeds = torch.cat(_netarc_embeds, dim=0).to(device)
        
        # generator training
        opt_G.zero_grad()
        
        #with autocast():
        if verbose_output:
            print(Xt.shape)
            print(Yt.shape)
            print(netarc_embeds.shape)
        
        Y, _ = G(Xt, netarc_embeds)
        #Y, _ = G(Xt_G, netarc_embeds)

            
        if True:
            #Teacher Loss
            teacher_loss = F.mse_loss(Yt, Y)
            teacher_loss_mult = 10000.0

            total_loss = universal_multiplier * (
                teacher_loss_mult*teacher_loss
            )

            
            # Backward and optimize
            opt_G.zero_grad()
            total_loss.backward()
            opt_G.step()

            # Progress Report
            batch_time = time.time() - start_time

            if iteration % args.show_step == 0:
                masked = torch.where(Y >= 0.9, Xt, torch.tensor(0))
                images = [Xt, Yt, Y, masked]
                image = make_image_list(images, normalize=False)
                os.makedirs('./output/images/', exist_ok=True)
                cv2.imwrite(f'./output/images/generated_image_{args.run_name}_{str(epoch)}_{iteration:06}.jpg', image[:,:,::-1])

            if iteration % 10 == 0:
                print(f'epoch:                      {epoch}    {iteration} / {len(dataloader)}')
                print(f'teacher_loss:               {universal_multiplier*teacher_loss_mult*teacher_loss.item()}')
                print(f'total_loss:                 {total_loss.item()} batch_time: {batch_time}s')
                
                if args.scheduler:
                    print(f'scheduler_G lr: {scheduler_G.get_last_lr()}')

            #if iteration % 2500 == 0:
            if iteration % args.save_interval == 0:
                os.makedirs(f'./output/saved_models_{args.run_name}/', exist_ok=True)
                os.makedirs(f'./output/current_models_{args.run_name}/', exist_ok=True)
                torch.save(G.state_dict(), f'./output/saved_models_{args.run_name}/G_latest.pth')
                torch.save(G.state_dict(), f'./output/current_models_{args.run_name}/G_' + str(epoch)+ '_' + f"{iteration:06}" + '.pth')
        else:
            Y = threshold_mask(Y)

            #Yt = threshold_mask(Yt)

            #masked = apply_mask(Xt, Yt)
            masked = torch.where(Yt >= 0.9, Xt, torch.tensor(0))

            images = [Xt, Yt, Y, masked]
            image = make_image_list(images, normalize=False)
            os.makedirs('./output/images/', exist_ok=True)
            cv2.imwrite(f'./output/images/generated_image_{args.run_name}_{str(epoch)}_{iteration:06}.jpg', image[:,:,::-1])
            return
            
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
    teacher
    
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
                        None,
                        None,
                        None,
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


