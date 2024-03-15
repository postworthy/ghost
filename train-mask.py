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

def is_any_nan(loss, name):
    if torch.isnan(loss).any():
        print(f"Warning: {name} contains NaN values")
        return True
    else:
        return False

def add_gridlines_to_tensor(images, N, W, color=(0, 0, 0)):
    """
    Adds gridlines to a batch of images represented as PyTorch tensors.
    
    Parameters:
        images (torch.Tensor): Batch of images with shape [BATCH_SIZE, 3, 256, 256].
        N (int): Distance between gridlines in pixels.
        W (int): Width of the gridlines in pixels.
        color (tuple): Color of the gridlines in RGB format (default is red).
        
    Returns:
        grid_images (torch.Tensor): Batch of images with gridlines.
    """
    # Check if input is a torch tensor
    if not isinstance(images, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    
    # Check if the images have the correct shape
    if images.shape[1:] != (3, 256, 256):
        raise ValueError("Each image in the batch must be of shape (3, 256, 256).")
    
    # Convert color to tensor and prepare for concatenation
    color_tensor = torch.tensor(color, dtype=torch.uint8).view(3, 1, 1)
    
    # Process each image in the batch
    for i in range(images.shape[0]):
        # Adding vertical gridlines
        #for x in range(N, 256, N):
        #    images[i, :, :, max(x - W//2, 0):min(x + W//2 + 1, 256)] = color_tensor
        
        # Adding horizontal gridlines
        for y in range(N, 256, N):
            images[i, :, max(y - W//2, 0):min(y + W//2 + 1, 256), :] = color_tensor

    return images

def update_color_channel(batch_images, channel, new_value):
    """
    Updates the specified color channel (R, G, or B) of a batch of images.

    Args:
    batch_images (torch.Tensor): A batch of images with shape [BATCH_SIZE, 3, 256, 256].
    channel (str): The color channel to update ('R', 'G', or 'B').
    new_value (int): The new value for the color channel (0 to 255).

    Returns:
    torch.Tensor: The batch of images with the updated color channel.
    """
    assert channel in ['R', 'G', 'B'], "channel must be 'R', 'G', or 'B'"
    assert 0 <= new_value <= 255, "new_value must be between 0 and 255"
    
    # Map channel names to tensor indices
    channel_indices = {'R': 0, 'G': 1, 'B': 2}
    channel_index = channel_indices[channel]
    
    # Normalize the new value to be between 0 and 1
    new_value_normalized = new_value / 255.0
    
    # Update the specified channel for all images in the batch
    batch_images[:, channel_index, :, :] = new_value_normalized
    
    return batch_images


def sobel_edge_detection_gray(batch_images):
    """
    Converts a batch of RGB images to grayscale and then applies Sobel edge detection.
    
    Parameters:
        batch_images (torch.Tensor): Batch of RGB images with shape [BATCH_SIZE, 3, HEIGHT, WIDTH].
        
    Returns:
        edges_batch (torch.Tensor): Batch of grayscale images with Sobel edges detected.
    """
    # Ensure input is a PyTorch tensor
    if not isinstance(batch_images, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    if batch_images.shape[1] != 3:
        raise ValueError("Input tensor should have 3 channels (RGB).")
    
    # RGB to Grayscale conversion coefficients
    rgb_to_gray_weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
    
    # Convert RGB images to grayscale
    graYt_Batch_images = torch.sum(batch_images * rgb_to_gray_weights.to(batch_images.device), dim=1, keepdim=True)
    
    # Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(batch_images.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(batch_images.device)
    
    # Edge detection
    edge_x = F.conv2d(graYt_Batch_images, sobel_x, padding=1)
    edge_y = F.conv2d(graYt_Batch_images, sobel_y, padding=1)
    
    # Combine the edges
    edges_batch = torch.sqrt(edge_x ** 2 + edge_y ** 2)
    
    return edges_batch

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

def apply_threshold_batch_rgb(batch_images, average_over=(5,5)):
    """
    Applies an individual threshold to each image in a batch of grayscale images represented in RGB format.
    The threshold for each image is determined by the average pixel value of its first 5x5 area.
    Pixel values below their respective thresholds are set to 255 across all channels, 
    and pixel values above the thresholds remain unchanged.

    Parameters:
        batch_images (torch.Tensor): A batch of grayscale image tensors in RGB format
                                     with shape [BATCH_SIZE, 3, HEIGHT, WIDTH].

    Returns:
        torch.Tensor: The batch of thresholded images.
    """
    # Ensure the input is a PyTorch tensor
    if not isinstance(batch_images, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    
    # Check if the images are a 4D tensor (batch of RGB grayscale)
    if batch_images.ndim != 4:
        raise ValueError("Input must be a 4D tensor for a batch of RGB grayscale images.")
    
    # Calculate the threshold for each image based on the average of the first pixels
    first = batch_images[:, :, :average_over[0], :average_over[0]]
    # Calculate the average across the 5x5 area for each channel, then take the mean of these averages
    thresholds = first.mean(dim=[2, 3]).mean(dim=1, keepdim=True)

    # Initialize output batch
    output_batch = batch_images.clone()

    # Apply the individual thresholds across all color channels
    for i, threshold in enumerate(thresholds):
        below_threshold = batch_images[i] > threshold
        output_batch[i][below_threshold] = 255  # Set pixels below individual threshold to white

    return output_batch

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

def get_landmarks(Xt):
    border_size=100
    lmks=[]
    error_index=[]
    for i, _  in enumerate(Xt):
        try:
            img = F.interpolate(Xt[i:i+1], size=(128, 128), mode='area')
            img = img.detach().cpu().numpy()
            img = img.transpose((0,2,3,1))[0]
            img = np.clip(255 * img, 0, 255).astype(np.uint8)[:,:,::-1]
            img = cv2.copyMakeBorder(img, top=border_size, bottom=border_size, left=border_size, right=border_size, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
            face = face_analyser.get(img)[0]
            lmks_tensor = torch.from_numpy(face.landmark_2d_106).unsqueeze(0).to(Xt.device)
            lmks.append(lmks_tensor)
        except Exception as e:
            lmks.append([])
            error_index.append(i)
            continue
    if len(error_index) < len(Xt):
        for _, i in enumerate(error_index):
            for _, lmk in enumerate(lmks):
                if len(lmk) > 0:
                    lmks[i] = torch.zeros_like(lmk, device=lmk.device)
    else:
        return None
    
    return torch.cat(lmks, dim=0)

def apply_skin_mask_to_batch(batch_images):
    """
    Applies a skin mask to a batch of images and whitens the background.

    Args:
    batch_images (torch.Tensor): A batch of images with shape (BATCH_SIZE, 3, 256, 256).

    Returns:
    torch.Tensor: A batch of images with skin areas highlighted and the background whitened.
    """
    global bounds_info
    # Initialize an empty list to store the masked images
    masked_images = []

    # Detect the device of the batch_images tensor
    device = batch_images.device

    # Convert each image in the batch from Torch tensor to NumPy array and apply the skin mask
    for image_tensor in batch_images:
        # Convert the tensor to an array: (3, 256, 256) -> (256, 256, 3)
        image_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
        image_np = (image_np * 255).astype(np.uint8)  # Rescale to [0, 255] for OpenCV
        
        # Convert the image from RGB to HSV color space
        hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

        # Define the lower and upper boundaries of the 'skin' color in HSV
        #lower_bound = np.array([0, 48, 80], dtype="uint8")
        #upper_bound = np.array([20, 255, 255], dtype="uint8")
        #lower_bound = bounds_info['average_lower_bound']
        #upper_bound = bounds_info['average_upper_bound']
        lower_bound = np.array([12, 7, 67])
        upper_bound = np.array([151, 109, 188])


        # Find the colors within the specified boundaries and apply the mask
        skin_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # Apply a series of erosions and dilations to the mask using an elliptical kernel
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        #skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
        #skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

        # Blur the mask to help remove noise
        skin_mask = cv2.GaussianBlur(skin_mask, (33, 33), 0)

        # Convert mask back to a binary mask [0, 1]
        skin_mask = (skin_mask / 255).astype(np.float32)

        # Invert the skin mask to make skin areas 0 and non-skin areas 1
        inverted_skin_mask = 1 - skin_mask

        # Apply the inverted skin mask to whiten the background
        # Original image areas (skin) are kept, background is turned to white
        background_whitened = image_np * skin_mask[:, :, np.newaxis] + (inverted_skin_mask[:, :, np.newaxis] * 255)

        # Convert the processed image back to a torch tensor and add to the list
        whitened_tensor = torch.from_numpy(background_whitened.transpose(2, 0, 1)).float() / 255.0
        masked_images.append(whitened_tensor)

    # Stack the list of tensors into a batch and move to the original device
    whitened_batch = torch.stack(masked_images).to(device)
    
    return whitened_batch

def calculate_dynamic_hsv_bounds(batch_images):
    """
    Calculate dynamic HSV bounds for skin detection based on a batch of images.

    Args:
    batch_images (torch.Tensor): A batch of images with shape [batch_size, 3, 256, 256].

    Returns:
    np.array: Lower bounds of HSV values.
    np.array: Upper bounds of HSV values.
    """
    # List to hold all HSV values from the batch
    hsv_values = []

    # Iterate through the batch of images
    for image_tensor in batch_images:
        # Convert tensor to numpy array and rescale [0, 1] to [0, 255]
        image_np = (image_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        
        # Convert the image from RGB to HSV color space
        hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        
        # Create a mask where all pixels are True except those where all channels are 255 (background)
        # Note: Background pixels were set to 1.0 (255 after rescaling) during preprocessing
        #foreground_mask = ~(image_np == [255, 255, 255]).all(axis=2)
        foreground_mask = ~(image_np == [1, 1, 1]).all(axis=2)

        # Extract HSV values of foreground pixels and append to the list
        hsv_values.append(hsv_image[foreground_mask])

    # Concatenate all HSV values from the batch
    hsv_values = np.concatenate(hsv_values, axis=0)

    # Compute lower and upper bounds: You might choose to set these based on percentiles
    # For example, using the 5th and 95th percentiles to exclude extreme values
    lower_bound = np.percentile(hsv_values, 5, axis=0).astype(np.uint8)
    upper_bound = np.percentile(hsv_values, 95, axis=0).astype(np.uint8)

    return lower_bound, upper_bound

def update_running_bounds(batch_images):
    global bounds_info
    """
    Updates the running sums, counts, and average bounds based on a new batch of images.

    Args:
    batch_images (torch.Tensor): A batch of images with shape [batch_size, 3, 256, 256].
    bounds_info (dict): A dictionary containing the running sums, counts, and average bounds.

    Returns:
    None: The bounds_info dictionary is updated in place.
    """
    # Calculate dynamic HSV bounds for the current batch
    lower_bound, upper_bound = calculate_dynamic_hsv_bounds(batch_images.cpu())  # Ensure the images are on CPU
    
    # Update the running sums and count
    bounds_info['lower_sum'] += lower_bound
    bounds_info['upper_sum'] += upper_bound
    bounds_info['count'] += 1
    
    # Update the running average bounds
    bounds_info['average_lower_bound'] = (bounds_info['lower_sum'] / bounds_info['count']).astype(np.uint8)
    bounds_info['average_upper_bound'] = (bounds_info['upper_sum'] / bounds_info['count']).astype(np.uint8)

def detect_sharp_regions_tensor(input_tensor, threshold=0.2):
    """
    Detects sharp regions in a batch of images represented as a 4D Tensor.
    
    Args:
        input_tensor (Tensor): A float tensor of shape [BATCH_SIZE, 3, HEIGHT, WIDTH]
                               representing a batch of images.
        threshold (float): Threshold for edge detection, normalized between 0 and 1.

    Returns:
        Tensor: A tensor with sharp regions highlighted, of shape [BATCH_SIZE, 1, HEIGHT, WIDTH].
    """
    # Ensure input is a float tensor
    input_tensor = input_tensor.float()

    # Move input tensor to device (CPU or GPU)
    device = input_tensor.device

    # Convert RGB to grayscale by averaging channels
    gray_scale = input_tensor.mean(dim=1, keepdim=True)

    # Define a Laplacian kernel and move it to the same device as the input tensor
    laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device)
    laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3).repeat(1, 1, 1, 1)

    # Apply padding for the same output size
    padded_gray = F.pad(gray_scale, (1, 1, 1, 1), mode='reflect')

    # Apply the Laplacian operator
    laplacian_output = F.conv2d(padded_gray, laplacian_kernel)

    # Find absolute values (since edges can be negative)
    abs_laplacian = torch.abs(laplacian_output)

    # Normalize the absolute values to be between 0 and 1
    norm_laplacian = abs_laplacian / torch.max(abs_laplacian)

    # Threshold to highlight sharp regions
    sharp_regions = (norm_laplacian > threshold).float()

    return sharp_regions


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
    for iteration, data in enumerate(dataloader):
        start_time = time.time()
        
        _, _, _, Xt, _ = data
        
        Xt = Xt.to(device)

        with torch.no_grad():
            netarc_embeds = netArc(F.interpolate(Xt, [112, 112], mode='bilinear', align_corners=False))

        #Xt_R = update_color_channel(Xt.clone(), 'R', 255)
        #Xt_G = update_color_channel(Xt.clone(), 'G', 255)
        #Xt_B = update_color_channel(Xt.clone(), 'B', 255)

        Xt_R = add_gridlines_to_tensor(Xt.clone(), N=3, W=1, color=(255, 0, 0))
        Xt_G = add_gridlines_to_tensor(Xt.clone(), N=3, W=1, color=(0, 255, 0))
        Xt_B = add_gridlines_to_tensor(Xt.clone(), N=3, W=1, color=(0, 0, 255))

        for i in range(0, 2):     
            #Xt = F.interpolate(Xt, [64, 64], mode='bilinear', align_corners=False)
            #Xt = F.interpolate(Xt, [256, 256], mode='area')
            Xt_G = upscale(Xt_G)

        #rgb_to_gray_weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
        #skin_detect = torch.sum(skin_detect * rgb_to_gray_weights.to(skin_detect.device), dim=1, keepdim=True)
        #skin_detect = apply_threshold_batch_rgb(skin_detect)
        #skin_detect = draw_hull_around_objects_in_batch(skin_detect)


        Xt_R = Xt_R.to(device)
        Xt_G = Xt_G.to(device)
        Xt_B = Xt_B.to(device)

        
        
        lmks = get_landmarks(Xt)

        # generator training
        opt_G.zero_grad()
        
        #with autocast():
        if verbose_output:
            print(Xt.shape)
            print(netarc_embeds.shape)
        
        #Get separate RGB results
        Yt_R, _ = teacher(Xt_R, netarc_embeds)
        Yt_G, _ = teacher(Xt_G, netarc_embeds)
        Yt_B, _ = teacher(Xt_B, netarc_embeds)

        Y, _ = G(Xt, netarc_embeds)
        #Y, _ = G(Xt_G, netarc_embeds)

        #Perform edge detection
        Yt_R = sobel_edge_detection_gray(Yt_R)
        Yt_G = sobel_edge_detection_gray(Yt_G)
        Yt_B = sobel_edge_detection_gray(Yt_B)

        #Apply gausian blur to clean up
        Yt_R = apply_Gaussian_blur(Yt_R)
        Yt_G = apply_Gaussian_blur(Yt_G)
        Yt_B = apply_Gaussian_blur(Yt_B)

        #Apply dynamic thresholding to get mask for R,G,B
        Yt_R = apply_threshold_batch_rgb(Yt_R, (2,2))
        Yt_G = apply_threshold_batch_rgb(Yt_G, (8,8))
        Yt_B = apply_threshold_batch_rgb(Yt_B, (2,2))

        #Combine R,G,B masks
        #Yt = torch.min(Yt_R, torch.min(Yt_G, Yt_B))
        Yt = Yt_G

        #Combine G,B masks
        #Yt = torch.min(Yt_G, Yt_B)

        Yt = draw_hull_around_objects_in_batch(Yt)

        #Black and White
        #Yt = (Yt == 255).float() * 255
        
        if False:
            #Teacher Loss
            teacher_loss = F.mse_loss(Yt, Y)
            teacher_loss_mult = 1.0

            #Other than skin loss
            Y = threshold_mask(Y)
            skin_loss = masked_color_consistency_loss(Xt, Y)
            skin_loss_mult = 1000.0

            masked = apply_mask(Xt, Y)
            with torch.no_grad():
                masked_netarc_embeds = netArc(F.interpolate(masked, [112, 112], mode='bilinear', align_corners=False))

            netarc_embeds_loss = (1 - torch.cosine_similarity(netarc_embeds, masked_netarc_embeds, dim=1)).mean()
            netarc_embeds_loss_mult = 1000.0

            lmks_loss_mult = 100000.0
            if lmks != None:
                lmks_masked = get_landmarks(masked)
                if lmks_masked != None:
                    #lmks_loss = F.mse_loss(lmks, lmks_masked)
                    lmks_loss = (1 - torch.cosine_similarity(lmks, lmks_masked, dim=1)).mean()
                    
                else:
                    lmks_loss = torch.tensor(0.0).to(device)
            else:
                lmks_loss = torch.tensor(0.0).to(device)


            update_running_bounds(masked)

            skin_detect = apply_skin_mask_to_batch(Xt)    

            masked_skin_detect_loss = F.mse_loss(masked, skin_detect)
            masked_skin_detect_loss_mult = 100.0

            total_loss = universal_multiplier * (
                teacher_loss_mult*teacher_loss #+ 
                #skin_loss_mult*skin_loss + 
                #netarc_embeds_loss_mult*netarc_embeds_loss +
                #lmks_loss_mult*lmks_loss +
                #masked_skin_detect_loss_mult*masked_skin_detect_loss
            )

            
            # Backward and optimize
            opt_G.zero_grad()
            total_loss.backward()
            opt_G.step()

            # Progress Report
            batch_time = time.time() - start_time

            if iteration % args.show_step == 0:
                Yt_R_Orig, _ = teacher(Xt_R, netarc_embeds)
                Yt_G_Orig, _ = teacher(Xt_G, netarc_embeds)
                Yt_B_Orig, _ = teacher(Xt_B, netarc_embeds)
                images = [Xt, Yt_R_Orig, Yt_G_Orig, Yt_B_Orig, Yt, Y, skin_detect, masked]
                image = make_image_list(images, normalize=False)
                os.makedirs('./output/images/', exist_ok=True)
                cv2.imwrite(f'./output/images/generated_image_{args.run_name}_{str(epoch)}_{iteration:06}.jpg', image[:,:,::-1])

            if iteration % 10 == 0:
                print(f'epoch:                      {epoch}    {iteration} / {len(dataloader)}')
                print("Average Lower Bound:", bounds_info['average_lower_bound'])
                print("Average Upper Bound:", bounds_info['average_upper_bound'])
                print(f'teacher_loss:               {universal_multiplier*teacher_loss_mult*teacher_loss.item()}')
                print(f'skin_loss:                  {universal_multiplier*skin_loss_mult*skin_loss.item()}')
                print(f'netarc_embeds_loss:         {universal_multiplier*netarc_embeds_loss_mult*netarc_embeds_loss.item()}')
                print(f'lmks_loss:                  {universal_multiplier*lmks_loss_mult*lmks_loss.item()}')
                print(f'masked_loss:                {universal_multiplier*masked_skin_detect_loss_mult*masked_skin_detect_loss.item()}')
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
            Yt_R_Orig, _ = teacher(Xt_R, netarc_embeds)
            Yt_G_Orig, _ = teacher(Xt_G, netarc_embeds)
            Yt_B_Orig, _ = teacher(Xt_B, netarc_embeds)
            Y = threshold_mask(Y)

            Yt = threshold_mask(Yt)

            masked = apply_mask(Xt, Yt)
            
            update_running_bounds(masked)
            print("Average Lower Bound:", bounds_info['average_lower_bound'])
            print("Average Upper Bound:", bounds_info['average_upper_bound'])
            skin_detect = apply_skin_mask_to_batch(Xt_G)
            skin_detect = threshold_mask(skin_detect)
            images = [Xt, Yt_R_Orig, Yt_G_Orig, Yt_B_Orig, Yt_R, Yt_G, Yt_B, Yt, Y, 
                      skin_detect, 
                      masked]
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


