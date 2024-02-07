
import numpy as np
import torch
import torch.nn.functional as F
import random
from PIL import Image

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

def hsv_to_rgb(hsv):
    """
    Convert a batch of images from HSV to RGB.
    
    :param hsv: Batch of images in HSV format, shape (BATCH_SIZE, 3, HEIGHT, WIDTH)
    :return: Batch of images in RGB format
    """
    h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
    c = v * s
    x = c * (1 - torch.abs((h * 6) % 2 - 1))
    m = v - c

    h1 = (h * 6).floor()
    rgb = torch.zeros_like(hsv)
    rgb[:, 0, :, :][(h1 == 0) | (h1 == 6)] = c[(h1 == 0) | (h1 == 6)]
    rgb[:, 1, :, :][h1 == 0] = x[h1 == 0]
    rgb[:, 2, :, :][h1 == 0] = 0

    rgb[:, 0, :, :][h1 == 1] = x[h1 == 1]
    rgb[:, 1, :, :][h1 == 1] = c[h1 == 1]
    rgb[:, 2, :, :][h1 == 1] = 0

    rgb[:, 0, :, :][h1 == 2] = 0
    rgb[:, 1, :, :][h1 == 2] = c[h1 == 2]
    rgb[:, 2, :, :][h1 == 2] = x[h1 == 2]

    rgb[:, 0, :, :][h1 == 3] = 0
    rgb[:, 1, :, :][h1 == 3] = x[h1 == 3]
    rgb[:, 2, :, :][h1 == 3] = c[h1 == 3]

    rgb[:, 0, :, :][h1 == 4] = x[h1 == 4]
    rgb[:, 1, :, :][h1 == 4] = 0
    rgb[:, 2, :, :][h1 == 4] = c[h1 == 4]

    rgb[:, 0, :, :][h1 == 5] = c[h1 == 5]
    rgb[:, 1, :, :][h1 == 5] = 0
    rgb[:, 2, :, :][h1 == 5] = x[h1 == 5]

    rgb += m.unsqueeze(1).expand_as(rgb)

    return rgb


def modify_images_with_hsv(batch1, batch2):
    """
    Modify batch2 to have the corresponding HSV values from batch1.

    :param batch1: First batch of images (RGB), shape (BATCH_SIZE, 3, HEIGHT, WIDTH)
    :param batch2: Second batch of images (RGB), same shape as batch1
    :return: Modified batch2 with HSV from batch1
    """
    # Compute HSV of the first batch
    hsv_batch1 = get_hsv(batch1)

    # Optionally compute HSV of the second batch if you want to retain some components
    hsv_batch2 = get_hsv(batch2)

    # Replace components. Here, replace all components for simplicity
    # If you want to retain some original components from batch2, selectively replace
    hsv_batch2[:, 0, :, :] = hsv_batch1[:, 0, :, :]  # Replace Hue
    hsv_batch2[:, 1, :, :] = hsv_batch1[:, 1, :, :]  # Replace Saturation
    hsv_batch2[:, 2, :, :] = hsv_batch1[:, 2, :, :]  # Replace Value

    # Convert the modified HSV back to RGB
    modified_batch2 = hsv_to_rgb(hsv_batch2)

    return modified_batch2


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

def batch_edge_loss(batch1, batch2, N):
    """
    Calculate a loss based on the difference in the outer N pixels of images in two batches.
    
    :param batch1: First batch of images, shape (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    :param batch2: Second batch of images, same shape as batch1
    :param N: Number of edge pixels to consider
    :return: Average edge loss for the batch
    """
    batch_size = batch1.size(0)
    total_loss = 0.0

    for i in range(batch_size):
        # Extract the edge pixels for the current pair of images
        top_edge1, top_edge2 = batch1[i, :, :N, :], batch2[i, :, :N, :]
        bottom_edge1, bottom_edge2 = batch1[i, :, -N:, :], batch2[i, :, -N:, :]
        left_edge1, left_edge2 = batch1[i, :, :, :N], batch2[i, :, :, :N]
        right_edge1, right_edge2 = batch1[i, :, :, -N:], batch2[i, :, :, -N:]

        # Compute the loss for each edge of the current pair
        loss = F.mse_loss(top_edge1, top_edge2) + \
               F.mse_loss(bottom_edge1, bottom_edge2) + \
               F.mse_loss(left_edge1, left_edge2) + \
               F.mse_loss(right_edge1, right_edge2)

        # Sum up the losses
        total_loss += loss

    # Average the loss over the batch
    average_loss = total_loss / batch_size

    return average_loss

def color_consistency_loss(batch, center_region = (30, 30, 50, 50), edge_region = (0, 0, 112, 112)):
    """
    Calculate a loss that penalizes color and brightness inconsistencies across a batch of images.
    
    :param batch: Batch of image tensors, shape (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    :param center_region: Tuple defining the center region (x, y, width, height)
    :param edge_region: Tuple defining the edge region (x, y, width, height)
    :return: Average color consistency loss for the batch
    """
    batch_size = batch.size(0)
    total_loss = 0.0

    for i in range(batch_size):
        # Extract the center and edge regions for the current image
        center = batch[i, :, center_region[1]:center_region[1]+center_region[3], center_region[0]:center_region[0]+center_region[2]]
        edge = batch[i, :, edge_region[1]:edge_region[1]+edge_region[3], edge_region[0]:edge_region[0]+edge_region[2]]

         # Calculate mean color/brightness for the edge region
        edge_mean = torch.mean(edge, dim=[1, 2], keepdim=True)

        # Apply the mean color/brightness of the edge to the center region
        center_adjusted = center - (torch.mean(center, dim=[1, 2], keepdim=True) - edge_mean)

        # Compute the loss as the difference between the adjusted center and the original center
        loss = F.mse_loss(center_adjusted, center)

        # Accumulate the loss
        total_loss += loss

    # Average the loss over the batch
    average_loss = total_loss / batch_size

    return average_loss

def to_grayscale_normalize(batch):
    # Convert to grayscale by averaging the channels
    grayscale_batch = torch.mean(batch, dim=1, keepdim=True)

    # Normalize the grayscale images
    normalized_batch = (grayscale_batch - grayscale_batch.min()) / (grayscale_batch.max() - grayscale_batch.min() + 1e-5)
    return normalized_batch

def emboss_filter():
    # Define a 3x3 emboss kernel
    kernel = torch.tensor([[-2, -1, 0],
                           [-1,  1, 1],
                           [ 0,  1, 2]], dtype=torch.float32)
    kernel = kernel.view(1, 1, 3, 3)
    return kernel

def apply_emboss(batch):
    kernel = emboss_filter()
    kernel = kernel.to(batch.device)

    # Apply the filter
    embossed_batch = F.conv2d(batch, kernel, padding=1)
    return embossed_batch


def emboss_loss_func(batch1, batch2):
    # Convert to grayscale and normalize
    batch1_gray = to_grayscale_normalize(batch1)
    batch2_gray = to_grayscale_normalize(batch2)

    # Apply emboss filter to grayscale images
    embossed_batch1 = apply_emboss(batch1_gray)
    embossed_batch2 = apply_emboss(batch2_gray)

    # Compute loss (e.g., mean squared error)
    loss = F.mse_loss(embossed_batch1, embossed_batch2)
    return loss

def to_grayscale(batch):
    """
    Convert a batch of images to grayscale.
    
    :param batch: Batch of images, shape (BATCH_SIZE, 3, HEIGHT, WIDTH)
    :return: Grayscale batch of images
    """
    # Weights for converting to grayscale using the luminosity method
    weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1).to(batch.device)
    
    # Convert to grayscale
    grayscale_batch = torch.sum(batch * weights, dim=1, keepdim=True)
    return grayscale_batch

def structural_loss(batch1, batch2):
    """
    Calculate a loss that ignores color differences and focuses on structure.
    
    :param batch1: First batch of images (RGB), shape (BATCH_SIZE, 3, HEIGHT, WIDTH)
    :param batch2: Second batch of images (RGB), same shape as batch1
    :return: Structural loss
    """
    # Convert both batches to grayscale
    batch1_gray = to_grayscale(batch1)
    batch2_gray = to_grayscale(batch2)

    # Compute loss (e.g., MSE)
    loss = F.mse_loss(batch1_gray, batch2_gray)
    return loss

def compute_eye_loss(eye_heatmaps):
    Xt_heatmap_left, Xt_heatmap_right, Y_heatmap_left, Y_heatmap_right = eye_heatmaps
    L_l2_eyes = F.mse_loss(Xt_heatmap_left, Y_heatmap_left) + F.mse_loss(Xt_heatmap_right, Y_heatmap_right)
    
    return L_l2_eyes

class RandomRGBtoBGR:
    """Transform to convert image from RGB to BGR with a probability."""
    def __init__(self, probability=1/25):
        self.probability = probability

    def __call__(self, img):
        if random.random() < self.probability:
            # Convert PIL Image to numpy array, change RGB to BGR, and convert back to PIL Image
            img = np.array(img)
            img = img[:, :, ::-1]  # This changes RGB to BGR
            img = Image.fromarray(img)
        return img
