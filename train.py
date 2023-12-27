print("started imports")

import sys
import argparse
import time
import cv2
import wandb
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
#sys.path.append('./apex/')

#from apex import amp
#from torch.cuda import amp
from network.AEI_Net import *
from network.MultiscaleDiscriminator import *
from utils.training.Dataset import FaceEmbedVGG2, FaceEmbed
from utils.training.image_processing import make_image_list, get_faceswap
from utils.training.losses import hinge_loss, compute_discriminator_loss, compute_generator_losses
#from utils.training.detector import detect_landmarks, paint_eyes
#from AdaptiveWingLoss.core import models
from arcface_model.iresnet import iresnet100

import insightface
import onnxruntime
from onnx import numpy_helper
import onnx
from insightface.utils import face_align

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

def train_one_epoch_new(G: 'generator model', 
                        D: 'discriminator model', 
                        opt_G: "generator opt", 
                        opt_D: "discriminator opt",
                        scheduler_G: "scheduler G opt",
                        scheduler_D: "scheduler D opt",
                        netArc: 'ArcFace model',
                        model_ft: 'Landmark Detector',
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
        #Xs_orig = Xs_orig.to(device)
        #Xs = Xs.to(device)
        #Xt = Xt.to(device)
        #same_person = same_person.to(device)

        # get the identity embeddings of Xs
        Xs_orig = []
        Xs = [] 
        Xt = [] 
        same_person = []
        embeds = []
        high_quality_results = []
        high_quality_results_pred = []
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

                high_quality_pred = high_quality_teacher_session.run(
                    [high_quality_output_names],
                    {
                        high_quality_input_names[0]: blob_128, 
                        high_quality_input_names[1]: embed
                    }
                )[0]
                if verbose_output:
                    img_fake = high_quality_pred.transpose((0,2,3,1))[0]
                    bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:,:,::-1]
                    Image.fromarray(cv2.cvtColor(bgr_fake,cv2.COLOR_RGBA2BGR)).save(f'./output/images/generated_image_HQ_OUT_{i}.jpg')
                high_quality_pred_tensor = torch.from_numpy(high_quality_pred).to(device)
                HQ_Resized = F.interpolate(high_quality_pred_tensor, [112, 112], mode='bilinear', align_corners=False)
                PRED = netArc(HQ_Resized)
                high_quality_results_pred.append(PRED)
                high_quality_results.append(HQ_Resized)
                
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
        
        #Y_resized = F.interpolate(Xt, size=(128, 128), mode='bilinear', align_corners=False)
        Y_resized = F.interpolate(Y, [112, 112], mode='bilinear', align_corners=False)
        ZY = netArc(Y_resized)
    
        high_quality_results_pred_combined = torch.cat(high_quality_results_pred, dim=0)
        high_quality_results_combined = torch.cat(high_quality_results, dim=0)
        
        #high_quality_pred = torch.from_numpy(high_quality_pred).to(device)
        positive_distillation_loss_a = l2_loss(ZY, high_quality_results_pred_combined)
        positive_distillation_loss_b = l2_loss(Y_resized, high_quality_results_combined)
        total_loss = (0.5) * positive_distillation_loss_a + (0.5) * positive_distillation_loss_b 
        #total_loss = positive_distillation_loss_b

        # Backward and optimize
        opt_G.zero_grad()
        total_loss.backward()
        opt_G.step()


        # Progress Report
        batch_time = time.time() - start_time

        if iteration % args.show_step == 0:
            images = [Xs, Xt, Y]
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



def train_one_epoch(G: 'generator model', 
                    D: 'discriminator model', 
                    opt_G: "generator opt", 
                    opt_D: "discriminator opt",
                    scheduler_G: "scheduler G opt",
                    scheduler_D: "scheduler D opt",
                    netArc: 'ArcFace model',
                    model_ft: 'Landmark Detector',
                    args: 'Args Namespace',
                    dataloader: torch.utils.data.DataLoader,
                    device: 'torch device',
                    epoch:int,
                    loss_adv_accumulated:int):
    
    for iteration, data in enumerate(dataloader):
        start_time = time.time()
        
        Xs_orig, Xs, Xt, same_person = data

        Xs_orig = Xs_orig.to(device)
        Xs = Xs.to(device)
        Xt = Xt.to(device)
        same_person = same_person.to(device)

        # get the identity embeddings of Xs
        with torch.no_grad():
            embed = netArc(F.interpolate(Xs_orig, [112, 112], mode='bilinear', align_corners=False))

        diff_person = torch.ones_like(same_person)

        if args.diff_eq_same:
            same_person = diff_person
    
        # generator training
        opt_G.zero_grad()
        
        Y, Xt_attr = G(Xt, embed)
        Di = D(Y)
        ZY = netArc(F.interpolate(Y, [112, 112], mode='bilinear', align_corners=False))
        
        #if args.eye_detector_loss:
        #    Xt_eyes, Xt_heatmap_left, Xt_heatmap_right = detect_landmarks(Xt, model_ft)
        #    Y_eyes, Y_heatmap_left, Y_heatmap_right = detect_landmarks(Y, model_ft)
        #    eye_heatmaps = [Xt_heatmap_left, Xt_heatmap_right, Y_heatmap_left, Y_heatmap_right]
        #else:
        eye_heatmaps = None
            
        lossG, loss_adv_accumulated, L_adv, L_attr, L_id, L_rec, L_l2_eyes = compute_generator_losses(G, Y, Xt, Xt_attr, Di,
                                                                             embed, ZY, eye_heatmaps,loss_adv_accumulated, 
                                                                             diff_person, same_person, args)
        
        #with amp.scale_loss(lossG, opt_G) as scaled_loss:
        #    scaled_loss.backward()
        #opt_G.step()
        #if args.scheduler:
        #    scheduler_G.step()
        
        # discriminator training
        opt_D.zero_grad()
        lossD = compute_discriminator_loss(D, Y, Xs, diff_person)
        #with amp.scale_loss(lossD, opt_D) as scaled_loss:
        #    scaled_loss.backward()

        if (not args.discr_force) or (loss_adv_accumulated < 4.):
            opt_D.step()
        if args.scheduler:
            scheduler_D.step()
        
        
        batch_time = time.time() - start_time

        if iteration % args.show_step == 0:
            images = [Xs, Xt, Y]
            if args.eye_detector_loss:
                Xt_eyes_img = paint_eyes(Xt, Xt_eyes)
                Yt_eyes_img = paint_eyes(Y, Y_eyes)
                images.extend([Xt_eyes_img, Yt_eyes_img])
            image = make_image_list(images)
            if args.use_wandb:
                wandb.log({"gen_images":wandb.Image(image, caption=f"{epoch:03}" + '_' + f"{iteration:06}")})
            else:
                cv2.imwrite('./images/generated_image.jpg', image[:,:,::-1])
        
        if iteration % 10 == 0:
            print(f'epoch: {epoch}    {iteration} / {len(dataloader)}')
            print(f'lossD: {lossD.item()}    lossG: {lossG.item()} batch_time: {batch_time}s')
            print(f'L_adv: {L_adv.item()} L_id: {L_id.item()} L_attr: {L_attr.item()} L_rec: {L_rec.item()}')
            if args.eye_detector_loss:
                print(f'L_l2_eyes: {L_l2_eyes.item()}')
            print(f'loss_adv_accumulated: {loss_adv_accumulated}')
            if args.scheduler:
                print(f'scheduler_G lr: {scheduler_G.get_last_lr()} scheduler_D lr: {scheduler_D.get_last_lr()}')
        
        if args.use_wandb:
            if args.eye_detector_loss:
                wandb.log({"loss_eyes": L_l2_eyes.item()}, commit=False)
            wandb.log({"loss_id": L_id.item(),
                       "lossD": lossD.item(),
                       "lossG": lossG.item(),
                       "loss_adv": L_adv.item(),
                       "loss_attr": L_attr.item(),
                       "loss_rec": L_rec.item()})
        
        if iteration % 5000 == 0:
            torch.save(G.state_dict(), f'./saved_models_{args.run_name}/G_latest.pth')
            torch.save(D.state_dict(), f'./saved_models_{args.run_name}/D_latest.pth')

            torch.save(G.state_dict(), f'./current_models_{args.run_name}/G_' + str(epoch)+ '_' + f"{iteration:06}" + '.pth')
            torch.save(D.state_dict(), f'./current_models_{args.run_name}/D_' + str(epoch)+ '_' + f"{iteration:06}" + '.pth')

        if (iteration % 250 == 0) and (args.use_wandb):
            ### Посмотрим как выглядит свап на трех конкретных фотках, чтобы проследить динамику
            G.eval()

            res1 = get_faceswap('examples/images/training//source1.png', 'examples/images/training//target1.png', G, netArc, device)
            res2 = get_faceswap('examples/images/training//source2.png', 'examples/images/training//target2.png', G, netArc, device)  
            res3 = get_faceswap('examples/images/training//source3.png', 'examples/images/training//target3.png', G, netArc, device)
            
            res4 = get_faceswap('examples/images/training//source4.png', 'examples/images/training//target4.png', G, netArc, device)
            res5 = get_faceswap('examples/images/training//source5.png', 'examples/images/training//target5.png', G, netArc, device)  
            res6 = get_faceswap('examples/images/training//source6.png', 'examples/images/training//target6.png', G, netArc, device)
            
            output1 = np.concatenate((res1, res2, res3), axis=0)
            output2 = np.concatenate((res4, res5, res6), axis=0)
            
            output = np.concatenate((output1, output2), axis=1)

            wandb.log({"our_images":wandb.Image(output, caption=f"{epoch:03}" + '_' + f"{iteration:06}")})

            G.train()


def train(args, device):
    # training params
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    
    # initializing main models
    G = AEI_Net(args.backbone, num_blocks=args.num_blocks, c_id=512).to(device)
    D = MultiscaleDiscriminator(input_nc=3, n_layers=5, norm_layer=torch.nn.InstanceNorm2d).to(device)
    G.train()
    D.train()
    
    # initializing model for identity extraction
    netArc = iresnet100(fp16=False)
    netArc.load_state_dict(torch.load('arcface_model/backbone.pth'))
    netArc=netArc.cuda()
    netArc.eval()
    
    #if args.eye_detector_loss:
    #    model_ft = models.FAN(4, "False", "False", 98)
    #    checkpoint = torch.load('./AdaptiveWingLoss/AWL_detector/WFLW_4HG.pth')
    #    if 'state_dict' not in checkpoint:
    #        model_ft.load_state_dict(checkpoint)
    #    else:
    #        pretrained_weights = checkpoint['state_dict']
    #        model_weights = model_ft.state_dict()
    #        pretrained_weights = {k: v for k, v in pretrained_weights.items() \
    #                              if k in model_weights}
    #        model_weights.update(pretrained_weights)
    #        model_ft.load_state_dict(model_weights)
    #    model_ft = model_ft.to(device)
    #    model_ft.eval()
    #else:
    model_ft=None
    
    opt_G = optim.Adam(G.parameters(), lr=args.lr_G, betas=(0, 0.999), weight_decay=1e-4)
    opt_D = optim.Adam(D.parameters(), lr=args.lr_D, betas=(0, 0.999), weight_decay=1e-4)

    #G, opt_G = amp.initialize(G, opt_G, opt_level=args.optim_level)
    #D, opt_D = amp.initialize(D, opt_D, opt_level=args.optim_level)
    
    if args.scheduler:
        scheduler_G = scheduler.StepLR(opt_G, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
        scheduler_D = scheduler.StepLR(opt_D, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    else:
        scheduler_G = None
        scheduler_D = None
        
    if args.pretrained:
        try:
            G.load_state_dict(torch.load(args.G_path, map_location=torch.device('cpu')), strict=False)
            D.load_state_dict(torch.load(args.D_path, map_location=torch.device('cpu')), strict=False)
            print("Loaded pretrained weights for G and D")
        except FileNotFoundError as e:
            print("Not found pretrained weights. Continue without any pretrained weights.")
    
    if args.vgg:
        dataset = FaceEmbedVGG2(args.dataset_path, same_prob=args.same_person, same_identity=args.same_identity)
    else:
        dataset = FaceEmbed([args.dataset_path], same_prob=args.same_person)
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    # Будем считать аккумулированный adv loss, чтобы обучать дискриминатор только когда он ниже порога, если discr_force=True
    loss_adv_accumulated = 20.
    
    for epoch in range(0, max_epoch):
        train_one_epoch_new(G,
                            D,
                            opt_G,
                            opt_D,
                            scheduler_G,
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
    
    print("Starting traing")
    train(args, device=device)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # dataset params
    parser.add_argument('--dataset_path', default='/VggFace2-crop/', help='Path to the dataset. If not VGG2 dataset is used, param --vgg should be set False')
    parser.add_argument('--G_path', default='./saved_models/G.pth', help='Path to pretrained weights for G. Only used if pretrained=True')
    parser.add_argument('--D_path', default='./saved_models/D.pth', help='Path to pretrained weights for D. Only used if pretrained=True')
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
    parser.add_argument('--discr_force', default=False, type=bool, help='If True Discriminator would not train when adversarial loss is high')
    parser.add_argument('--scheduler', default=False, type=bool, help='If True decreasing LR is used for learning of generator and discriminator')
    parser.add_argument('--scheduler_step', default=5000, type=int)
    parser.add_argument('--scheduler_gamma', default=0.2, type=float, help='It is value, which shows how many times to decrease LR')
    parser.add_argument('--eye_detector_loss', default=False, type=bool, help='If True eye loss with using AdaptiveWingLoss detector is applied to generator')
    # info about this run
    parser.add_argument('--use_wandb', default=False, type=bool, help='Use wandb to track your experiments or not')
    parser.add_argument('--run_name', required=True, type=str, help='Name of this run. Used to create folders where to save the weights.')
    parser.add_argument('--wandb_project', default='your-project-name', type=str)
    parser.add_argument('--wandb_entity', default='your-login', type=str)
    # training params you probably don't want to change
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr_G', default=4e-4, type=float)
    parser.add_argument('--lr_D', default=4e-4, type=float)
    parser.add_argument('--max_epoch', default=2000, type=int)
    parser.add_argument('--show_step', default=500, type=int)
    parser.add_argument('--save_epoch', default=1, type=int)
    parser.add_argument('--optim_level', default='O2', type=str)
    parser.add_argument('--verbose_output', default=False, type=bool, help='More print() when training')

    args = parser.parse_args()
    
    if args.vgg==False and args.same_identity==True:
        raise ValueError("Sorry, you can't use some other dataset than VGG2 Faces with param same_identity=True")
    
    if args.use_wandb==True:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, settings=wandb.Settings(start_method='fork'))

        config = wandb.config
        config.dataset_path = args.dataset_path
        config.weight_adv = args.weight_adv
        config.weight_attr = args.weight_attr
        config.weight_id = args.weight_id
        config.weight_rec = args.weight_rec
        config.weight_eyes = args.weight_eyes
        config.same_person = args.same_person
        config.Vgg2Face = args.vgg
        config.same_identity = args.same_identity
        config.diff_eq_same = args.diff_eq_same
        config.discr_force = args.discr_force
        config.scheduler = args.scheduler
        config.scheduler_step = args.scheduler_step
        config.scheduler_gamma = args.scheduler_gamma
        config.eye_detector_loss = args.eye_detector_loss
        config.pretrained = args.pretrained
        config.run_name = args.run_name
        config.G_path = args.G_path
        config.D_path = args.D_path
        config.batch_size = args.batch_size
        config.lr_G = args.lr_G
        config.lr_D = args.lr_D
    elif not os.path.exists('./images'):
        os.mkdir('./images')
    
    # Создаем папки, чтобы было куда сохранять последние веса моделей, а также веса с каждой эпохи
    if not os.path.exists(f'./saved_models_{args.run_name}'):
        os.mkdir(f'./saved_models_{args.run_name}')
        os.mkdir(f'./current_models_{args.run_name}')
    
    main(args)


