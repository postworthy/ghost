import torch
import numpy as np
from network.AEI_Net import AEI_Net
from arcface_model.iresnet import iresnet100
import torch.nn.functional as F

def normalize_and_torch(image: np.ndarray) -> torch.tensor:
    """
    Normalize image and transform to torch
    """
    image = torch.tensor(image.copy(), dtype=torch.float32).cuda()
    if image.max() > 1.:
        image = image/255.
    
    image = image.permute(2, 0, 1).unsqueeze(0)
    image = (image - 0.5) / 0.5

    return image

def export_ghost_onnx():
    # main model for generation
    torch_model = AEI_Net('unet', num_blocks=2, c_id=512)
    torch_model.eval()
    torch_model.load_state_dict(torch.load('weights/G_unet_2blocks.pth', map_location=torch.device('cpu')))
    torch_model = torch_model.cuda()
    torch_model = torch_model

    batch_size = 1
    target = torch.randn(batch_size, 3, 256, 256, requires_grad=True)
    source = torch.randn(1, 512, requires_grad=True).unsqueeze(0)

    #source = (source - 0.5) / 0.5

    target = target.cuda()
    source = source.cuda()

    img, other = torch_model(target, source)
    print(img.shape)
    for i, x in enumerate(other):
        print(x.cpu().detach().numpy().shape)
        
    
    # Export the model
    torch.onnx.export(torch_model,                                      # model being run
                    (target, source),                                   # model input (or a tuple for multiple inputs)
                    "/app/ghost/output/ghost.onnx",                     # where to save the model (can be a file or file-like object)
                    export_params=True,                                 # store the trained parameter weights inside the model file
                    opset_version=11,                                   # the ONNX version to export the model to
                    do_constant_folding=True,                           # whether to execute constant folding for optimization
                    input_names = ['target', 'source'],                 # the model's input names
                    output_names = ['output'],                          # the model's output names
                    dynamic_axes={'target' : {0 : 'batch_size'},        # variable length axes
                                  'output' : {0 : 'batch_size'}})

def export_arcface_onnx():
    torch_model = iresnet100(fp16=False)
    torch_model.load_state_dict(torch.load('arcface_model/backbone.pth'))
    torch_model=torch_model.cuda()
    torch_model.eval()

    batch_size = 1
    input = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
    input = F.interpolate(input, scale_factor=0.5, mode='bilinear', align_corners=True)
    input = input.cuda()
    input_embeds = torch_model(input)

    # Export the model
    torch.onnx.export(torch_model,                                      # model being run
                    input,                                             # model input (or a tuple for multiple inputs)
                    "/app/ghost/output/arcface.onnx",                   # where to save the model (can be a file or file-like object)
                    export_params=True,                                 # store the trained parameter weights inside the model file
                    opset_version=11,                                   # the ONNX version to export the model to
                    do_constant_folding=True,                           # whether to execute constant folding for optimization
                    input_names = ['input'],                            # the model's input names
                    output_names = ['output'],                          # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},         # variable length axes
                                  'output' : {0 : 'batch_size'}})
                    

if __name__ == "__main__":
    #export_ghost_onnx()
    export_arcface_onnx()