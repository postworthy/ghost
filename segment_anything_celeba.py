import os
import sys
import cv2
import argparse
from insightface_func.face_detect_crop_single import Face_detect_crop
from pathlib import Path
from tqdm import tqdm
import insightface
from insightface.utils import face_align
from PIL import Image
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

def main(args):
    # Initialize SAM for segmentation
    sam = sam_model_registry["vit_h"](checkpoint="/app/sam_vit_h_4b8939.pth")
    sam.cuda()  # If using GPU
    #mask_generator = SamAutomaticMaskGenerator(sam)
    predictor = SamPredictor(sam)

    face_analyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_analyser.prepare(ctx_id=0, det_size=(640, 640))

    # Ensure the destination directory exists
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    # List all files in the dataset directory
    image_files = os.listdir(args.path_to_dataset)
    
    for image_name in tqdm(image_files):
        try:
            image_path = os.path.join(args.path_to_dataset, image_name)
            save_path = os.path.join(args.save_path, f"{image_name}")            

            if not os.path.exists(save_path):
                image = cv2.imread(image_path)
                final_face = None
                
                try:
                    face = face_analyser.get(image)[0]
                except Exception as e_3:
                    print(e_3)
                    face = None
                if face:
                    warped_img, _ = face_align.norm_crop2(image, face.kps, 256)
                    final_face = cv2.cvtColor(warped_img, cv2.COLOR_RGB2BGR)
                
                # After obtaining the final face...
                if final_face is not None:
                    # Convert PIL image to OpenCV format for SAM processing
                    final_face_cv = final_face
                    # Generate the mask
                    predictor.set_image(final_face_cv)
                    masks, scores, logits = predictor.predict(
                        point_coords=face.kps,
                        point_labels=np.array([1, 1, 1, 1, 1]),
                        multimask_output=False,
                    )
                    #masks = mask_generator.generate(final_face_cv)

                    # You can overlay the mask on the face, here is just a simple overlay example:
                    for i, mask in enumerate(masks):
                        cv2.imwrite(save_path, mask * 255)
                        # Create a colored mask for visualization
                        #colored_mask = np.zeros_like(final_face_cv)
                        #color = np.random.randint(0, 255, (3,), dtype=np.uint8)
                        #colored_mask[mask] = color
                        # Combine face image with the colored mask
                        #combined_image = cv2.addWeighted(final_face_cv, 0.7, colored_mask, 0.3, 0)
                        #output_image = Image.fromarray(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
                        #output_image = Image.fromarray(combined_image)

                    #if output_image:
                    #    output_image.save(save_path)

        except Exception as e:
            import traceback
            print(f"Error processing {image_name}: {e}")
            traceback.print_exc()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_dataset', default='/img_align_celeba', type=str)
    parser.add_argument('--save_path', default='/img_align_celeba_crop_masked', type=str)
    
    args = parser.parse_args()
    
    main(args)
