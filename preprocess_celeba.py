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

def main(args):
    face_analyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_analyser.prepare(ctx_id=0, det_size=(640, 640))

    # Ensure the destination directory exists
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    # List all files in the dataset directory
    image_files = os.listdir(args.path_to_dataset)
    
    for image_name in tqdm(image_files):
        try:
            image_path = os.path.join(args.path_to_dataset, image_name)
            save_path = os.path.join(args.save_path, image_name)

            if not os.path.exists(save_path):
                image = cv2.imread(image_path)
                final_face=None
                if args.max_darken == True:
                    for i in range(33, 255):
                        brightness_factor = 1 - (i / 255.0)
                        brightness_factor = max(brightness_factor, 0)
                        image = image.astype(np.float32) / 255.0
                        image = (np.clip(image * brightness_factor, 0, 1) * 255).astype(np.uint8)
                        try:
                            face = face_analyser.get(image)[0]
                        except Exception as e_2:
                            #print(i)
                            #print(e_2)
                            break
                        if face:
                            warped_img, _ = face_align.norm_crop2(image, face.kps, 256)
                            final_face = Image.fromarray(cv2.cvtColor(warped_img, cv2.COLOR_RGBA2BGR))
                        else:
                            break
                else:
                    try:
                        face = face_analyser.get(image)[0]
                    except Exception as e_3:
                        print(e_3)
                        face = None
                    if face:
                        warped_img, _ = face_align.norm_crop2(image, face.kps, 256)
                        final_face = Image.fromarray(cv2.cvtColor(warped_img, cv2.COLOR_RGBA2BGR))
                
                if final_face != None:
                    final_face.save(save_path)

        except Exception as e_1:
            print(e_1)
            pass
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--max_darken', default=False, type=bool)
    parser.add_argument('--path_to_dataset', default='/img_align_celeba', type=str)
    parser.add_argument('--save_path', default='/img_align_celeba_crop', type=str)
    parser.add_argument('--max_images_per_dir', default=500000, type=int)
    
    args = parser.parse_args()
    
    main(args)
