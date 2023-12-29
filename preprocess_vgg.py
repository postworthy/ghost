import os
import sys
import cv2
import argparse
from insightface_func.face_detect_crop_single import Face_detect_crop
from pathlib import Path
from tqdm import tqdm
import insightface

def prune_directory(directory, keep_count):
    for file in sorted(os.listdir(directory))[keep_count:]:
        os.remove(os.path.join(directory, file))

def main(args):
    face_analyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_analyser.prepare(ctx_id=0, det_size=(640, 640))
    border_size = 100

    dirs = os.listdir(args.path_to_dataset)
    for i in tqdm(range(len(dirs))):
        d = os.path.join(args.path_to_dataset, dirs[i])
        dir_to_save = os.path.join(args.save_path, dirs[i])

        if not Path(dir_to_save).exists() or len(os.listdir(dir_to_save)) < args.max_images_per_dir:
            Path(dir_to_save).mkdir(parents=True, exist_ok=True)
            skip_count = len(os.listdir(dir_to_save))

            image_names = os.listdir(d)
            for image_name in image_names[skip_count:skip_count+args.max_images_per_dir]:
                try:
                    image_path = os.path.join(d, image_name)
                    save_path = os.path.join(dir_to_save, image_name)

                    if not os.path.exists(save_path):
                        image = cv2.imread(image_path)
                        image_bordered = cv2.copyMakeBorder(image, top=border_size, bottom=border_size, left=border_size, right=border_size, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
                        face = face_analyser.get(image_bordered)[0]
                        if face:
                            cv2.imwrite(save_path, image)
                except:
                    pass
        elif Path(dir_to_save).exists() and len(os.listdir(dir_to_save)) > args.max_images_per_dir:
            prune_directory(dir_to_save, args.max_images_per_dir)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path_to_dataset', default='/VggFace2', type=str)
    parser.add_argument('--save_path', default='/VggFace2-crop', type=str)
    parser.add_argument('--max_images_per_dir', default=10, type=int)
    
    args = parser.parse_args()
    
    main(args)
