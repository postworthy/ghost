import os
import cv2
import insightface
import argparse
from insightface.utils import face_align
from PIL import Image

def main(args):
    # Initialize InsightFace
    face_analyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_analyser.prepare(ctx_id=0, det_size=(640, 640))

    image_directory = args.path_to_dataset
    male_directory = args.save_path_male
    female_directory = args.save_path_female

    if not os.path.exists(male_directory):
        os.makedirs(male_directory)
    if not os.path.exists(female_directory):
        os.makedirs(female_directory)

    # Process each image in the directory
    for filename in os.listdir(image_directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Add other file types if needed
            image_path = os.path.join(image_directory, filename)
            img = cv2.imread(image_path)

            # Use InsightFace for face detection and attributes on the resized image
            faces = face_analyser.get(img)

            for idx, face in enumerate(faces):
                gender = 'male' if face.gender == 1 else 'female'
                save_path = os.path.join(male_directory if gender == 'male' else female_directory, f'{filename}')
                warped_img, _ = face_align.norm_crop2(img, face.kps, 256)
                Image.fromarray(cv2.cvtColor(warped_img, cv2.COLOR_RGBA2BGR)).save(save_path)
                cv2.imwrite(save_path, img)
                break

    print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path_to_dataset', default='/img_align_celeba_crop', type=str)
    parser.add_argument('--save_path_male', default='/img_align_celeba_gender/male', type=str)
    parser.add_argument('--save_path_female', default='/img_align_celeba_gender/female', type=str)
    
    args = parser.parse_args()
    
    main(args)