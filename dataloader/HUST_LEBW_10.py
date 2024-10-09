import os
import numpy as np
import cv2
import torch
import re
import torch.utils.data as data
from utils.imutils import *
from utils.transforms import *

class HUST_LEBW(data.Dataset):
    def __init__(self, cfg, train=False):
        self.img_folder = cfg.root_path
        self.is_train = train
        self.inp_res = cfg.data_shape
        self.out_res = cfg.output_shape
        self.pixel_means = cfg.pixel_means
        self.num_class = cfg.num_class
        self.cfg = cfg
        self.bbox_extend_factor = cfg.bbox_extend_factor
        #self.eye = cfg.eye

        # Load and sort image filenames based on the numerical part before "face.bmp"
        self.image_files = sorted(
            [f for f in os.listdir(os.path.join(self.img_folder, 'check', 'images_input_test'))
             if f.endswith('face.bmp')],
            key=lambda x: int(re.search(r'\d+', x).group(0))
        )

        ## Load GT for computing the metrics afterward. To compare GT with test/train output results.
        #self.blink_gt_path = os.path.join(self.img_folder, 'check', f'gt_blink_{self.eye}.txt')
       # self.non_blink_gt_path = os.path.join(self.img_folder, 'check', f'gt_non_blink_{self.eye}.txt')

        #with open(self.blink_gt_path, "r") as blink_file:
        #    blink = blink_file.readlines()

        #with open(self.non_blink_gt_path, "r") as non_blink_file:
        #    non_blink = non_blink_file.readlines()

        #self.anno = blink + non_blink

        # Load eye positions
        eye_pos_path = os.path.join(self.img_folder, 'check/images_input_test', 'eye_pos_relative.txt')
        with open(eye_pos_path, "r") as pos_file:
            self.eye_positions = [line.strip().split() for line in pos_file.readlines()]

        # Ensure that the number of eye positions matches the number of images
        assert len(self.image_files) == len(self.eye_positions), "Mismatch between images and eye positions"

    def augmentationCropImage(self, img):
        height, width = self.inp_res[0], self.inp_res[1]
        img = cv2.resize(img, (width, height))
        return img

    def data_augmentation(self, img, leftmap, affrat, angle):

        leftmap = cv2.resize(leftmap, (192, 256))

        img = img.astype(np.float32) / 255

        left_eye = np.mean(np.where(leftmap == np.max(leftmap)), 1)

        return img, left_eye


    def __getitem__(self, index):
        # Ensure index + time_size doesn't exceed the number of available images
        if index + self.cfg.time_size > len(self.image_files):
            raise IndexError("End of dataset")

        images = []
        original_image_paths = []
        eye_positions = []

        # Load 10 consecutive images
        for i in range(index, index + self.cfg.time_size):
            img_filename = self.image_files[i]
            img_path = os.path.join(self.img_folder, 'check', 'images_input_test', img_filename)


            # Now load the image
            image = cv2.imread(img_path)
            if image is None:
                raise RuntimeError(f"Error reading image {img_path}")
            else:
                # Store the original image path
                original_image_paths.append(img_path)

            # Get eye position for the current image
            pos_line = self.eye_positions[i]
            pos_cur = [float(p) for p in pos_line]
            eye = 'right' ##Need to modify this
            eye_pos = [int(pos_cur[3] / image.shape[1] * 192), int(pos_cur[4] / image.shape[0] * 256)] if eye == 'right' else [int(pos_cur[1] / image.shape[1] * 192), int(pos_cur[2] / image.shape[0] * 256)]

            # Preprocess image
            img = self.augmentationCropImage(image)
            img = img[..., ::-1].copy()  # Convert BGR to RGB
            img = img.astype(np.float32) / 255
            img = im_to_torch(img)
            img = color_normalize(img, self.pixel_means)

            images.append(img)
            eye_positions.append(eye_pos)

        imgs = torch.stack(images)
        eye_positions_tensor = torch.tensor(eye_positions)

        return imgs, eye_positions_tensor, original_image_paths

    def __len__(self):
        return len(self.image_files) - self.cfg.time_size + 1
