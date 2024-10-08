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
        self.eye = cfg.eye

        if train:
            self.scale_factor = cfg.scale_factor
            self.rot_factor = cfg.rot_factor
            self.symmetry = cfg.symmetry

        # Load image filenames from the images_input_test directory
        self.image_files = sorted(
            [os.path.join('check', 'images_input_test', re.sub(r'face', '', f))
             for f in os.listdir(os.path.join(self.img_folder, 'check', 'images_input_test'))
             if f.endswith('.bmp')],
            key=lambda x: int(re.search(r'\d+', x).group(0))
        )

        ## Load GT for computing the metrics afterward. To compare GT with test/train output results.
        self.blink_gt_path = os.path.join(self.img_folder, 'check', f'gt_blink_{self.eye}.txt')
        self.non_blink_gt_path = os.path.join(self.img_folder, 'check', f'gt_non_blink_{self.eye}.txt')

        with open(self.blink_gt_path, "r") as blink_file:
            blink = blink_file.readlines()

        with open(self.non_blink_gt_path, "r") as non_blink_file:
            non_blink = non_blink_file.readlines()
        # anno contains the whole GT (blink + no-blink).
        self.anno = blink + non_blink

        ## Load eye positions from the single eye_pos_relative.txt
        eye_pos_path = os.path.join(self.img_folder, 'check/images_input_test', 'eye_pos_relative.txt')
        with open(eye_pos_path, "r") as pos_file:
            self.eye_positions = [line.strip().split() for line in pos_file.readlines()]
        # Ensure the number of eye positions matches the number of images
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
        # Get the image path and read the image
        img_path = os.path.join(self.img_folder, self.image_files[index])
        img_path = img_path.strip('.bmp')
        img_path = img_path + 'face.bmp'
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Error reading image {img_path}")

        # Read the eye position for the current image
        pos_cur = self.eye_positions[index]
        pos_cur = [float(p) for p in pos_cur]

        # Calculate the eye position
        if self.cfg.eye == 'right':
            eye_pos = [int(float(pos_cur[3]) / image.shape[1] * 192), int(float(pos_cur[4]) / image.shape[0] * 256)]
        else:
            eye_pos = [int(float(pos_cur[1]) / image.shape[1] * 192), int(float(pos_cur[2]) / image.shape[0] * 256)]

        # Preprocess the image
        img = self.augmentationCropImage(image)
        img = img[..., ::-1].copy()  # Convert BGR to RGB
        img = img.astype(np.float32) / 255
        img = im_to_torch(img)
        img = color_normalize(img, self.pixel_means)

        # Return the image and corresponding eye position
        eye_pos_tensor = torch.tensor(eye_pos)
        return img, eye_pos_tensor


    def __len__(self):
        return len(self.image_files)
