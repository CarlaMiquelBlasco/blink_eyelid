import os
import torch
import torch.nn.parallel
import torch.optim
from config import cfg
from networks import atten_net
from networks.blink_eyelid_net_10 import BlinkEyelidNet
from dataloader.HUST_LEBW_10 import HUST_LEBW
import numpy as np
from tqdm import tqdm
import argparse
import cv2
from collections import defaultdict


## os.environ["CUDA_VISIBLE_DEVICES"]="2"

def csv_collator(samples):
    imgs = samples[0][0]  # The images
    eye_poses = samples[0][1]  # The eye positions
    img_paths = samples[0][2]  # The image paths
    #blink_label=[]
    #blink_label.append(sample[2])
    for i in range(1, len(samples)):
        imgs = torch.cat((imgs, samples[i][0]), 0)  # Concatenate the images
        eye_poses = torch.cat((eye_poses, samples[i][1]), 0)  # Concatenate the eye positions
        img_paths.extend(samples[i][2])  # Append the image paths

    return imgs, eye_poses, img_paths  # Return all three

def main(args):
    # Device setup remains unchanged
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("gpu")
    else:
        print("cpu")
        device = torch.device("cpu")


    # Load the models as before
    atten_generator = atten_net.__dict__[cfg.model](cfg.output_shape, cfg.num_class, cfg)
    atten_generator = torch.nn.DataParallel(atten_generator, device_ids=[0]).to(device)

    blink_eyelid_net = BlinkEyelidNet(cfg).to(device)
    blink_eyelid_net = torch.nn.DataParallel(blink_eyelid_net, device_ids=[0]).to(device)

    test_loader = torch.utils.data.DataLoader(
        HUST_LEBW(cfg, train=False),
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True, collate_fn=csv_collator, drop_last=False)
    print(len(test_loader))

    # Load model checkpoints as before
    checkpoint_file = os.path.join(args.checkpoint_dir, args.eye_type, 'atten_generator.pth.tar')
    checkpoint_file2 = os.path.join(args.checkpoint_dir, args.eye_type, 'blink_eyelid_net.pth.tar')

    atten_generator.load_state_dict(torch.load(checkpoint_file, map_location=device)['state_dict'])
    blink_eyelid_net.load_state_dict(torch.load(checkpoint_file2, map_location=device)['state_dict'])

    atten_generator.eval()
    blink_eyelid_net.eval()

    # Dictionary to store all predictions for each image index
    predictions_dict = defaultdict(list)

    # Create an array to store the maximum prediction for each image
    num_images = len(test_loader.dataset) + cfg.time_size - 1
    predictions = np.zeros(num_images, dtype=int)

    # Dictionary to store the eye positions and image paths for each image
    eye_positions_dict = defaultdict(lambda: None)  # Default to None
    image_paths_dict = defaultdict(lambda: None)  # Default to None

    print('testing...')

    for i, (inputs, pos, img_paths) in enumerate(tqdm(test_loader)):
        print(inputs.shape)
        with torch.no_grad():
            input_var = torch.autograd.Variable(inputs.to(device))
            global_outputs, refine_output = atten_generator(input_var)

            height = int(0.4 * refine_output.shape[2]) * 4
            width = height

            if args.eye_type == 'right':
                outputs, _ = blink_eyelid_net(input_var, height, width, pos.numpy(), torch.chunk(refine_output, 2, 1)[1], device)
            else:
                outputs, _ = blink_eyelid_net(input_var, height, width, pos.numpy(), torch.chunk(refine_output, 2, 1)[0], device)

            _, predicted = torch.max(outputs.data, 1)
            predict = predicted.cpu().numpy()

            #Update predictions and store eye positions and image paths
            for j in range(len(predict)):
                img_index = i + j
                predictions_dict[img_index].append(predict[j])  # Store all predictions for this image

                # Only update image paths and eye positions if they aren't already set
                if image_paths_dict[img_index] is None:
                    eye_positions_dict[img_index] = pos[j].cpu().numpy()
                    image_paths_dict[img_index] = img_paths[j]

    # Output the final predictions, images, and eye positions
    for img_index in range(num_images):
        if image_paths_dict[img_index] is not None:
            # Apply majority voting
            prediction_list = predictions_dict[img_index]
            final_prediction = 1 if prediction_list.count(1) > prediction_list.count(0) else 0

            print(f"Image: {image_paths_dict[img_index]}")
            print(f"Final Prediction: {'Blink' if final_prediction == 1 else 'No Blink'}")
            print(f"Eye Position: {eye_positions_dict[img_index]}")

        # Optionally, display the image using OpenCV
        #image = cv2.imread(image_paths_dict[img_index])
        #if image is not None:
        #    cv2.imshow(f"Image {img_index}", image)
        #    cv2.waitKey(0)  # Wait until a key is pressed before continuing

    print('Testing complete.')

    #print(f'Final predictions: {predictions}')
          #target=blink_label.data.numpy()
          #for (pre,tar) in zip(predict,target):

          #    if (abs(tar-1)<1e-5):
          #        blink_count+=1
          #        if (abs(pre-1)<1e-5):
          #            blink_right+=1
          #    else :
          #        unblink_count+=1
          #        if (abs(pre-0)<1e-5):
          #            unblink_right+=1

  #Recall=blink_right/(blink_count)
  #Precision=blink_right/(blink_right+unblink_count-unblink_right)
  #F1=2.0/(1.0/Recall+1.0/Precision)
  #print(f'{args.eye_type} eye: F1 = {F1}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='configs')
    parser.add_argument('-e', '--eye_type', default='right', type=str, metavar='N',
                        help='left or right eye (in image)')
    parser.add_argument('-c', '--checkpoint_dir', default='./pretrained_models',type=str, metavar='PATH',
                        help='path to load checkpoint (default: checkpoint)')
    args = parser.parse_args()
    os.chdir('/Users/carlamiquelblasco/Desktop/MASTER BERGEN/Q1/NONMANUAL/BLINK_EYELID/blink_eyelid/256.192.model')
    main(args)
