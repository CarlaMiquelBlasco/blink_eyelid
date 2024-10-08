import os
import torch
import torch.nn.parallel
import torch.optim
from config import cfg
from networks import atten_net
from networks.blink_eyelid_net import BlinkEyelidNet
from dataloader.HUST_LEBW import HUST_LEBW
import numpy as np
from tqdm import tqdm
import argparse

def csv_collator(samples):
    sample = samples[0]
    imgs = sample[0]
    eye_poses = sample[1]
    #blink_label=[]
    #blink_label.append(sample[2])
    #print(sample)
    for i in range(1, len(samples)):
        sample = samples[i]
        img = sample[0]
        eye_pos = sample[1]
        imgs = torch.cat((imgs, img), 0)
        eye_poses = torch.cat((eye_poses, eye_pos), 0)
    return imgs, eye_poses

def main(args):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("gpu")
    else:
        print("cpu")
        device = torch.device("cpu")

    # initialize the model with the provided parameters, and assigns it to atten_generator
    ## Default parameters: model=CPN18 // output_shape=(64, 48) // num_class=2
    atten_generator = atten_net.__dict__[cfg.model](cfg.output_shape, cfg.num_class, cfg)
    ## wraps the atten_generator model with torch.nn.DataParallel, which allows it to be parallelized over multiple GPUs if available (not in mac)
    atten_generator = torch.nn.DataParallel(atten_generator, device_ids=[0]).to(device)
    # Initializes another model, blink_eyelid_net, by creating an instance of the BlinkEyelidNet class. Parameters set in configuration
    blink_eyelid_net = BlinkEyelidNet(cfg).to(device)
    ## wraps the atten_generator model with torch.nn.DataParallel, which allows it to be parallelized over multiple GPUs if available (not in mac)
    blink_eyelid_net = torch.nn.DataParallel(blink_eyelid_net, device_ids=[0]).to(device)

    cfg.eye = args.eye_type
    test_loader = torch.utils.data.DataLoader(
        HUST_LEBW(cfg, train=False),
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True, collate_fn=csv_collator, drop_last=False)

    checkpoint_file = os.path.join(args.checkpoint_dir, args.eye_type, 'atten_generator.pth.tar')
    checkpoint_file2 = os.path.join(args.checkpoint_dir, args.eye_type, 'blink_eyelid_net.pth.tar')
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    atten_generator.load_state_dict(checkpoint['state_dict'])
    checkpoint2 = torch.load(checkpoint_file2, map_location=torch.device('cpu'))
    blink_eyelid_net.load_state_dict(checkpoint2['state_dict'])
    print("=> loaded checkpoint '{}'".format(checkpoint_file))

    atten_generator.eval()
    blink_eyelid_net.eval()
    print('testing...')

    for i, (inputs, pos) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            input_var = torch.autograd.Variable(inputs.to(device).unsqueeze(0))
           # print("inputs: ", inputs)
            global_outputs, refine_output = atten_generator(input_var)

            height = np.int64(0.4 * refine_output.shape[2]) * 4
            width = height
            #print("-"*100)
            #print("pos.numpy()", pos.numpy())
            #print("-"*100)
            #print(pos.numpy())
            if args.eye_type == 'right':
                outputs, b = blink_eyelid_net(input_var, height, width, pos.numpy(), torch.chunk(refine_output, 2, 1)[1], device)
            else:
                outputs, b = blink_eyelid_net(input_var, height, width, pos.numpy(), torch.chunk(refine_output, 2, 1)[0], device)

            _, predicted = torch.max(outputs.data, 1)
            predict = predicted.data.cpu().numpy()

            # Here you can add evaluation logic or print predictions
            print(f'Predictions: {predict}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='configs')
    parser.add_argument('-e', '--eye_type', default='right', type=str, metavar='N',
                        help='left or right eye (in image)')
    parser.add_argument('-c', '--checkpoint_dir', default='./pretrained_models', type=str, metavar='PATH',
                        help='path to load checkpoint (default: checkpoint)')
    args = parser.parse_args()
    main(args)
