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


## os.environ["CUDA_VISIBLE_DEVICES"]="2"

def csv_collator(samples):
    sample = samples[0]
    imgs=sample[0]
    eye_poses=sample[1]
    #blink_label=[]
    #blink_label.append(sample[2])
    for i in range(1,len(samples)):
      sample = samples[i]
      img=sample[0]
      eye_pos=sample[1]
      imgs=torch.cat((imgs,img),0)
      eye_poses=torch.cat((eye_poses,eye_pos),0)
      #blink_label.append(sample[2])
    #blink_labels=torch.stack(blink_label)
    return imgs,eye_poses

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

    # Load model checkpoints as before
    checkpoint_file = os.path.join(args.checkpoint_dir, args.eye_type, 'atten_generator.pth.tar')
    checkpoint_file2 = os.path.join(args.checkpoint_dir, args.eye_type, 'blink_eyelid_net.pth.tar')

    atten_generator.load_state_dict(torch.load(checkpoint_file, map_location=device)['state_dict'])
    blink_eyelid_net.load_state_dict(torch.load(checkpoint_file2, map_location=device)['state_dict'])

    atten_generator.eval()
    blink_eyelid_net.eval()

    predictions = np.zeros(len(test_loader.dataset) + cfg.time_size - 1)

    print('testing...')

    for i, (inputs, pos) in enumerate(tqdm(test_loader)):
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

            # Update predictions (maximum for overlapping windows)
            for j in range(len(predict)):
                idx = i + j
                predictions[idx] = max(predictions[idx], predict[j])

    print(f'Final predictions: {predictions}')
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
