import os
import torch
import torch.nn.parallel
import torch.optim
from config import cfg
from networks import atten_net
from networks.blink_eyelid_net_orig import BlinkEyelidNet
from dataloader.HUST_LEBW_orig import HUST_LEBW
import numpy as np
from tqdm import tqdm
import argparse
import csv


#os.environ["CUDA_VISIBLE_DEVICES"]="2"

def csv_collator(samples):
    sample = samples[0]
    imgs=sample[0]
    eye_poses=sample[1]
    blink_label=[]
    blink_label.append(sample[2])
    for i in range(1,len(samples)):
        sample = samples[i]
        img=sample[0]
        eye_pos=sample[1]
        imgs=torch.cat((imgs,img),0)
        eye_poses=torch.cat((eye_poses,eye_pos),0)
        blink_label.append(sample[2])
    blink_labels=torch.stack(blink_label)
    return imgs,eye_poses,blink_labels
def main(args):
    # Device setup
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

    cfg.eye = args.eye_type
    test_loader = torch.utils.data.DataLoader(
        HUST_LEBW(cfg, train=False),
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True, collate_fn=csv_collator, drop_last=False)

    # Load model checkpoints as before
    checkpoint_file = os.path.join(args.checkpoint_dir, args.eye_type, 'atten_generator.pth.tar')
    checkpoint_file2 = os.path.join(args.checkpoint_dir, args.eye_type, 'blink_eyelid_net.pth.tar')

    atten_generator.load_state_dict(torch.load(checkpoint_file, map_location=device)['state_dict'])
    blink_eyelid_net.load_state_dict(torch.load(checkpoint_file2, map_location=device)['state_dict'])


    print("=> loaded checkpoint '{}'".format(checkpoint_file))

    atten_generator.eval()
    blink_eyelid_net.eval()
    print('testing...')

    #blink_count = 0
    #unblink_count = 0
    #blink_right = 0
    #unblink_right = 0
    # Open a CSV file for writing the predictions
    with open('predictions.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Path', 'Prediction'])  # Write CSV header


        for i, (inputs, pos, blink_label) in enumerate(tqdm(test_loader)):

            with torch.no_grad():
                input_var = torch.autograd.Variable(inputs.to(device))

                global_outputs, refine_output = atten_generator(input_var)  # refineout:(b*t, 2, 64, 48)

                height = np.int64(0.4*refine_output.shape[2])*4 # height = 100
                width = height
                if args.eye_type == 'right':
                    outputs, b = blink_eyelid_net(input_var, height, width, pos.numpy(), torch.chunk(refine_output, 2, 1)[1], device)
                else:
                    outputs, b = blink_eyelid_net(input_var, height, width, pos.numpy(), torch.chunk(refine_output, 2, 1)[0], device)

                _, predicted = torch.max(outputs.data, 1)
                predict=predicted.data.cpu().numpy()
                print(f"path {i+1}, prediction: {predict}")
                writer.writerow([f"path {i+1}", predict[0]])
                #target=blink_label.data.numpy()
                #for (pre,tar) in zip(predict,target):

                #    if (abs(tar-1)<1e-5):
                #      blink_count+=1
                #      if (abs(pre-1)<1e-5):
                #          blink_right+=1
                #    else :
                #      unblink_count+=1
                #      if (abs(pre-0)<1e-5):
                #          unblink_right+=1

        #Recall=blink_right/(blink_count)
        #Precision=blink_right/(blink_right+unblink_count-unblink_right)
        #F1=2.0/(1.0/Recall+1.0/Precision)
        #print(f'{args.eye_type} eye: F1 = {F1}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='configs')
    parser.add_argument('-e', '--eye_type', default='left', type=str, metavar='N',
                        help='left or right eye (in image)')
    parser.add_argument('-c', '--checkpoint_dir', default='./pretrained_models',type=str, metavar='PATH',
                        help='path to load checkpoint (default: checkpoint)')
    args = parser.parse_args()
    os.chdir('/Users/carlamiquelblasco/Desktop/MASTER BERGEN/Q1/NONMANUAL/BLINK_EYELID/blink_eyelid/256.192.model')
    main(args)
