import os
import torch
import torch.nn.parallel
import torch.optim
from config import cfg
from networks import atten_net
from networks.blink_eyelid_net_orig import BlinkEyelidNet
import numpy as np
from tqdm import tqdm
import csv
from PIL import Image
import torchvision.transforms as transforms

def read_eye_positions(eye_pos_file):
    eye_positions = {}
    with open(eye_pos_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            eye_positions[parts[0]] = np.array([float(p) for p in parts[1:]])
    return eye_positions

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 192)),  # resize to model input size
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # add batch dimension

def main():
    # Hardcoded paths (modify these as per your requirements)
    eye_type = 'left'  # or 'right' depending on the eye being processed
    checkpoint_dir = './pretrained_models'  # path to the folder with checkpoints
    image_dir =  os.path.join(os.pardir, 'Video/face_frames')  # folder where images are stored
    eye_pos_file = os.path.join(os.pardir,'Video/face_frames/eye_pos_relative.txt')  # file containing eye positions
    output_dir = os.path.join(os.pardir,'Metrics/predictions')  # folder where CSV will be saved

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load models
    atten_generator = atten_net.__dict__[cfg.model](cfg.output_shape, cfg.num_class, cfg)
    atten_generator = torch.nn.DataParallel(atten_generator, device_ids=[0]).to(device)

    blink_eyelid_net = BlinkEyelidNet(cfg).to(device)
    blink_eyelid_net = torch.nn.DataParallel(blink_eyelid_net, device_ids=[0]).to(device)

    # Load model checkpoints
    checkpoint_file = os.path.join(checkpoint_dir, eye_type, 'atten_generator.pth.tar')
    checkpoint_file2 = os.path.join(checkpoint_dir, eye_type, 'blink_eyelid_net.pth.tar')

    atten_generator.load_state_dict(torch.load(checkpoint_file, map_location=device)['state_dict'])
    blink_eyelid_net.load_state_dict(torch.load(checkpoint_file2, map_location=device)['state_dict'])

    atten_generator.eval()
    blink_eyelid_net.eval()

    print("Models loaded successfully")

    # Load eye positions from the text file
    eye_positions = read_eye_positions(eye_pos_file)

    # Prepare CSV file for writing predictions
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, 'predictions.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Blink Status'])

        # Process images in the folder
        for image_file in tqdm(os.listdir(image_dir)):
            if image_file.endswith(('.bmp', '.png')):
                image_path = os.path.join(image_dir, image_file)
                inputs = load_image(image_path).to(device)
                eye_pos = torch.tensor(eye_positions.get(image_file, [0, 0, 0, 0])).unsqueeze(0).to(device)

                with torch.no_grad():
                    global_outputs, refine_output = atten_generator(inputs)

                    height = int(0.4 * refine_output.shape[2]) * 4  # height = 100
                    width = height

                    if eye_type == 'right':
                        outputs, _ = blink_eyelid_net(inputs, height, width, eye_pos.cpu().numpy(), torch.chunk(refine_output, 2, 1)[1], device)
                    else:
                        outputs, _ = blink_eyelid_net(inputs, height, width, eye_pos.cpu().numpy(), torch.chunk(refine_output, 2, 1)[0], device)

                    _, predicted = torch.max(outputs.data, 1)
                    predict = predicted.data.cpu().numpy()[0]

                    # Determine blink status
                    blink_status = 'Blink' if predict == 1 else 'Non-Blink'

                    # Write to CSV
                    writer.writerow([image_file, blink_status])
                    print(f"Processed {image_file}: {blink_status}")

    print(f"Predictions saved to {csv_file}")

if __name__ == '__main__':
    main()
