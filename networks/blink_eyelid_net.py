# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import math
import numpy as np



class BlinkEyelidNet(nn.Module):
    def __init__(self, cfg):
        super(BlinkEyelidNet, self).__init__()
        self.upsam = nn.UpsamplingBilinear2d(scale_factor=4)
        self.hi_channel = 128
        self.batch_size = cfg.batch_size
        self.time_size = cfg.time_size
        self.conv1 = nn.Conv2d(3, 24, 5, stride=3, bias=False)
        self.pool1 = nn.MaxPool2d(2, 2, padding=0)
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 48, 3, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, 80, 3, stride=2, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(80)
        self.lstm = nn.LSTM(input_size=80 * 2, hidden_size=self.hi_channel, num_layers=2, dropout=0.5)
        self.fc6 = nn.Linear(self.hi_channel*2, 2,bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, image, height, width, pos, heatmap, device, phase='train'):
        datatemp = []
        img_height = image.shape[2]
        img_width = image.shape[3]
        heatall = self.upsam(heatmap)
        bbox = np.int64(pos)
        bbox[0] = np.int64(bbox[0] - height * 0.5)
        bbox[1] = np.int64(bbox[1] - width * 0.5)
        image = image.to(device)

        heatall = heatall.to(device)
        heat = torch.sigmoid(heatall[0])  # Process the single heatmap
        heat = heat.repeat(3, 1, 1)
        img = image * heat
        img_temp = torch.zeros(3, img_height + 100, img_width + 100)
        img_temp[:, 50:50 + img_height, 50:50 + img_width] = img
        img_process = img_temp[:, bbox[1]:bbox[1] + height, bbox[0]:bbox[0] + width]
        datatemp.append(img_process)

        x = torch.stack(datatemp).to(device)

        # CNN layers
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.pool1(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.pool1(out)

        # Process the image directly
        out = torch.squeeze(out)  # Remove unnecessary dimensions

        # Normalize the output
        out = torch.nn.functional.normalize(out, dim=0)

        # Create a "zero-difference" (or could be the same as out) for concatenation
        zero_diff = torch.zeros_like(out).to(device)

        # Concatenate the feature and zero-difference (or the same feature) to match LSTM's expected input size of 160
        lstm_input = torch.cat((out, zero_diff), dim=0)  # Resulting in [160]

        # Pass to LSTM
        lstm_input = lstm_input.unsqueeze(0).unsqueeze(1)  # Shape becomes [1, 1, 160]

        # Now LSTM will expect input size of 160
        outputs, _ = self.lstm(lstm_input)

        # Get the last hidden state from the LSTM (only one since sequence length = 1)
        h_state_1 = outputs[-1]  # Last hidden state (shape: [1, 128])

        # Since there's no second hidden state, you only pass h_state_1 (size: [1, 128])
        # To match the 256 size expected by fc6, we can replicate h_state_1
        h = torch.cat((h_state_1, h_state_1), dim=1)  # Shape [1, 256]

        # Final classification
        logits = self.fc6(h)

        return logits, h