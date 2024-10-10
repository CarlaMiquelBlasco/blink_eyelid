# BlinkEyelidNet Model Overview

## 1. **Model Structure**

The `BlinkEyelidNet` model is designed to process a sequence of images (a group of consecutive frames) and predict a binary classification (blink or unblink) based on **temporal** information. The model is composed of the following key components:

- **Convolutional Layers**:
    - Three 2D convolutional layers (`conv1`, `conv2`, `conv3`) are used to extract features from the input images. Each layer is followed by batch normalization (`bn1`, `bn2`, `bn3`), ReLU activations, and max pooling layers.

- **Upsampling Layer**:
    - An upsampling layer is applied to the heatmap input, which is used for attention-like processing by modulating the image features.

- **LSTM Layer**:
    - The core of the model is a 2-layer LSTM (Long Short-Term Memory) network that processes the extracted features from the convolutional layers in a sequential manner. This allows the model to capture temporal dependencies between frames.

- **Fully Connected Layer**:
    - The LSTM’s output (hidden states) is passed to a fully connected layer (`fc6`) that produces the final logits for binary classification.

## 2. **Forward Function Breakdown**

The `forward` method defines how data moves through the network:

1. **Preprocessing**:
    - Each image is processed with its corresponding heatmap to emphasize specific regions of interest. The image is padded and cropped based on bounding box (`pos`) information.

2. **Convolutional Feature Extraction**:
    - The images are passed through the convolutional layers, and feature maps are extracted.

3. **Temporal Processing**:
    - The output from the convolutional layers is reshaped into a sequence format, suitable for the LSTM. The model calculates temporal differences between consecutive frames, and this difference is concatenated with the original feature maps.

4. **LSTM Processing**:
    - The LSTM processes the feature maps and outputs hidden states representing temporal dependencies between the images in the sequence.

5. **Classification**:
    - The final hidden states of the LSTM are concatenated and passed through a fully connected layer, producing the final logits for classification.

## 3. **Required Input**

The model expects the following inputs:

- **image**: A **batch** of images in sequence, with each image being a frame from a video.

- **height, width**: These values represent the dimensions of the bounding boxes applied during image preprocessing (256X192).

- **pos**: Bounding box positions for each image, used to extract specific regions of the images.

- **heatmap**: A heatmap tensor that is upsampled and used to modulate the input images.

- **device**: The device on which the model operates (CPU, MPS or GPU).

- **time_size**: The number of images (frames) expected in the input sequence. This defines how many frames the model processes at once.

### **Note**:
The model is designed to process a **sequence** of images. The number of images in this sequence is defined by `time_size`, and the LSTM expects multiple frames to capture temporal dependencies.

## 4. **Why a Single Image as Input Causes an Error**

If a single image is provided as input instead of a sequence, it causes an error in the following line:

```python
inputs = out.reshape(-1, self.time_size, 80).transpose(1, 0)
```

Reason:

1. **Reshaping the Output**:
    - The line `out.reshape(-1, self.time_size, 80)` reshapes the output from the convolutional layers into a tensor with dimensions `(-1, time_size, 80)`, where `time_size` is the number of frames in the sequence, and `80` is the number of channels.

2. **Expectation of a Sequence**:
    - If only a single image is provided, `time_size` will be greater than 1, but there is only one set of features. This causes a **dimension mismatch** when the model attempts to reshape the output. Specifically, the model expects a 3D tensor (batch size, sequence length, channels), but with one image, it cannot create a valid sequence.

3. **Transpose Operation**:
    - After reshaping, the model transposes the sequence to switch the time and batch dimensions (`transpose(1, 0)`), which further assumes multiple time steps (frames). With a single image, this transposition would not work as intended, leading to further errors or crashes.

### **Error**:
The reshape operation raise the following error:
```
RuntimeError: shape '[-1, time_size, 80]' is invalid for input of size X
```
This indicates that the input size is too small to be reshaped into the expected sequence format.

#### Reason Why `time_size = 1` Will Cause an Error:

Setting `time_size = 1` in the current model would likely raise an error or lead to unexpected behavior due to several reasons related to how the model is structured to process **sequences of images** and perform **temporal difference calculations**.

1. **Temporal Difference Calculation**:
   In the `forward` function, the model calculates the difference between consecutive frames (features from images) before passing them to the LSTM:

   ```python
   feature = inputs[1:10, :, :]
   differ = feature - inputs[0:9, :, :]
   inputs = torch.cat((feature, differ), 2)
   ```

    - **`inputs[1:10, :, :]`**: This extracts features from the 2nd to 10th time steps (images) in the sequence.
    - **`inputs[0:9, :, :]`**: This extracts features from the 1st to 9th time steps.
    - **`feature - inputs[0:9, :, :]`**: The model calculates the difference between consecutive frames to capture changes over time.

   With `time_size = 1`, there is only one time step (i.e., a single image), so there is no second time step to perform the difference calculation. Attempting to index beyond the first image (`inputs[1:10, :, :]` or `inputs[0:9, :, :]`) would raise an **indexing error** because there is only one time step in the input sequence.

2. **LSTM Sequence Processing**:
   The LSTM layer is designed to process a sequence of features over time. By setting `time_size = 1`, the input sequence effectively has only one frame, which negates the need for the LSTM's temporal processing. While technically the LSTM can process a sequence of length 1, it makes the entire temporal sequence modeling aspect of the LSTM **irrelevant**. We can prove this by running test.py which is the code modified to accept only one image as input. The output from this are wrong predicitons.
