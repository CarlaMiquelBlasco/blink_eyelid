# README: Instructions for Running the Code (original code)

This guide outlines the step-by-step process to run the provided code, including frame extraction, preprocessing, and testing. Follow these instructions carefully to ensure the code runs as expected.

## 1. Convert Video to Raw Frames (25 Frames per Second)

First, convert your input video into individual frames at a rate of 25 frames per second using `ffmpeg`. Create a new folder for the frames and then extract them using the following command:

```bash
# Create a new folder to store frames
mkdir Frames

# Convert video to frames (25 frames per second)
ffmpeg -i input_video.mp4 -vf fps=25 output_frames/frame_%04d.png
```

## 2. Preprocess Frames and Organize Input Data

Next, preprocess the extracted frames and organize them into the required directory structure.

### Steps:
1. Open `preprocess_clean.py` and update the `frames_path` variable inside the `main()` function to point to the directory where your frames are located.
2. Run the preprocessing script:

```bash
python preprocess_clean.py
```

## 3. Validate the Preprocessing Output

Check that the last folder inside `Data/test/check/blink/` contains the correct number of images (`time_size`). If the folder does not contain the expected number of images, delete this folder.

## 4. Run the Testing Script

Once the preprocessing is complete, you can run the test script with the following command:

```bash
python test_orig.py
```

---

## Detailed Overview of `preprocess_clean.py`

The `preprocess_clean.py` script handles the following tasks:

1. **Face and Eye Detection:**
    - Detect faces in the original frames using the `insightface` library.
    - Crop the images based on the bounding boxes of the detected faces.
    - Detect eye positions within the cropped images and generate the file `eye_pos_relative_v1.txt` containing this information.

2. **Generate and Organize Eye Position Data:**
    - Create `eye_pos_relative.txt`, which is an ordered version of `eye_pos_relative_v1.txt`. This file is used by the model.
    - Optionally, verify that the detected eye positions align with the cropped images. This step will create a new folder with the cropped images, annotated with eye position markers.

3. **Organize Cropped Images:**
    - Group the cropped images into sets of ten and relocate them into the following path: `Data/test/check/blink/X/10`, where `X` is an increasing integer index.
    - Along with the images, place the corresponding eye position file (`eye_pos_relative.txt`) in each folder.

4. **Create `gt_blink.txt` File:**
    - Generate a `gt_blink.txt` file that contains the paths to all images. Although this file is named as if it represents ground truth data, it is required for the code to run correctly. It doesn't contain any GT, but only the path to the images.

5. **Create Additional Required Files:**
    - Create a new empty folder: `Data/test/check/unblink`.
    - In the `Data/test/check` directory, create the following files:
        - `gt_blink_left.txt` (copy content from `gt_blink.txt`)
        - `gt_blink_right.txt` (copy content from `gt_blink.txt`)
        - `gt_blink.txt` (copy content from `gt_blink.txt`)
        - `gt_non_blink_left.txt` (leave empty)
        - `gt_non_blink_right.txt` (leave empty)
        - `gt_non_blink.txt` (leave empty)

---

## Final Directory Structure

After running the preprocessing script, the `Data/test/check/` directory should have the following structure:

```
Data/test/check/
│
├── blink/
│   ├── 1/
│   │   └── 10/ (contains cropped images and eye_pos_relative.txt)
│   ├── 2/
│   │   └── 10/ (contains cropped images and eye_pos_relative.txt)
│   └── ... (and so on)
│
├── unblink/ (empty)
├── gt_blink.txt
├── gt_blink_left.txt
├── gt_blink_right.txt
├── gt_non_blink.txt (empty)
├── gt_non_blink_left.txt (empty)
└── gt_non_blink_right.txt (empty)
```

Ensure this structure is in place before proceeding with the testing script.