import os
import cv2
import tqdm
import re
from insightface.app import FaceAnalysis
import pandas as pd
import reorganize
import check_eye_pos



def detect_face_eye(frames_path):
    '''
    :param frames_path: path containing extracted frames from a video
    :return:
     1. face_frames: new path with the same frames augmented to bounding box face detection
     2. face_frames/eye_pos_relative_v1.txt: contains eye position for each face detected in new frames.
     3. face_frames/eye_pos_relative.txt: contains eye position for each face detected in new frames and sorted by image index.
    '''

    # Initialize the face analysis application
    app = FaceAnalysis()
    app.prepare(ctx_id=-1)  # Using CPU

    image_folder = frames_path  # Folder containing extracted frames
    output_folder = os.path.join(os.pardir, 'Video/face_frames')  # Folder to save detected faces
    #output_txt_file = './test/output_folder_small_20/eye_pos_original.txt'  # Output txt file for eye positions (original image)
    output_txt_file_cropped = os.path.join(os.pardir,'Video/face_frames/eye_pos_relative_v1.txt' ) # Output txt file for eye positions (cropped image)
    output_txt_file_cropped_sorted = os.path.join(os.pardir,'Video/face_frames/eye_pos_relative.txt') # Output txt file for eye positions (cropped image) sorted by index

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize empty lists to store the frame index and eye positions for the cropped images
    eye_positions_cropped = []

    # Process each image in the folder
    for image_file in tqdm.tqdm(os.listdir(image_folder)):
        img_path = os.path.join(image_folder, image_file)
        img = cv2.imread(img_path)

        # Check if the image was successfully loaded
        if img is None:
            print(f"Warning: Could not read image {image_file}. Skipping.")
            continue

        # Extract the index number from the image file name using regex
        match = re.search(r'(\d+)', image_file)
        if match:
            frame_idx = int(match.group(1))  # Extract the number part as frame index
        else:
            print(f"Warning: Could not extract index from filename {image_file}. Skipping.")
            continue

        # Detect faces
        faces = app.get(img)

        for idx, face in enumerate(faces):
            bbox = face['bbox']  # Get face bounding box (left, top, right, bottom)
            landmarks = face['kps']  # Get facial landmarks (key points)

            # Extract eye coordinates from landmarks
            left_eye = landmarks[0]  # Typically, the first landmark is the left eye
            right_eye = landmarks[1]  # The second landmark is the right eye

            # Crop the face from the bounding box while preserving the natural pose
            x1, y1, x2, y2 = map(int, bbox)
            cropped_face = img[y1:y2, x1:x2]

            # Resize the cropped face
            resized_face = cv2.resize(cropped_face, (256, 192))

            # Save cropped face with the correct name
           # output_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}face.bmp")
            # Extract the number part from the filename (assuming the format is frame_x.bmp)
            number_str = os.path.splitext(image_file)[0][6:]  # Extracts the number after 'frame_'
            # Strip leading zeros but preserve the number itself (no zero stripping for numbers like '102')
            number_str_no_padding = str(int(number_str))  # Convert to int to remove leading zeros, then back to string
            # Save cropped face with the new name: xface.bmp
            output_path = os.path.join(output_folder, f"{number_str_no_padding}face.bmp")
            cv2.imwrite(output_path, resized_face)

            # Calculate the eye positions relative to the cropped image (bounding box)
            left_eye_rel_x = (left_eye[0] - x1) / (x2 - x1) * 256
            left_eye_rel_y = (left_eye[1] - y1) / (y2 - y1) * 192
            right_eye_rel_x = (right_eye[0] - x1) / (x2 - x1) * 256
            right_eye_rel_y = (right_eye[1] - y1) / (y2 - y1) * 192

            # Append the frame index and relative eye positions to the cropped list
            eye_positions_cropped.append((frame_idx, left_eye_rel_x, left_eye_rel_y, right_eye_rel_x, right_eye_rel_y))

            #print(f"Cropped and saved face from {image_file} with original name preserved")
            #print(f"Saved eye positions for frame {frame_idx}")

    # Write the eye positions to the cropped image txt file
    with open(output_txt_file_cropped, 'w') as f:
        for pos in eye_positions_cropped:
            f.write(f"{pos[0]} {pos[1]} {pos[2]} {pos[3]} {pos[4]}\n")


    # Write the eye positions sorted by index to the cropped image sorted txt file
    data = pd.read_csv(output_txt_file_cropped, sep=' ', header=None)
    # Sort the data by the first column (index column)
    data_sorted = data.sort_values(by=0)
    # Save the sorted data to a new file
    data_sorted.to_csv(output_txt_file_cropped_sorted, sep=' ', header=False, index=False)

    return output_folder, output_txt_file_cropped_sorted

def main():
    ## Modify the path where the frames from the video are located:
    frames_path='../Video/frames'

    # Detect faces and eye positions
    output_folder, output_txt_file_cropped_sorted = detect_face_eye(frames_path)
    print(f"Face and eye positions detected. New frames stored in {output_folder}, and eye position stored in {output_txt_file_cropped_sorted}")

    # Optionally check that the cropped images correspond to the eye position detected:
    check_eyes_path = check_eye_pos.check(output_folder, output_txt_file_cropped_sorted)
    print(f"{check_eyes_path} created with the cropped images marked with eye positions")

    # Reorganize the input structure so it aligns with the required.
    reorganize.organize_images_and_write_txt(output_folder, output_txt_file_cropped_sorted, batch_size=10)
    reorganize.create_additional_files()
    print(f"Files reorganized.")

if __name__ == '__main__':
    main()





