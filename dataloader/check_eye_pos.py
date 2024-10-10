import os
import cv2
import re
import tqdm

def check(input_folder, eye_position_file):
    '''

    :param input_folder: Folder containing the cropped images
    :param eye_position_file: The eye position txt file
    :return: images with marked eyes
    '''

    output_folder = os.path.join(os.pardir, 'Video/face_frames_eyes_marked')  # Folder to save the images with marked eyes

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load eye positions from the txt file
    eye_positions = []
    with open(eye_position_file, 'r') as f:
        for line in f.readlines():
            # Each line contains: frame_idx left_eye_x left_eye_y right_eye_x right_eye_y
            parts = line.strip().split()
            frame_idx = int(parts[0])
            left_eye_x = float(parts[1])
            left_eye_y = float(parts[2])
            right_eye_x = float(parts[3])
            right_eye_y = float(parts[4])
            eye_positions.append((frame_idx, left_eye_x, left_eye_y, right_eye_x, right_eye_y))

    # Process each image and mark the eyes
    for image_file in tqdm.tqdm(os.listdir(input_folder)):
        img_path = os.path.join(input_folder, image_file)
        img = cv2.imread(img_path)

        # Extract the index number from the image file name using regex
        match = re.search(r'(\d+)', image_file)
        if match:
            frame_idx = int(match.group(1))  # Extract the number part as frame index
        else:
            print(f"Warning: Could not extract index from filename {image_file}. Skipping.")
            continue

        # Find the corresponding eye positions for the current frame index
        eye_position = next((pos for pos in eye_positions if pos[0] == frame_idx), None)
        if eye_position is None:
            print(f"Warning: No eye positions found for frame {frame_idx}. Skipping.")
            continue

        # Extract eye positions
        left_eye_x, left_eye_y = int(eye_position[1]), int(eye_position[2])
        right_eye_x, right_eye_y = int(eye_position[3]), int(eye_position[4])

        # Mark the eyes on the image (draw circles)
        eye_radius = 5  # You can adjust the radius
        left_eye_color = (0, 255, 0)  # Green color for the left eye
        right_eye_color = (0, 0, 255)  # Red color for the right eye
        thickness = 2

        # Draw circles at the eye positions
        cv2.circle(img, (left_eye_x, left_eye_y), eye_radius, left_eye_color, thickness)
        cv2.circle(img, (right_eye_x, right_eye_y), eye_radius, right_eye_color, thickness)

        # Save the image with the marked eyes
        output_path = os.path.join(output_folder, image_file)
        if img is not None:
            cv2.imwrite(output_path, img)

    return output_folder
