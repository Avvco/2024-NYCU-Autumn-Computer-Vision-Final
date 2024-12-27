import os
import cv2
import numpy as np
import pandas as pd
from lu_vp_detect import VPDetection


def process_image(image_path, calibration_path, length_thresh, seed):
    print(f"Processing image: {image_path}")
    
    calibration = pd.read_csv(calibration_path, delimiter=' ', header=None, index_col=0)
    P2 = np.array(calibration.loc['P2:']).reshape((3, 4))
    
    principal_point = (P2[0][2], P2[1][2])
    focal_length = P2[0][0]

    vpd = VPDetection(length_thresh, principal_point, focal_length, seed)

    vps = vpd.find_vps(image_path)
    print('Principal point: {}'.format(vpd.principal_point))

    print("The vanishing points in 3D space are: ")
    for i, vp in enumerate(vps):
        print(f"Vanishing Point {i + 1}: {vp}")

    vp2D = vpd.vps_2D
    print("\nThe vanishing points in image coordinates are: ")
    for i, vp in enumerate(vp2D):
        print(f"Vanishing Point {i + 1}: {vp}")

    debug_output_path = os.path.join("./output_images_KITTI", os.path.basename(image_path))
    vpd.create_debug_VP_image(False, debug_output_path)
    print(f"Debug image saved to: {debug_output_path}")

    debug_image = cv2.imread(debug_output_path)
    for i, (x, y) in enumerate(vp2D):
        if 0 <= x < debug_image.shape[1] and 0 <= y < debug_image.shape[0]:
            cv2.circle(debug_image, (int(x), int(y)), 10, (0, 0, 255), -1)  # Red dot for VP
            cv2.putText(debug_image, f"VP{i + 1}", (int(x) + 10, int(y) + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # Green label

    annotated_output_path = os.path.join("./output_images_KITTI", os.path.basename(image_path))
    os.makedirs("./output_images_KITTI", exist_ok=True)
    cv2.imwrite(annotated_output_path, debug_image)
    print(f"Annotated image saved to: {annotated_output_path}\n")

def main():
    image_dir = '/path_to_image_dir'
    calib_dir = '/path_to_calib_dir'
    
    length_thresh = 60
    seed = 1337

    os.makedirs("./output_images_KITTI", exist_ok=True)

    for image_file in os.listdir(image_dir):
        if image_file.endswith(".png"):
            image_path = os.path.join(image_dir, image_file)
            
            calib_file = os.path.splitext(image_file)[0] + ".txt"
            calib_path = os.path.join(calib_dir, calib_file)

            if os.path.exists(calib_path):
                process_image(image_path, calib_path, length_thresh, seed)
            else:
                print(f"Calibration file not found for image: {image_file}")

if __name__ == "__main__":
    main()