import os
import cv2
import math
import numpy as np

# Find vanishing point's coordinate on image
# Origin point (0, 0) of image is at upper left.

def filtering(lines):

    filtered = []
    for line in lines:
        [[x1, y1, x2, y2]] = line

        if x1 != x2:
            m = (y2-y1)/(x2-x1) # tan(theta)

            theta = math.atan(m) # radian [-pi/2, pi/2]
            theta = math.degrees(theta) # degree [-90 degree, 90 degree]

            if 10 <= theta <= 80:
                c = y1 - m*x1
                l = math.sqrt((y2-y1)**2 + (x2-x1)**2)
                filtered.append([x1, y1, x2, y2, m, c, l])

    if len(filtered) > 13:
        filtered = sorted(filtered, key=lambda x: x[6], reverse=True)
        filtered = filtered[:13]

    return filtered


def intesection(filtered_lines, img):

    VP = None
    min_err = 100000000000

    for i in range(len(filtered_lines)):
        for j in range(i+1, len(filtered_lines)):
            # line 1
            # y = m1*x + c1
            m1 = filtered_lines[i][4]
            c1 = filtered_lines[i][5]

            # line 2
            # y = m2*x + c2
            m2 = filtered_lines[j][4]
            c2 = filtered_lines[j][5]


            if m1 != m2: # not parellel
                # interaction point (x0, y0)
                x0 = (c1-c2) / (m2-m1)
                y0 = m1*x0 + c1

                # The interaction point should be within the frame.
                if 0 < x0 < img.shape[1] and 0 < y0 < img.shape[0]:

                    # distance of (x0, y0) and all other lines
                    err = 0
                    for k in range(len(filtered_lines)):
                        m = filtered_lines[k][4]
                        c = filtered_lines[k][5]

                        m_ = -1/m
                        c_ = y0 - m_*x0
                        x_ = (c-c_) / (m_-m)
                        y_ = m_ * x_ + c_

                        l = math.sqrt((y_-y0)**2 + (x_-x0)**2)
                        err += l**2

                    err = math.sqrt(err)

                    if err < min_err:
                        min_err = err
                        VP = [x0, y0]

    return VP


def find_VP(img):

    # Step 1: gray-scale conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: edge detection
    # low_threshold:high_threshold = 1：2 or 1：3, recommended by John Canny (algo. author)
    edges = cv2.Canny(gray, 100, 300)

    # Step 3: line detection
    # smaller threshold, more straight lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 110, 10, 15)
    if lines is None:
        return None, None

    # Step 4: line filtering
    filtered_lines = filtering(lines)
    if filtering is None:
        return None, None

    # Step 5: intesection points of any two lines
    VP = intesection(filtered_lines, img)
    if VP is None or VP[0]>=img.shape[1] or VP[1]>=img.shape[0]:
        return None, None

    return VP, filtered_lines

# Annotate the vanishing point with a red dot/bounding box if needed
def draw_VP(img, vp):

    if vp is not None:
        x, y = vp
        cv2.circle(img, (int(x), int(y)), 10, (0, 0, 255), -1)

        #box_size = 10
        #x1 = int(x - box_size/2)
        #y1 = int(y - box_size/2)
        #x2 = int(x + box_size/2)
        #y2 = int(y + box_size/2)
        #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # thickness=2

    return img

def draw_VP_with_lines(img, vp, lines):
    """
    Draw the vanishing point and the detected lines on the image.
    """
    img_with_lines = img.copy()
    
    # Draw all the lines
    for line in lines:
        x1, y1, x2, y2, _, _, _ = line
        cv2.line(img_with_lines, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Blue lines

    # Draw the vanishing point
    if vp is not None:
        x, y = vp
        cv2.circle(img_with_lines, (int(x), int(y)), 10, (0, 0, 255), -1)  # Red dot for VP

    return img_with_lines

def process_images(input_folder, output_folder):
    """
    Process all images in the input folder to detect vanishing points and lines.
    Save the results in the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        img = cv2.imread(image_path)

        if img is None:
            print(f"Failed to read {image_path}")
            continue

        print(f"Processing {image_file}, size: {img.shape[1]}*{img.shape[0]}")
        vp, filtered_lines = find_VP(img)

        if vp is not None and filtered_lines is not None:
            img_VP_with_lines = draw_VP_with_lines(img, vp, filtered_lines)

            output_path = os.path.join(output_folder, f"{image_file}")
            cv2.imwrite(output_path, img_VP_with_lines)
            print(f"Saved processed image to {output_path}")

        else:
            print(f"No vanishing point detected for {image_file}")

input_folder = '/path_to_image_dir'
output_folder = '/path_to_output_dir'

process_images(input_folder, output_folder)