from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob
import math
from scipy import interpolate
from PIL import Image, ImageDraw
from scipy import ndimage
import cv2
import math
import copy


team_members_names = ['إسراء ياسر ابوالقاسم',
                      'ايمان علاء فرج كامل',
                      'منة الله مصطفى مصطفى عوض',
                      'منة محيي الدين محمود',
                      'ميرنا محمد يسري']
team_members_seatnumbers = ['2016170080',
                            '2016170113'
                            '2016170437',
                            '2016170438',
                            '2016170450']

# region Hardcoded values
video_names = ['White Lane', 'Yello Lane']
roi_indices = np.array([[[130, 540], [900, 539], [550, 329], [419, 329]]], 'int32')  # BL BR UR UL
# endregion


# region Region of Interest
def extract_roi(img, roi_indices):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, roi_indices, [255, 255, 255])
    return cv2.bitwise_and(mask, img)
# endregion


# region Color Conversion and Thresholding
def convert_rgb_to_grayscale(img):
    # This function will do color transform from RGB to Gray
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def convert_rgb_to_hsv(img):
    # This function will do color transform from RGB to HLS
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)


def color_thresholding(img, low_threshold, high_threshold):
    # define the fixed values of the pixels
    strong = 255
    outimg = np.zeros((img.shape[0],img.shape[1]))

    # extract the indices of the pixels per condition for each channel
    in_i, in_j, in_ch = np.where((low_threshold <= img & (img <= high_threshold)))
    irrelevant_i, irrelevant_j, irrelevant_ch = np.where((img < low_threshold) | (img > high_threshold))

    # update the indices with the corresponding pixel value
    outimg[in_i, in_j] = strong
    outimg[irrelevant_i, irrelevant_j] = 0

    return outimg
# endregion


# region Canny
def plotting(img, fig_num, title):
    """
    Helper Function, to reduce the redundant plotting code
    """
    plt.figure(fig_num)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    return


def gaussian_kernel(sigma=0.5):
    # Specify the kernel size
    if sigma > 1:
        size = 5
    else:
        size = 3

    # Calculate the formula
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = normal * np.exp(-((x**2 + y**2) / (2.0*sigma**2)))

    return g


def convolve(input_arr, weights, c_type='conv', mode='reflect'):
    """
    Two-dimensional convolution.

    The array is convolved with the given kernel.

    Parameters
    -----------
    input_arr : array_like
        Input array to filter.

    weights : array_like
        Array of weights(kernel), same number of dimensions as input.

    c_type : {'conv', 'loop'}, optional
        The `c_type` parameter determines the implementation of the
        convolution.

        'conv'(default)
            Use built-in convolve.

        'loop'
            Use implemented loop.

    mode : {'constant', 'reflect'}, optional
        The `mode` parameter determines how the array borders are
        handled.

        'constant'
            Pads with a constant value.

        'reflect'(default)
            Pads with the reflection of the vector mirrored on the first and last values of the vector along each axis.

    Returns
    -------
    img : ndarray
        The result of convolution of `input` with `weights`.
    """
    img = copy.deepcopy(input_arr)
    if c_type == 'loop':
        window_size = weights.shape[0]
        offset = window_size // 2
        # Pad the working image
        padded_img = np.pad(img, offset, mode=mode)
        height = padded_img.shape[0]
        width = padded_img.shape[1]
        img_temp = np.zeros((height, width))
        # The filter algorithm
        for i in range(offset, height - offset):
            for j in range(offset, width - offset):
                img_temp[i, j] = np.sum(
                    np.multiply(padded_img[i - offset: i + offset + 1, j - offset: j + offset + 1], weights))
        img = img_temp[offset:height - offset, offset:width - offset]
    else:
        img = ndimage.filters.convolve(input_arr, weights, mode=mode)
    return img


def noise_reduction(img, c_type='conv', mode='reflect'):
    kernel = gaussian_kernel(1.4)
    return convolve(img, kernel, c_type=c_type, mode=mode)


def gradient_calculation(img, c_type='conv', mode='reflect'):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    Ix = convolve(img, Kx, c_type=c_type, mode=mode)
    Iy = convolve(img, Ky, c_type=c_type, mode=mode)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    G = G.astype(int)
    return G, theta


def non_max_suppression(img, theta):
    height, width = img.shape
    out = np.zeros((height, width), dtype=int)
    # Change from radian to degree
    angle = theta * 180 / np.pi
    angle[angle < 0] += 180

    for i in range(1, height-1):
        for j in range(1, width-1):
            point_1 = 255
            point_2 = 255

            # Angle 0°
            # gradient magnitude is greater than the magnitudes at pixels in the (east and west) directions
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                point_1 = img[i, j-1]
                point_2 = img[i, j+1]
            # Angle 45°
            # gradient magnitude is greater than the magnitudes at pixels in the (north east and south west) directions.
            elif 22.5 <= angle[i, j] < 67.5:
                point_1 = img[i-1, j+1]
                point_2 = img[i+1, j-1]
            # Angle 90°
            # gradient magnitude is greater than the magnitudes at pixels in the (north and south) directions.
            elif 67.5 <= angle[i, j] < 112.5:
                point_1 = img[i-1, j]
                point_2 = img[i+1, j]
            # Angle 135°
            # gradient magnitude is greater than the magnitudes at pixels in the (north west and south-east) directions.
            elif 112.5 <= angle[i, j] < 157.5:
                point_1 = img[i-1, j-1]
                point_2 = img[i+1, j+1]

            # if the pixel's value is the maximum then update the out, otherwise leave it with value = 0
            if (img[i, j] >= point_1) and (img[i, j] >= point_2):
                out[i, j] = img[i, j]
    return out


def double_thresholding(img, low_threshold=2, high_threshold=25):
    # define the fixed values of the pixels
    weak = 25
    strong = 255

    # extract the indices of the pixels per condition
    strong_i, strong_j = np.where(img >= high_threshold)
    weak_i, weak_j = np.where((low_threshold <= img) & (img <= high_threshold))
    irrelevant_i, irrelevant_j = np.where(img < low_threshold)

    # update the indices with the corresponding pixel value
    img[strong_i, strong_j] = strong
    img[weak_i, weak_j] = weak
    img[irrelevant_i, irrelevant_j] = 0

    return img, weak, strong


def hysteresis_edge_tracking(img, weak, strong):
    height, width = img.shape
    # Get the indices of the weak pixels
    weak_i, weak_j = np.where(img == weak)
    for (i, j) in zip(weak_i, weak_j):
        if i == 0 or j == 0 or i == height - 1 or j == width - 1:
            continue
        # Check if any of the  8-connected neighborhood pixels is a strong pixel
        if np.any(np.where(img[i-1:i+2, j-1:j+2] == strong)):
            img[i, j] = strong
        # If not, then suppress the weak pixel
        else:
            img[i, j] = 0
    return img


def detect_edges_canny(img, low_threshold=2, high_threshold=50, c_type='conv', mode='reflect'):
    """
    Arguments:
        :param img: np.ndarray((height, width)), gray-image
        :param low_threshold: an integer
        :param high_threshold: an integer
        :param c_type: str in ['conv', 'loop']
        :param mode: str in ['reflect', 'constant']
    :return: edges image
    """
    img = img.copy()

    # 1. Noise reduction
    # using Gaussian Smoothing Kernel
    img = noise_reduction(img, c_type=c_type, mode=mode)
    #plotting(img, 1, 'Gaussian Filter')

    # 2. Gradient Calculations
    # using Sobel kernels
    img, theta = gradient_calculation(img, c_type=c_type, mode=mode)
    #plotting(img, 2, 'Sobel Filter (G)')

    # 3. Non-Maximum Suppression
    img = non_max_suppression(img, theta)
    #plotting(img, 3, 'Non-Maximum Suppression')

    # 4. Double Thresholding
    img, weak, strong = double_thresholding(img, low_threshold, high_threshold)
    #plotting(img, 4, 'Double Thresholding')

    # 5. Edge Tracking by Hysteresis
    img = hysteresis_edge_tracking(img, weak, strong)
    #plotting(img, 5, 'Edge Tracking by Hysteresis')

    #plt.show()
    return img
# endregion


# region Remove Guassian Noise
def remove_noise(img, kernel_size, width, height):
    # Implementation of Mean Filter
    final_image = img.copy()
    mask_one_dim = math.sqrt(kernel_size)
    offset = int(mask_one_dim // 2)
    for i in range(offset, height - offset):
        for j in range(offset, width - offset):
            sum = 0
            for r in range(i - offset, i + offset + 1):
                for c in range(j - offset, j + offset + 1):
                    sum += img[r][c]
            final_image[i][j] = sum // kernel_size
    return final_image
# endregion


# region Masking Image
def mask_image(img, vertices, width, height):
    # Mask out the pixels outside the region defined in vertices (set the color to black)
    out_img = img.copy();
    '''for i in range(0, height):
        for j in range(0, width):
            if vertices[i][j] == 0:
                out_img[i][j] = 0
    return out_img'''
    return cv2.bitwise_and(out_img, vertices)
# endregion


# region Hough Transform
def hough_transform(img, accepted_ratio=0.1, rho_step=1, theta_step=1):
    """
    Apply Hough transform to find the lanes
    Input:
      img - numpy array: binary image containing the edges after applying Canny Detector
    Output:
      lines - list of list: list of lines, each line = [x1, y1, x2, y2]
    """
    # Calculating diagonal length of the image to define the range of the rhos
    height, width = img.shape
    diagonal_length = round(math.sqrt(height ** 2 + width ** 2))
    rhos = np.arange(-diagonal_length, diagonal_length, step=rho_step)

    # Setting the thetas and their cosines/sines
    thetas = np.arange(-90.0, 90.0, step=theta_step)
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))

    accumulator = np.zeros((len(rhos), len(thetas)))

    edge_pts_ys, edge_pts_xs = np.nonzero(img)
    for i in range(len(edge_pts_xs)):
        x = edge_pts_xs[i]
        y = edge_pts_ys[i]
        for theta_idx in range(len(thetas)):
            rho = x * cos_thetas[theta_idx] + y * sin_thetas[theta_idx]
            rho_idx = np.argmin(np.abs(rhos - rho))  # find the index of the closest rho to the computed rho
            accumulator[rho_idx, theta_idx] += 1

    # lines_rhos = []
    # lines_thetas = []
    lines_rhos_indices, lines_thetas_indices = np.nonzero(accumulator >= accumulator.max() * accepted_ratio)
    lines = []
    for i in range(len(lines_rhos_indices)):
        rho_idx = lines_rhos_indices[i]
        rho = rhos[rho_idx]
        # lines_rhos.append(rho)

        theta_idx = lines_thetas_indices[i]
        theta = thetas[theta_idx]
        # lines_thetas.append(theta)

        a = cos_thetas[theta_idx]
        b = sin_thetas[theta_idx]
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * a)
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * a)
        lines.append([x1, y1, x2, y2])
    return lines


def draw_hough_lines(img, lines, path, color=[255, 0, 0], thickness=8):
    d = ImageDraw.Draw(img)
    for line in lines:
        #d.line(line, fill=(255, 255, 255), width=thickness)
        d.line(line, fill=(color[0], color[1], color[2]), width=thickness)
    #img.save(path)
    return img


def hough_test():
    input_image = read_image('Images/hough_test_1.jpg')
    edges = detect_edges_canny(input_image, 2, 50)
    lines = hough_transform(edges, 0.3)
    draw_hough_lines(Image.open('Images/hough_test_1.jpg'), lines, 'Images/hough_test_1_out.jpg')

    input_image = read_image('Images/hough_test_2.jpg')
    edges = detect_edges_canny(input_image, 2, 50)
    lines = hough_transform(edges)
    draw_hough_lines(Image.open('Images/hough_test_2.jpg'), lines, 'Images/hough_test_2_out.jpg')
    return
# endregion


# region Getting Lane Lines
def get_line_length(line):
    x1, y1, x2, y2 = line
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_line_slope_intercept(line):
    """
    This function returns the slope (m) and the intercept (b) of the input line.
    y = mx + b
    """
    x1, y1, x2, y2 = line
    if x2 - x1 == 0:
        return math.inf, 0
    slope = (y2 - y1) / (x2 - x1)
    # b = y - mx
    intercept = y1 - slope * x1
    return slope, intercept


def get_avg_slope_intercept(lines):
    """
    The upper left corner of the image is (0, 0)
    X increases left-right
    Y increases top-down
    Left line: as Y increases, X decreases, so it has a negative slope
    Right line: as Y increase, X increases, so it has a positive slope

    Input: a list of lines, where each line = [x1, y1, x2, y2]
    Output: avg_of_left_lines, avg_of_right_lines, each is a tuple(avg_slope, avg_intercept)
    """
    right_lines = []
    right_lengths = []
    left_lines = []
    left_lengths = []
    for line in lines:
        slope, intercept = get_line_slope_intercept(line)
        if slope == math.inf or (slope >= -0.4 and slope <= 0.4):
            continue
        line_length = get_line_length(line)
        if slope < 0:  # left line
            left_lines.append([slope, intercept])
            left_lengths.append(line_length)
        else:  # right line
            right_lines.append([slope, intercept])
            right_lengths.append(line_length)

    # Weighted average of all right lines
    if len(right_lines) > 0:
        avg_of_right_lines = np.dot(right_lengths, right_lines) / np.sum(right_lengths)
    else:
        avg_of_right_lines = None

    # Weighted average of all left lines
    if len(left_lines) > 0:
        avg_of_left_lines = np.dot(left_lengths, left_lines) / np.sum(left_lengths)
    else:
        avg_of_left_lines = None

    return avg_of_right_lines, avg_of_left_lines


def get_line_endpoints(y1, y2, line):
    """
    Input: y1, y2, line in the form (slope, intercept)
    Output: line in the form [x1, y1, x2, y2]
    y = mx + b, x = (y - b) / m
    """
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    y1 = int(y1)
    x2 = int((y2 - intercept) / slope)
    y2 = int(y2)
    return [x1, y1, x2, y2]


def get_lane_lines(img, lines):
    """
    Hough returns a lot of lines for one lane line, so we average them to have one line per lane line.
    """
    avg_of_left_lines, avg_of_right_lines = get_avg_slope_intercept(lines)
    y1 = img.shape[0]
    y2 = img.shape[0] * 0.59
    left_lane = get_line_endpoints(y1, y2, avg_of_left_lines)
    right_lane = get_line_endpoints(y1, y2, avg_of_right_lines)
    return left_lane, right_lane


def draw_lines_connected(img, hough_lines, color=[0, 0, 255], thickness=8):
    """
    This function should draw lines to the images (default color is red and thickness is 8)
    """
    lane_lines = get_lane_lines(img, hough_lines)
    lines_image = np.zeros_like(img)
    for line in lane_lines:
        if line is None:
            continue
        x1, y1, x2, y2 = line
        point_one = (line[0], line[1])
        point_two = (line[2], line[3])
        cv2.line(lines_image, point_one, point_two, color, thickness)
    final_image = cv2.addWeighted(img, 0.8, lines_image, beta=0.95, gamma=0)
    #cv2.imshow("image with lines", final_image)
    #cv2.waitKey(0)
    return final_image
# endregion


# region Input/Output
def read_image(img_path):
    return mpimg.imread(img_path)


def read_video(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    return video, fps


def get_video_frames(video):
    frames = []
    width = int(video.get(3))
    height = int(video.get(4))
    number_of_frames = int(video.get(7))
    for i in range(number_of_frames):
        success, frame = video.read()
        if success:
            frames.append(frame)
    return frames, width, height


def write_video(pathOut, frames, fps, width, height):
    size = (width, height)
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for frame in frames:
        out.write(frame)
    out.release()
# endregion


# region Pipelines
def image_pipeline(image, width, height, video_frame_flag=True):
    if not video_frame_flag:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image

    # extract region of interest
    image = extract_roi(image, roi_indices)

    hsv_img = convert_rgb_to_hsv(image)

    # 3. convert to Gray
    gray_img = convert_rgb_to_grayscale(image)

    # 4. Threshold HSV for Yellow and White (combine the two results together)

    # 4.1 Threshold HSV for Yellow
    lower = np.uint8([10, 0, 100])
    upper = np.uint8([40, 255, 255])
    yellow_mask = color_thresholding(hsv_img, lower, upper)

    # 4.1 Threshold HSV for White
    lower = np.uint8([0, 200, 0])
    upper = np.uint8([200, 255, 255])
    white_mask = color_thresholding(hsv_img, lower, upper)

    # 4.1 Combine the resuls together
    image = cv2.bitwise_or(white_mask, yellow_mask)
    image = image.astype("int32")

    # 5. Mask the gray image using the threshold output from step 4
    # masked_gray_img = mask_image(gray_img, mask, width, height)

    # 6. Apply noise remove (gaussian) to the masked gray image
    # masked_gray_img_no_noise = remove_noise(image, 3, width, height)

    # 7. use canny detector and fine tune the thresholds (low and high values)
    canny_out_img = detect_edges_canny(image, low_threshold=10, high_threshold=50)

    # 8. mask the image using the canny detector output
    masked_with_canny = mask_image(image, canny_out_img, width, height)

    # 9. apply hough transform to find the lanes
    hough_lines = hough_transform(masked_with_canny)

    # 10. apply the pipeline you developed to the challenge videos
    image_with_lines = draw_lines_connected(original_image, hough_lines)

    return image_with_lines


def video_pipeline(video_frames, width, height, fps, name):
    new_video_frames = []
    for frame in tqdm(video_frames):
        new_frame = image_pipeline(frame, width, height)
        new_video_frames.append(new_frame)
    write_video('Videos/' + name + " Output.mp4", new_video_frames, fps, width, height)
    print("Video saved succesffuly")
# endregion


# region MAIN
for video_name in video_names:
    print(video_name)
    video, fps = read_video('Videos/' + video_name + '.mp4')
    video_frames, width, height = get_video_frames(video)
    video_pipeline(video_frames, width, height, fps, video_name)
# endregion
