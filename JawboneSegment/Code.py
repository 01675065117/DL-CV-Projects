from datetime import datetime
import numpy as np
import cv2
import math
import time
# --- Apply Homomorphy Filter Algorithm---


def HomomorphicFilter(img, yh=5, yl=4.5, cutoff=40, c=1):
    """
    Homomorphic Filter for image
    """
    # Convert img from int8 to float32. Because OpenCV require float32
    img_float = np.float32(img)

    # Normalize Img to calculate faster
    img_float_normalized = img_float / 255

    rows, cols, dims = img.shape

    # Convert from BGR to YCrCb
    img_YCrCb = cv2.cvtColor(img_float_normalized, cv2.COLOR_BGR2YCrCb)

    # Split the img to 3 color chanel (Y, Cr, Cb). Y represent for Gray level
    y, cr, cb = cv2.split(img_YCrCb)
    #cv2.imshow('Y',y)

    
    # Get Logarit space
    y_log = np.log(y + 0.01)  # Add 0.01 because to avoid log(0)

    # Apply Fourier Transform
    fft = np.fft.fft2(y_log)

    fft_shift = np.fft.fftshift(fft)

    Dx = cols / cutoff
    G = np.ones((rows, cols))

    for i in range(rows):
        for j in range(cols):
            D_square = (i - rows/2)**2 + (j - cols/2)**2
            G[i][j] = (yh - yl)*(1 - np.exp(-c*(D_square/(2*(Dx**2))))) + yl

    result_filter = G * fft_shift

    # Inverse Fourier Transform
    fft_ishift = np.fft.ifftshift(result_filter)
    ifft = np.fft.ifft2(fft_ishift)

    # Get x from (x + yi) (complex)
    result_interm = np.real(ifft)

    # Inverse Logarit Space
    result_img = np.exp(result_interm)
    
    cv2.imshow('Homo',result_img)
    return result_img


# --- Apply Constrast Enhancement (CLAHE)---
def ConstrastEnhancement(img, clipLimit=2, tileGrid=7):
    clahe = cv2.createCLAHE(clipLimit=clipLimit,
                            tileGridSize=(tileGrid, tileGrid))
    
    img_enhanced = clahe.apply(img)
    #img_enhanced = cv2.equalizeHist(imgGray)
    return img_enhanced


# Apply Adaptive Median Filter
# Append image with padding size
def padding(img, pad):
    padded_img = np.zeros((img.shape[0]+2*pad, img.shape[1] + 2*pad))
    print(img.shape[0])
    padded_img[pad:-pad, pad:-pad] = img
    return padded_img


def level_A(mat, x, y, s, sMax):
    while True:
        window = mat[x - (s//2): x+(s//2) + 1, y - (s//2): y + (s//2) + 1]

        Zmin = np.min(window)
        Zmed = np.median(window)
        Zmax = np.max(window)

        A1 = Zmed - Zmin
        A2 = Zmed - Zmax

        if A1 > 0 and A2 < 0:
            return level_B(window, Zmin, Zmed, Zmax)
        else:
            s += 2
            if s <= sMax:
                continue
            else:
                return Zmed


def level_B(window, Zmin, Zmed, Zmax):

    h, w = window.shape
    Zxy = window[h//2, w//2]

    B1 = Zxy - Zmin
    B2 = Zxy - Zmax

    if B1 > 0 and B2 < 0:
        return Zxy
    else:
        return Zmed


def AdaptiveMedianFilter(img, s=3, sMax=7):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    H, W = img.shape

    a = sMax // 2
    padded_img = padding(img, a)

    f_img = np.zeros(padded_img.shape)

    for i in range(a, H + a + 1):
        for j in range(a, W + a + 1):
            value = level_A(padded_img, i, j, s, sMax)
            f_img[i, j] = value

    return f_img[a:-a, a:-a]


def DrawRectToImage(img, pt1, pt2, color):
    cv2.rectangle(img, pt1, pt2, color, 2)
    return img


def KMeans(img, x, y, w, h, k=3):
    img_ROI = img[y: y + h, x: x + w]

    img_reshaped = img_ROI.reshape((-1, 1))
    img_reshaped = np.float32(img_reshaped)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    _, labels, centers = cv2.kmeans(
        img_reshaped, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(img_ROI.shape)

    unique_value = np.unique(segmented_image)
    min_value = (unique_value[1] - unique_value[0])/2

    segmented_image = np.where(segmented_image > min_value + 0.1, 0, 1)
    cv2.imshow('Canny', np.float32(segmented_image))
    return segmented_image


def DetectCEJLocation(seg_img):
    canny = np.ones((seg_img.shape[0], seg_img.shape[1], 3))

    border_up = []
    border_down = []

    H, W = seg_img.shape
    for i in range(W):
        for j in range(H):
            if seg_img[j][i] == 0:
                canny[j][i] = (1, 0, 0)  # RGB
                border_up.append((i, j))
                break

    for i in range(W):
        for j in range(H - 1, 0, -1):
            if seg_img[j][i] == 0:
                border_down.append((i, j))
                break

    # draw up-down
    for point_x, point_y in zip(border_up, border_down):
        cv2.line(canny, (point_x[0], point_x[1] + 1), point_y, (255, 0, 0), 1)

    max_j = {'distance': 0,
             'start': (0, 0),
             'end': (0, 0)}  # (distance, start, end)

    # draw up
    start = border_up[0]
    for point in border_up:
        end = point
        cv2.line(canny, start, end, (0, 0, 255), 1)

        absolute_j = math.sqrt(
            (start[0] - end[0])**2 + (start[1] - end[1])**2)
        if absolute_j > max_j['distance']:
            max_j['distance'] = absolute_j
            max_j['start'] = start
            max_j['end'] = end
        start = end

    return max_j


def Timing(func):
    def wrapper():
        begin = time.time()
        func()
        end = time.time()
        print('Time: ', end - begin)

    return wrapper

### MAIN ####


@Timing
def main():
    img = cv2.imread('bone1.PNG')
    homo_img = HomomorphicFilter(img, yh=2.5, yl=1.8, cutoff=100, c=3)
    cv2.imshow('homo_img',homo_img)
    #cons_img = ConstrastEnhancement(homo_img)
    #cv2.imshow(cons_img)
    amf_imf = AdaptiveMedianFilter(homo_img, s=3, sMax=11)
    cv2.imshow('AdaptiveMedianFilter',amf_imf)
    # Large Rectangle Box
    LX, LY, LW, LH = 40, 40, 180, 180
    
    # Medium Rectangle Box
    delta = 20
    MX = LX + delta
    MY = LY + delta
    MW = LW - 2*delta
    MH = LH - 2*delta

    # Small Rectangle BOX
    delta = 20
    SX = MX + delta
    SY = MY + delta
    SW = MW - 2*delta
    SH = MH - 2*delta

    # Current Rectangle to Get CEJ location
    x, y, w, h = SX, SY, SW, SH#LX,LY,LW,LH

    kmeans_img = KMeans(amf_imf, x, y, w, h, k=2)
    #cv2.imshow('kmeans',kmeans_img)
    max_j = DetectCEJLocation(kmeans_img)
    max_point = max_j['start'] if max_j['start'][1] > max_j['end'][1] else max_j['end']

    # print(max_j)

    img_brg = cv2.cvtColor(np.float32(amf_imf), cv2.COLOR_GRAY2BGR)
    cv2.circle(img_brg, (x + max_point[0], y + max_point[1]), 3, (0, 0, 255), -1)

    #Large Box
    cv2.rectangle(img_brg, (LX, LY), (LX + LW, LY + LH),
                  (0, 0, 255), 2)

    #Medium Box
    cv2.rectangle(img_brg, (MX, MY), (MX + MW, MY + MH),
                  (0, 255, 0), 2)

    #Small Box
    cv2.rectangle(img_brg, (SX, SY), (SX + SW, SY + SH),
                  (255, 255, 0), 2)

    cv2.imshow('Final', img_brg)
    cv2.waitKey(0)
    cv2.destroyAllWindows

main()