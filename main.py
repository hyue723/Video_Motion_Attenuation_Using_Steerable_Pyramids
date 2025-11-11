"""
Yue Xu
yxu7@caltech.edu
June 3rd, 2020
CS101c Final Project - Motion Attenuation Using Steerable Pyramids
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

Data_Folder = "./data/"
Result_Folder = "./results/"

#------------------------------------------------------------------------------
# Main functions 

"""
the wrapper function that performs motion attenuation described in the report
window_size: must be odd and must be smaller than total frames or max_num_frames
fastYIQ: if True, process only Y channel in YIQ color space; 
        if False, process all 3 RGB channels
max_num_frames: if not zero, limit maximum number of frames to process
"""
def processVideoMedianPhase(video_filepath,
                               result_filepath,
         window_size = 11, # must be odd number, smaller than total frames
         fastYIQ = True,
        max_num_frames = 0,
        verbose = False):
    # load video and its parameters
    video, fps, fourcc = readVideoMatrix(video_filepath, 
                                         max_num_frames = max_num_frames)
    num_frames, height, width, num_channels = video.shape
    r_pi, theta_pi = polarFrequencyCoordinates(width, height, 
                             xmin = -np.pi, xmax = np.pi, 
                             ymin = -np.pi, ymax = np.pi)
    
    # construct filters
    K = 4 # number of sub-bands for each level
    N = maxLevel(height, width)
    filters, filter_ranges = computeFilters(r_pi, theta_pi, 
                            K, N, verbose = False)
    _, _, total_elements = pyramidSize(filters)
    if verbose:
        print(f"there are total {num_frames} frames")
        print(f"each frame has height={height}, width={width}, channels={num_channels}")
        
    # write new video
    new_video = cv2.VideoWriter(result_filepath, 
                                fourcc,
                               fps,
                               (width, height))
    
    for frame_idx in range(num_frames):
        if verbose:
            print(f"process frame #{frame_idx}")
        if frame_idx == 0:
            assert(window_size//2 < num_frames)
            start_window = 0
            curr_window_size = window_size//2 + 1
            end_window = start_window + curr_window_size
            if fastYIQ:
                frames_pyr = np.zeros((curr_window_size, total_elements),
                                     dtype = 'complex128')
            else:
                frames_pyr = np.zeros((num_channels, curr_window_size, 
                                       total_elements),
                                     dtype = 'complex128')
            for i in range(start_window, end_window):
                frame = video[i]#(height,width,3)
                if fastYIQ:
                    frame_YIQ = rgbToYIQ(frame)#(height,width,3)
                    frames_pyr[i, :] = steerPyramid(frame_YIQ[:, :, 0], 
                                        filters, filter_ranges,
                                       collapse = True)
                else:
                    for chan in range(num_channels):
                        frames_pyr[chan, i, :] = steerPyramid(
                                        frame[:, :, chan], 
                                        filters, filter_ranges,
                                       collapse = True)
        else:
            start_window = np.max([frame_idx - window_size//2, 0])
            # non-inclusive end
            end_window = np.min([frame_idx + window_size//2 + 1, num_frames]) 
            curr_window_size = end_window - start_window
            overlap_start_offset = start_window - prev_start
            overlap_size = prev_window_size - overlap_start_offset
            if fastYIQ:
                frames_pyr = np.zeros((curr_window_size, total_elements),
                                     dtype = 'complex128')
                frames_pyr[:overlap_size, :] = prev_frames_pyr[
                                            overlap_start_offset:, :]
            else:
                frames_pyr = np.zeros((num_channels, curr_window_size, 
                                       total_elements),
                                     dtype = 'complex128')
                frames_pyr[:, :overlap_size, :] = prev_frames_pyr[:, 
                                                overlap_start_offset:, :]
            for i in range(overlap_size, curr_window_size):
                frame = video[start_window + i] #(height,width,3)
                if fastYIQ:
                    frame_YIQ = rgbToYIQ(frame)#(height,width,3)
                    frames_pyr[i, :] = steerPyramid(frame_YIQ[:, :, 0], 
                                        filters, filter_ranges,
                                       collapse = True)
                else:
                    for chan in range(num_channels):
                        frames_pyr[chan, i, :] = steerPyramid(
                                        frame[:, :, chan], 
                                        filters, filter_ranges,
                                       collapse = True)
        # update parameters for next layer
        prev_frames_pyr = frames_pyr
        prev_start = start_window
        prev_window_size = curr_window_size
        
        # compute the new frame with median phase
        if fastYIQ:
            curr_frame_pyr = frames_pyr[frame_idx - start_window, :]
            curr_frame_abs = np.absolute(curr_frame_pyr)
            frames_pyr_phase = frames_pyr/np.absolute(frames_pyr)
            curr_frame_medianphase = np.median(frames_pyr_phase, axis = 0)
            curr_frame_pyr_new = curr_frame_medianphase*curr_frame_abs
            curr_frame_recon = np.real(reconstruct(curr_frame_pyr_new, 
                                     filters, filter_ranges, 
                                     fft = False,
                                    collapse = True))
            curr_frame_orig = video[frame_idx]
            curr_frame_YIQ = rgbToYIQ(curr_frame_orig)
            curr_frame_YIQ[:, :, 0] = curr_frame_recon
            curr_frame_new = YIQTorgb(curr_frame_YIQ)
            new_video.write(convertToInt(curr_frame_new))
        else:
            curr_frame_pyr = frames_pyr[:, frame_idx - start_window, :]
            curr_frame_abs = np.absolute(curr_frame_pyr)
            
            frames_pyr_phase = frames_pyr/np.absolute(frames_pyr)
            curr_frame_medianphase = np.median(frames_pyr_phase, axis = 1)
            curr_frame_pyr_new = curr_frame_medianphase*curr_frame_abs
            curr_frame_new = np.zeros((height, width, num_channels),
                                       dtype = np.float64)
            for chan in range(num_channels):
                curr_frame_new[:, :, chan] = np.real(reconstruct(
                                     curr_frame_pyr_new[chan, :], 
                                     filters, filter_ranges, 
                                     fft = False,
                                    collapse = True))
            new_video.write(convertToInt(curr_frame_new))

        if frame_idx%10 == 0:
            print(f"processed {frame_idx}/{num_frames} frames")
    new_video.release()


"""
run to create figures used in the reports
add figures to a folder under Result_Folder named "figures"
"""
def createReportFigures():
    video, _, _ = readVideoMatrix(Data_Folder+"moon.avi", max_num_frames = 10)

    Image.fromarray(convertToInt(video[0])).save(Result_Folder + 
                                                 "figures/moon.png")

    image = rgbToYIQ(video[0])[:, :, 0]
    Image.fromarray(convertToInt(image)).save(Result_Folder + 
                                              "figures/moon_Y.png")

    image_fft = np.fft.fftshift(np.fft.fft2(image))
    image_fft_show = np.log(np.abs(image_fft))

    Image.fromarray(convertToInt(image_fft_show/np.max(image_fft_show))).save(Result_Folder + 
                                                 "figures/moon_Y_fft.png")

    height, width = image.shape
    r_pi, theta_pi = polarFrequencyCoordinates(width, height, 
                                 xmin = -np.pi, xmax = np.pi, 
                                 ymin = -np.pi, ymax = np.pi)

    K = 4 # number of sub-bands for each level
    N = maxLevel(height, width)
    filters, filter_ranges = computeFilters(r_pi, theta_pi, 
                                K, N, verbose = False)

    Image.fromarray(convertToInt(filters[0])).save(Result_Folder + 
                                                 "figures/H0.png")

    for i in range(1, len(filters) - 1):
        Image.fromarray(convertToInt(filters[i])).save(Result_Folder + 
                            f"figures/B_{(i-1)//4}_{(i-1)%4}.png")

    Image.fromarray(convertToInt(filters[-1])).save(Result_Folder + 
                                                 "figures/L_residual.png")

    pyramids = steerPyramid(image, filters, filter_ranges, collapse = False)

    Image.fromarray(convertToInt(np.real(pyramids[0]
                            )/np.max(np.real(pyramids[0])))).save(
                            Result_Folder + 
                            "figures/pyr_H0_adjust.png")

    for i in range(1, 9):
        Image.fromarray(convertToInt(np.real(pyramids[i]
                            )/np.max(np.real(pyramids[i])
                            ))).save(
                            Result_Folder + 
                            f"figures/pyr_B_{(i-1)//4}_{(i-1)%4}_adjust.png")

    for i in range(9, len(pyramids) - 1):
        Image.fromarray(convertToInt(np.real(pyramids[i]))).save(
                            Result_Folder + 
                            f"figures/pyr_B_{(i-1)//4}_{(i-1)%4}.png")

    Image.fromarray(convertToInt(np.real(pyramids[-1]))).save(
                            Result_Folder + 
                            "figures/pyr_L_residual.png")

#------------------------------------------------------------------------------
# utility functions

"""
convert an image from int data type in [0, 255] to float64 in [0, 1]
"""
def convertToFloat(im):
    return np.float64(im/255)
"""
convert an image from float64 data type in [0, 1] to int in [0, 255]
"""
def convertToInt(im):
    return np.uint8(255 * im.clip(0, 1))

"""
convert a 3d array image (with values in [0, 1]) from RGB to YIQ color space
"""
def rgbToYIQ(im):
    r = im[:, :, 0]
    g = im[:, :, 1]
    b = im[:, :, 2]
    y = 0.299*r + 0.587*g + 0.114*b
    i = 0.5959*r  - 0.2746*g - 0.3213*b
    q = 0.2115*r  - 0.5227*g + 0.3112*b
    return np.stack([y, i, q], axis = 2)

"""
convert a 3d array image (with values in [0, 1]) from YIQ to RGB color space
"""
def YIQTorgb(im):
    y = im[:, :, 0]
    i = im[:, :, 1]
    q = im[:, :, 2]
    r = y + 0.956*i + 0.619*q
    g = y -0.272*i -0.647*q
    b = y -1.106*i + 1.703*q
    return np.stack([r, g, b], axis = 2)

#------------------------------------------------------------------------------
# functions for creating steerable pyramid filters

'''
Computes polar coordinates r and theta as 2D arrays of shape (height, width)
the origin is at the center and assume the corresponding cartesian coordinates
span the range [-pi, pi]
'''
def polarFrequencyCoordinates(width, height, xmin = -1, xmax = 1, 
                             ymin = -1, ymax = 1):
    nx, ny = (width, height)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xv, yv = np.meshgrid(xs, ys)
    r = np.sqrt(xv**2 + yv**2)
    # eliminate zeros because we will compute log of r later
    if np.sum(r == 0) > 0:
        rows, cols = np.where(r == 0)
        r[r == 0] = r[rows[0], cols[0] - 1]
        
    theta = np.arctan2(yv, xv)
    return (r, theta)

"""
compute the angular part of Bk filters, G(theta)
please refer to the description in the report
thetas is the output theta of function "polarFrequencyCoordinates"
"""
def computeG(K, k, thetas):
    # note k is an integer in range [0, K-1]
    # K is number of bands
    denominator = np.sqrt(K * np.math.factorial(2*(K-1)))
    alpha = 2**(K-1)*np.math.factorial(K-1)/denominator 
    # counter-clockwise rotate by np.pi*k/K to form the sub-band
    new_thetas = thetas - np.pi*k/K # may out of [-pi, pi] range
    
    # wrapping around the theta angles using code from 
    # https://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap/15927914
    new_thetas = (new_thetas + np.pi)% (2 * np.pi) - np.pi # correct range
    results = alpha * np.cos(new_thetas)**(K-1)
    results[np.abs(new_thetas) >= np.pi/2] = 0
    return results

"""
compute the radial part of Bk filters, H(r)
please refer to the description in the report
rs is the output r of function "polarFrequencyCoordinates"
"""
def computeH(rs, thresh_low = np.pi/4, thresh_high = np.pi/2):
    results = np.cos(np.pi/2*np.log2(2/np.pi*rs))
    results[rs >= thresh_high] = 1
    results[rs <= thresh_low] = 0
    return results

"""
compute the L filter
please refer to the description in the report
rs is the output r of function "polarFrequencyCoordinates"
"""
def computeLOrigin(rs, thresh_low = np.pi/4, thresh_high = np.pi/2):
    results = 2 * np.cos(np.pi/2 * np.log2(4/np.pi*rs))
    results[rs >= thresh_high] = 0
    results[rs <= thresh_low] = 2
    return results

"""
compute the L filter, by following the requirement of |L|^2 + |H|^2 = 1
please refer to the description in the report
h is the output of the function "computeH"
"""
def computeLDirect(h):
    return np.sqrt(1 - h**2)

"""
compute the steerable pyramid filters in frequency domain
please refer to the description in the report
rs and thetas are the outputs of function "polarFrequencyCoordinates"
"""
def computeFilters(rs, thetas, K, num_levels, verbose = False):
    downsample_factor = 2
    height, width = rs.shape
    filters = []
    num_filters = num_levels*K+2
    filter_ranges = np.zeros((num_filters, 4), dtype = int)
    filter_idx = 0
    Gk = [computeG(K, x, thetas) for x in range(K)]
    for level in range(num_levels+1):
        # compute region of downsampled frequency spectrum
        if verbose: print(f"function computeFilters processing level {level}")
        if level == 0:
            rs0 = rs/downsample_factor
            H0 = computeH(rs0) # four corners
            filters.append(H0)
            filter_ranges[filter_idx, :] = np.array([0, height,
                                                   0, width])
            filter_idx += 1
            L = computeLDirect(H0) # circular region for next level

        else:
            num_rows = height//(2**(level-1))
            num_cols = width//(2**(level-1))
            row_start = height//2 - num_rows//2
            row_end = row_start + num_rows
            col_start = width//2 - num_cols//2
            col_end = col_start + num_cols
            filter_range = np.array([row_start, row_end, 
                           col_start, col_end])
                               
            rs_level = rs*downsample_factor**(level - 1)
            H_temp = computeH(rs_level) # all high region
            H = H_temp * L # band region radial part
            L = computeLDirect(H_temp) # update L for the next level
            # Bk only covers half of the spectrum because of complex space
            Bk = [np.multiply(H, Gk[x]) for x in range(K)]
            Bk_cut = [f[row_start:row_end, col_start:col_end] for f in Bk]
            filters.extend(Bk_cut)
            filter_ranges[filter_idx:filter_idx + K, :] = \
                    np.repeat(filter_range[np.newaxis, :], K, 
                              axis= 0)
            filter_idx += K

    # add residual lowpass filter
    num_rows = height//(2**(level))
    num_cols = width//(2**(level))
    row_start = height//2 - num_rows//2
    row_end = row_start + num_rows
    col_start = width//2 - num_cols//2
    col_end = col_start + num_cols
    filter_range = np.array([row_start, row_end, 
                   col_start, col_end])
    filters.append(L[row_start:row_end, col_start:col_end])
    filter_ranges[filter_idx, :] = filter_range

    return filters, filter_ranges

#------------------------------------------------------------------------------
# functions for create steerable pyramids of an image


"""
compute maximum levels of steerable pyramids that an image can have
last_filter_size defines the minimum size of last layer
"""
def maxLevel(height, width, downsample_factor = 2,
            last_filter_size = 4):
    min_size = np.min([height, width])
    return int(np.floor(np.log2(min_size) - 
                    np.log2(last_filter_size)))

"""
read a video file into array
max_num_frames: if not 0, can limit number of frames read
"""
def readVideoMatrix(video_filepath, max_num_frames = 0, 
                    num_channels = 3):
    video = cv2.VideoCapture(video_filepath)
    if not video.isOpened:
        print("Error opening video file")
        
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height =int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
    fps = video.get(cv2.CAP_PROP_FPS)
    if max_num_frames > 0 and max_num_frames < num_frames:
        num_frames = max_num_frames
    print(f"read {num_frames} frames into array")
    
    # load frames into array
    videoM = np.zeros((num_frames, height, width, num_channels), 
                      dtype = np.float64)
    for i in range(num_frames):
        ret, frame = video.read()
        if ret:
            videoM[i, :, :] = convertToFloat(frame)
        else:
            print(f"Error reading frame #{i}")
            break

    video.release()
    return videoM, fps, fourcc


"""
return parameters used to access collapsed pyramids
"""
def pyramidSize(filters):
    pyramids_sizes = []
    num_elements = 0
    for f in filters:
        f_size = np.size(f)
        pyramids_sizes.append(f_size)
        num_elements += f_size
    pyramids_startidx = np.cumsum([0] + pyramids_sizes)
    pyramids_sizes = np.array(pyramids_sizes)
    return pyramids_sizes, pyramids_startidx, num_elements

"""
create steerable pyramids of an image 
filters, filter_ranges are outputs of function "computeFilters"
im: must be 2d image
fft: if True, return pyramids in frequency domain
collapse: if True, return a 1D array of concatenated pyramids
          if False, return a list of 2D arrays of filters in decreasing frequency order
"""
def steerPyramid(im, filters, filter_ranges, fft = False,
                collapse = False):
    # im is assumed to be 2d
    im_fft = np.fft.fftshift(np.fft.fft2(im))
    num_filters = len(filters)
    if collapse:
        pyramids_sizes, pyramids_startidx, num_elements = pyramidSize(filters)
        pyramids = np.zeros((num_elements), dtype = 'complex128')
    else:
        pyramids = []
    
    for i in range(num_filters):
        [row_start, row_end, col_start, col_end] = filter_ranges[i]
        im_fft_cut = im_fft[row_start:row_end, col_start:col_end]
        temp_fft = filters[i]*im_fft_cut
        if fft:
            pyramid = temp_fft
        else:
            pyramid =  np.fft.ifft2(np.fft.ifftshift(temp_fft))
            
        if collapse:
            pyramid_flat = pyramid.flatten()
            pyramid_size = np.size(pyramid_flat)
            pyramids[pyramids_startidx[i]:
                     pyramids_startidx[i+1]] = pyramid_flat
        else:
            pyramids.append(pyramid)

    return pyramids

"""
reconstruct image from steerable pyramids
pyramids is the output of function "steerPyramids"
filters, filter_ranges are outputs of function "createFilters"
fft: if True, return image in the frequency domain
collapse: if True, the pyramids is a 1d array; else, it is a list of
            2D arrays
"""
def reconstruct(pyramids, filters, filter_ranges, fft = False,
               collapse = False):
    height, width = filters[0].shape
    num_filters = len(filters)
    if collapse:_, pyramids_startidx, _ = pyramidSize(filters)
    for i in range(num_filters):
        if collapse:
            pyramid = pyramids[pyramids_startidx[i]:pyramids_startidx[i+1]]
            pyramid = pyramid.reshape(filters[i].shape)
            temp_fft = np.fft.fftshift(np.fft.fft2(pyramid))
        else:
            temp_fft = np.fft.fftshift(np.fft.fft2(pyramids[i]))
        [row_start, row_end, col_start, col_end] = filter_ranges[i]
        if i == 0: # the first highpass filter (four corners)
            im_fft = temp_fft * filters[i]
        elif i == num_filters-1: # the last lowpass residual
            im_fft[row_start:row_end, col_start:col_end] += temp_fft * filters[i]
        else: # multiply by 2 because the sub-band covers half of the complex spectrum
            im_fft[row_start:row_end, col_start:col_end] += 2 * temp_fft * filters[i]
    if fft:
        return im_fft
    else:
        return np.fft.ifft2(np.fft.ifftshift(im_fft))

#------------------------------------------------------------------------------

if __name__ == '__main__':
    print("uncomment different sections to produce actual results")

    processVideoMedianPhase(
        Data_Folder + "moon.avi",
        Result_Folder + "test.avi",
             window_size = 11, 
             fastYIQ = True,
    verbose = False,
    max_num_frames = 25)


    # calls made to create the results
    # processVideoMedianPhase(
    #     Data_Folder + "moon.avi",
    #     Result_Folder + "moon_attenuate_YIQ.avi",
    #          window_size = 11, 
    #          fastYIQ = True,
    # verbose = False,
    # max_num_frames = 0)

    # processVideoMedianPhase(
    #     Data_Folder + "moon.avi",
    #     Result_Folder + "moon_attenuate_RGB.avi",
    #          window_size = 11, 
    #          fastYIQ = False,
    # verbose = False,
    # max_num_frames = 0)

    # processVideoMedianPhase(
    #     Data_Folder + "subway.mp4",
    #     Result_Folder + "subway_attenuated_YIQ.mp4",
    #          window_size = 11, 
    #          fastYIQ = True,
    # verbose = False,
    # max_num_frames = 0)

    # processVideoMedianPhase(
    #     Data_Folder + "fly03_short.avi",
    #     Result_Folder + "fly03_short_attenuated_YIQ.avi",
    #          window_size = 11, 
    #          fastYIQ = True,
    # verbose = False,
    # max_num_frames = 0)

    # processVideoMedianPhase(
    #     Data_Folder + "fly03_short.avi",
    #     Result_Folder + "fly03_short_attenuated_YIQ_win25.avi",
    #          window_size = 25, 
    #          fastYIQ = True,
    # verbose = False,
    # max_num_frames = 0)

    # processVideoMedianPhase(
    #     Data_Folder + "fly03_short.avi",
    #     Result_Folder + "fly03_short_attenuated_RGB.avi",
    #          window_size = 11, 
    #          fastYIQ = False,
    # verbose = False,
    # max_num_frames = 0)

    # processVideoMedianPhase(
    #     Data_Folder + "subway.mp4",
    #     Result_Folder + "subway_attenuated_RGB.mp4",
    #          window_size = 11, 
    #          fastYIQ = False,
    # verbose = False,
    # max_num_frames = 0)

    # processVideoMedianPhase(
    #     Data_Folder + "subway.mp4",
    #     Result_Folder + "subway_attenuated_RGB_win25.mp4",
    #          window_size = 25, 
    #          fastYIQ = False,
    # verbose = False,
    # max_num_frames = 0)

