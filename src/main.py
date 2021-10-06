import cv2
import imutils
import numpy as np

from utils import crop_image_centered, plot_image_realtime, resource_path, show_images, plot_image, convert_image_rgb2gray

"""Frankot and Chellappa algorithm. Manually converted from matlab source at https://peterkovesi.com/matlabfns/
"""
def frankotchellappa(dzdx: np.ndarray, dzdy: np.ndarray):
  assert dzdx.shape == dzdy.shape, "Gradient images have to be the same size"
  assert len(dzdx.shape) == 2 and len(dzdx.shape) == 2, "Gradients have to be grayscale images"

  # Store rows and cols since we will need it later
  rows, cols = dzdx.shape

  # The following sets up matrices specifying frequencies in the x and y
  # directions corresponding to the Fourier transforms of the gradient
  # data.  They range from -0.5 cycles/pixel to + 0.5 cycles/pixel. The
  # fiddly bits in the line below give the appropriate result depending on
  # whether there are an even or odd number of rows and columns
  wx, wy = np.meshgrid( (np.arange(1, cols + 1) - (np.fix(cols / 2) + 1)) / (cols - (cols % 2)),
                        (np.arange(1, rows + 1) - (np.fix(rows / 2) + 1)) / (rows - (rows % 2)))

  # Quadrant shift to put zero frequency at the appropriate edge
  wx = np.fft.ifftshift(wx)
  wy = np.fft.ifftshift(wy)

  # Calculate fast furier transformation
  dzdxFFT = np.fft.fft2(dzdx)
  dzdyFFT = np.fft.fft2(dzdy)

  # Debug info, display images
  #show_images([np.real(dzdxFourier), np.real(dzdyFourier)])

  # Integrate in the frequency domain by phase shifting by pi/2 and
  # weighting the Fourier coefficients by their frequencies in x and y and
  # then dividing by the squared frequency.  eps is added to the
  # denominator to avoid division by 0.
  # Equation 21 from paper
  Z = (-np.lib.scimath.sqrt(-1) * wx * dzdxFFT - np.lib.scimath.sqrt(-1) * wy * dzdyFFT) / \
      ((wx ** 2) + (wy ** 2) + np.finfo(float).eps)

  return np.real(np.fft.ifft2(Z))

if __name__ == "__main__":
  # Load braille images
  # @TODO loading the images as grayscale right away does the same as (load image color, convert to grayscale, normalize) but
  #       does skip the step of manually filtering the grayscale values
  topLeftD = cv2.imread(str(resource_path("topLeftD.bmp")), cv2.IMREAD_COLOR)
  topRightD = cv2.imread(str(resource_path("topRightD.bmp")), cv2.IMREAD_COLOR)
  bottomRightD = cv2.imread(str(resource_path("bottomRightD.bmp")), cv2.IMREAD_COLOR)
  bottomLeftD = cv2.imread(str(resource_path("bottomLeftD.bmp")), cv2.IMREAD_COLOR)

  # Load white images (Next to braille dot with no elevation)
  topLeftW = cv2.imread(str(resource_path("topLeftW.bmp")), cv2.IMREAD_COLOR)
  topRightW = cv2.imread(str(resource_path("topRightW.bmp")), cv2.IMREAD_COLOR)
  bottomRightW = cv2.imread(str(resource_path("bottomRightW.bmp")), cv2.IMREAD_COLOR)
  bottomLeftW = cv2.imread(str(resource_path("bottomLeftW.bmp")), cv2.IMREAD_COLOR)

  # Slightly blur white images for less irritation by texture or background noise
  topLeftW = cv2.medianBlur(topLeftW, 5)
  topRightW = cv2.medianBlur(topRightW, 5)
  bottomRightW = cv2.medianBlur(bottomRightW, 5)
  bottomLeftW = cv2.medianBlur(bottomLeftW, 5)

  # Store width and height since we will need it a few times
  height, width = topLeftD.shape[:2]

  # Rotate images since lights are located at top left, top right,... instead of top, right, ...
  # @TODO instead of rotating, work with diagonal images directly ( (x +- y) / sqrt(2))
  topD = imutils.rotate(topLeftD, 45)
  rightD = imutils.rotate(topRightD, 45)
  bottomD = imutils.rotate(bottomRightD, 45)
  leftD = imutils.rotate(bottomLeftD, 45)

  topW = imutils.rotate(topLeftW, 45)
  rightW = imutils.rotate(topRightW, 45)
  bottomW = imutils.rotate(bottomRightW, 45)
  leftW = imutils.rotate(bottomLeftW, 45)

  # Debug info, display images
  #show_images([topD, rightD, bottomD, leftD], 0.5)  

  # Crop rotated images to the part that interests us (the braille dot)
  topCropD = crop_image_centered(topD, height // 2, height // 2)
  rightCropD = crop_image_centered(rightD, height // 2, height // 2)
  bottomCropD = crop_image_centered(bottomD, height // 2, height // 2)
  leftCropD = crop_image_centered(leftD, height // 2, height // 2)

  topCropW = crop_image_centered(topW, height // 2, height // 2)
  rightCropW = crop_image_centered(rightW, height // 2, height // 2)
  bottomCropW = crop_image_centered(bottomW, height // 2, height // 2)
  leftCropW = crop_image_centered(leftW, height // 2, height // 2)

  # Debug info, display images
  #show_images([topCropD, rightCropD, bottomCropD, leftCropD])

  # Normalize images for brightness gradient
  topCropN = np.float32(topCropD / topCropW)
  rightCropN = np.float32(rightCropD / rightCropW)
  bottomCropN = np.float32(bottomCropD / bottomCropW)
  leftCropN = np.float32(leftCropD / leftCropW)

  # Debug info, display images
  #show_images([topCropN, rightCropN, bottomCropN, leftCropN])

  # Convert images to grayscale f32 because we need 1 intensity channel for dft
  # @TODO maybe manually do this step to achieve more consistent results (discard values > 255, take max value of remaining channels)
  topCropNG = np.float32(cv2.cvtColor(topCropN, cv2.COLOR_RGB2GRAY))
  rightCropNG = np.float32(cv2.cvtColor(rightCropN, cv2.COLOR_RGB2GRAY))
  bottomCropNG = np.float32(cv2.cvtColor(bottomCropN, cv2.COLOR_RGB2GRAY))
  leftCropNG = np.float32(cv2.cvtColor(leftCropN, cv2.COLOR_RGB2GRAY))

  # Debug info, display images
  show_images([topCropNG, rightCropNG, bottomCropNG, leftCropNG])

  # Calculate gradient
  a = 15 # Distance from lense to object in mm
  b = 21 # Distance from LED to lense (radius)
  ratio = 4 / 3 # Ratio of the source image
  cropHeight, cropWidth = topCropNG.shape # Width and height of the cropped image
  magnification = 70.4 # Maginification of the microscope
  factor = 390.72 / magnification / width # Size of a pixel in mm (Full image is 390.72mm wide on magnification 1)
  #focalLength = 40 # Focal length of the microscope
  
  # Meshgrid containing the distance of each pixel to the center of the cropped image
  wx, wy = np.meshgrid( np.arange(1, cropWidth + 1) - (np.fix(cropWidth / 2) + 1),
                        np.arange(1, cropHeight + 1) - (np.fix(cropHeight / 2) + 1))
  # Calculate distance of each pixel to the lense
  wx = np.sqrt(a ** 2 + (np.absolute(wx) * factor) ** 2)
  wy = np.sqrt(a ** 2 + (np.absolute(wy) * factor) ** 2)
  # Calculate tan(theta) for each pixel
  # @TODO We are assuming we are working with a right triangle which is not true but should be close enough for testing.
  #       Need to work out how to get an accurate tan(theta)
  wx = b / wx
  wy = b / wy
  
  # From Peer (?)
  #wx, wy = np.meshgrid(np.arange(1, cropWidth + 1) - cropWidth / 2, np.arange(1, cropHeight + 1) - cropHeight / 2)
  #wx = (wx ** 2 - b ** 2 + a ** 2) / (2 * a * b)
  #wx = np.sqrt(wx + 1) + wx
  #wy = (wy ** 2 - b ** 2 + a ** 2) / (2 * a * b)
  #wy = np.sqrt(wy + 1) + wy
  
  dzdx = (leftCropNG - rightCropNG) / (leftCropNG + rightCropNG) / wx
  dzdy = (topCropNG - bottomCropNG) / (topCropNG + bottomCropNG) / wy

  # Debug info, display images
  #show_images([dzdx, dzdy])
  
  result = frankotchellappa(dzdx, dzdy)

  # Debug info, display images
  #show_images([result])
  #plot_image(result)
  plot_image_realtime(result)
