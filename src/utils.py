import sys
import pathlib

import cv2
import imutils
from numpy import ndarray
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def resource_path(relative_path: str):
  """Construct the resource patch for a resource.

  Args:
      relative_path (str): The path relative to the resource path

  Returns:
      pathlib.Path: The path to the given resource
  """
  # Get absolute path to resource, works for dev and for PyInstaller
  if getattr(sys, 'frozen', False):
    # PyInstaller creates a temp folder and stores path in _MEIPASS
    base_path = pathlib.Path(pathlib.sys._MEIPASS)
  else:
    base_path = pathlib.Path()

  return base_path / "res" / relative_path

def show_images(images: list, scale: float = None):
  for i, image in enumerate(images):
    if scale is None:
      cv2.imshow(f"{i}", image)
    else:
      cv2.imshow(f"{i}", imutils.resize(image, (int) (image.shape[1] * scale), (int) (image.shape[0] * scale)))

  cv2.waitKey()

def scale_image(img: ndarray, scale: float):
  """Scales an image by a given scale value equally in vertical and horizontal direction.

  Args:
      img ([type]): The source image
      scale (float): The scale value

  Returns:
      numpy.ndarray: The resized image
  """
  return cv2.resize(img, ((int) (img.shape[1] * scale), (int) (img.shape[0] * scale)))

def crop_image(img: ndarray, position: tuple, width: int, height: int):
  return img[position[1]:position[1] + height, position[0]:position[0] + width]

def crop_image_centered(img: ndarray, width: int, height: int):
  position = ((img.shape[1] // 2) - (width // 2), (img.shape[0] // 2) - (height // 2))
  
  return crop_image(img, position, width, height)

def plot_image_realtime(img):
  assert len(img.shape) == 2, "Image has to have 2 channels"

  # https://plotly.com/python/3d-surface-plots/
  fig = go.Figure(data=[go.Surface(z=img)])
  fig.show()

def plot_image(img):
  assert len(img.shape) == 2, "Image has to have 2 channels"

  # https://stackoverflow.com/questions/31805560/how-to-create-surface-plot-from-greyscale-image-with-matplotlib
  # This approach is very slow, need to look for a more realtime oriented way of doing things

  # Create x and y coordinate arrays
  xx, yy = np.mgrid[0: img.shape[0], 0:img.shape[1]]

  fig = plt.figure()
  ax = plt.axes(projection="3d")
  ax.plot_surface(xx, yy, img, rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)
  ax.set_title("image plot")

  plt.show()

def convert_image_rgb2gray(img):
  assert len(img.shape) == 3, "Image has to have 3 channels"

  result = np.zeros(img.shape[:2], np.uint8)

  def clipped_value(values: np.ndarray):
    return np.max(np.minimum(np.maximum(values, 0), 255), axis=2)

  return clipped_value(img)


def normalize_image(img: np.ndarray):
  minValue = img.min()
  maxValue = img.max()
  result = np.float32((img - minValue) * (255 / (maxValue - minValue)))

  return result, minValue, maxValue 