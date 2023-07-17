# ===================================================================================================================== #
# --------------------------------------- Image Processing and Transformations ---------------------------------------- #
# ===================================================================================================================== #
# Apply Transform: 
def apply_transform(images, transform):
  '''
  :images: numpy array.
  :transform: callable.
  '''
  images_ = images.copy()
  for i in range(len(images)):
    images_[i] = transform(images[i]) 
  return images_
# ========================================================
def adjust_contrast(im, debug=False):
  """
  Adjusts the contrast of an image using CLAHE.

  Args:
    im: The image to be adjusted.
    debug: Whether to return a debug image showing the original and adjusted images.

  Returns:
    The adjusted image.
  """
  im_ = im.copy()
  lab= cv.cvtColor(im_, cv.COLOR_BGR2LAB)
  l_channel, a, b = cv.split(lab)

  # Applying CLAHE to L-channel
  # feel free to try different values for the limit and grid size:
  clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  cl = clahe.apply(l_channel)

  # merge the CLAHE enhanced L-channel with the a and b channel
  limg = cv.merge((cl,a,b))

  # Converting image from LAB Color model to BGR color spcae
  im_ = cv.cvtColor(limg, cv.COLOR_LAB2BGR)

  if debug:
    # Stacking the original image with the enhanced image
    im_ = np.hstack((im, im_))
  return im_
# ============================================================
def hist_equalize(im, debug=False):
  """
  Performs histogram equalization on an image.

  Args:
    im: The image to be histogram equalized.
    debug: Whether to return a debug image showing the original and histogram equalized images.

  Returns:
    The histogram equalized image.
  """
  im_ = im.copy()

  # Equalize each channel
  for i in range(im.shape[2]):
    im_[:, :, i] = cv.equalizeHist(im[:, :, i])
  
  if debug:
    im_ = np.hstack((im, im_))

  return im_
# ============================================================
def white_balance(im, debug=False):
  """
  Performs white balancing on an image.

  Args:
    im: The image to be white balanced.
    debug: Whether to return a debug image showing the original and white balanced images.

  Returns:
    The white balanced image.
  """
  im_ = im.copy()
  
  im_ = cv.cvtColor(im_, cv.COLOR_BGR2LAB)
  avg_a = np.average(im_[:, :, 1])
  avg_b = np.average(im_[:, :, 2])
  im_[:, :, 1] = im_[:, :, 1] - ((avg_a - 128) * (im_[:, :, 0] / 255.0) * 1.2)
  im_[:, :, 2] = im_[:, :, 2] - ((avg_b - 128) * (im_[:, :, 0] / 255.0) * 1.2)
  im_ = cv.cvtColor(im_, cv.COLOR_LAB2BGR)

  if debug:
    im_ = np.hstack([im_, im])
  return im_
# ============================================================
def median_filter(im, kernel_size):
  """
  Applies a median filter to an image.

  Args:
    image: The image to be filtered.
    kernel_size: The size of the kernel to be used.

  Returns:
    The filtered image.
  """
  im_ = im.copy()
  filtered_image = np.zeros_like(im_)
  for i in range(im_.shape[0]):
    for j in range(im_.shape[1]):
      for k in range(im_.shape[2]):
        window = im_[i:i + kernel_size, j:j + kernel_size, k]
        filtered_image[i, j, k] = np.median(window)

  return filtered_image
# ============================================================
def gaussian_blur(im, kernel_size=5, debug=False):
  """
  Applies a Gaussian blur to an image.

  Args:
    image: The image to be blurred.
    kernel_size: The size of the kernel to be used.
    debug: Whether to return a debug image showing the original and white balanced images.

  Returns:
    The blurred image.
  """
  im_ = im.copy()
  im_ = cv.GaussianBlur(im_, (kernel_size, kernel_size), 0)
  if debug:
    im_ = np.hstack([im, im_])

  return im_
# ============================================================
def sharpen_image(im, center_weight=5, edges_weight=-1, debug=False):
  """
  Sharpens an image using OpenCV.

  Args:
    image: The image to be sharpened.
    kernel_size: The size of the kernel to be used.

  Returns:
    The sharpened image.
  """
  im_ = im.copy()
  # Create a sharpening kernel
  kernel = np.array([[0, edges_weight, 0], [edges_weight, center_weight, edges_weight], [0, edges_weight, 0]])

  # Apply the sharpening kernel to the image
  im_ = cv.filter2D(im_, -1, kernel)
  if debug:
    im_ = np.hstack([im, im_])

  return im_
# ============================================================
