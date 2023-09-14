# ===================================================================================================================== #
# -------------------------------------------------- Visualizations --------------------------------------------------- #
# ===================================================================================================================== #
# Plot images: 
def plot_images(images, titles=None, rows=-1, cols=-1):

  """
  Plots a given set of images.

  Args:
    images (list): A list of images.
    titles (list, optional): A list of titles for the images. If not provided,
        the titles will be the indices of the images.
    rows (int, optional): The number of rows in the plot. Defaults to -1,
        which means that the number of rows will be determined automatically.
    cols (int, optional): The number of columns in the plot. Defaults to -1,
        which means that the number of columns will be determined automatically.

  Returns:
    None.

  Raises:
    TypeError: If images is not a list of images.
    TypeError: If titles is not None and is not a list of strings.
    Exception: If there is an error plotting an image.
  """

  # Check the input arguments.
  if not isinstance(images, list):
    raise TypeError('images must be a list of images')
  if titles is not None and not isinstance(titles, list):
    raise TypeError('titles must be a list of strings')
  if titles is None: 
    titles = np.arange(len(images))
  # Set the default values for rows and cols if not specified.
  if rows == -1 and cols == -1:
    rows = np.floor(np.sqrt(len(images))).astype(int)
    cols = np.ceil((len(images) / rows)).astype(int)

  # Create the figure and subplots.
  fig, axs = plt.subplots(rows, cols, figsize=(20, 12))
  if axs.ndim==1: axs = np.expand_dims(axs, 0)

  # Plot the images.
  for row in range(rows):
    for col in range(cols):
        i = row*cols + col
        if(i>=len(images)): break
        img = images[i]
        ax = axs[row][col]
        try:
            ax.imshow(img)
            ax.set_title(f'{titles[i]}')
            ax.set_xticks([])
            ax.set_yticks([])
        except:
            raise Exception('Error plotting image')

  # Show the plot.
  plt.show()
# ============================================================  
# Plots a random selection of images from the given directory.  
def plot_random_images_from_dir(path, n_images=64):

    """
    Plots a random selection of `n_images` images from the directory `path`.

    Args:
        path (str): The path to the directory containing the images.
        n_images (int, optional): The number of images to plot. Defaults to 64.

    Returns:
        None.
    """

    # Get the list of all images in the directory.
    IMAGES = os.listdir(path)

    # Generate a random sample of `n_images` indices.
    indices = np.random.randint(0, len(IMAGES), n_images)

    # Load the images at the specified indices.
    def load(im_path):
        """Loads an image from the given path."""
        img = cv.imread(os.path.join(path, im_path))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return img

    images = [load(IMAGES[i]) for i in indices]

    # Plot the images.
    plot_images(images)
# ============================================================  
# PLot Histogram of colors
def plot_chist(im):
  if(im.ndim==2): 
    im = np.expand_dims(im, 0)
    im = np.vstack([im, im, im])
  plt.figure(figsize=(20, 12))
  im = np.float32(im)
  reds = im[:, :, 0].flatten()
  greens = im[:, :, 1].flatten()
  blues = im[:, :, 2].flatten()

  # Separate Histograms for each color
  plt.subplot(3, 1, 1)
  plt.title("histogram of Red")
  sns.distplot(reds, kde=True, hist=False, color='red')

  plt.subplot(3, 1, 2)
  plt.title("histogram of Green")
  sns.distplot(greens, kde=True, hist=False, color='green')

  plt.subplot(3, 1, 3)
  plt.title("histogram of Blue")
  sns.distplot(blues, kde=True, hist=False, color='blue')
  plt.show();
# ============================================================
 # To compare two images
 def compare(images, images_):
  """
  Compares two sets of images.

  Args:
    images: The first set of images.
    images_: The second set of images.

  Returns:
    A list of images showing the comparison of each pair of images.

  Raises:
    ValueError: If the two sets of images do not have the same length.

  """
  assert len(images)==len(images_), "Images and Images_ should have the same length"
  results = []
  for i in range(len(images)):
    im_ = images_[i].copy()
    im = images[i].copy()
    im_ = cv.resize(im_, im.shape[:-1][::-1])
    res = np.hstack([im, im_])
    results.append(res)
  return results
# ============================================================
# Plot line using plotly
def plot_line(y_data=None, x_data=None, name=""):
  """
  Plots a line graph.

  Args:
    y_data: The y-axis data.
    x_data: The x-axis data. If None, it will be automatically generated.
    name: The name of the line.

  Returns:
    A plotly figure object.
  """

  if x_data is None:
    x_data = list(range(len(y_data)))

  assert y_data is not None, f"y_data should have iterative data structure: {str(type(y_data))} is not allowed"

  fig = go.Figure(
      layout=go.Layout(
          plot_bgcolor="rgba(0, 0, 0, 1)", paper_bgcolor="rgba(70, 10, 10, 1)",
      )
  )

  fig.add_trace(go.Scatter(x=x_data, y=y_data, name=name, marker=dict(color="white")))

  for i in range(len(x_data)):
    fig.add_annotation(
        x=x_data[i], y=y_data[i], text="o", showarrow=False, font=dict(color="white", size=20)
    )

  fig.update_layout(
      xaxis=dict(color="white"),
      yaxis=dict(color="white"),
  )

  return fig
# ============================================================
# To Tune Hyperparameters
def plot_params(im, transform, params):

  """
  Plots the results of applying a transform to an image for different parameters.

  Args:
    im: The image to be adjusted.
    transform: A function that takes an image and a dictionary of parameters as input and returns the transformed image.
    params: A list of dictionaries, each of which specifies the parameters for the transform.

  Returns:
      None.
  """

  im_ = im.copy()
  images_ = []
  titles = []
  for v in params:
    images_.append(transform(im, debug=True, **v))
    titles.append(str(v))
  plot_images(images_, titles)
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
def adjust_contrast(im, clipLimit=2.0, tileGridSize=(8, 8), debug=False):
  """
  Adjusts the contrast of an image using CLAHE.

  Args:
    im: The image to be adjusted.
    clipLimit: The clip limit for CLAHE. This is a value that controls how much the contrast is enhanced.
    tileGridSize: The tile grid size for CLAHE. This is the size of the local regions that are used to equalize the histogram.
    debug: Whether to return a debug image showing the original and adjusted images.

  Returns:
    The adjusted image.
  """
  im_ = im.copy()
  lab= cv.cvtColor(im_, cv.COLOR_BGR2LAB)
  l_channel, a, b = cv.split(lab)

  # Applying CLAHE to L-channel
  clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
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
  im_ = np.float32(im_)
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
# ===================================================================================================================== #
# --------------------------------------------------- Various Utils --------------------------------------------------- #
# ===================================================================================================================== #
def create_video(src_dir, out_path):
  """
  Creates a video from a directory of images.

  Args:
    src_dir: The directory containing the images.
    out_path: The path to the output video file.

  Returns:
    None.
  """

  img_array = []
  file_names = sorted(glob.glob(src_dir + '/*.jpg'))
  for filename in file_names:
    img = cv.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

  out = cv.VideoWriter(out_path, cv.VideoWriter_fourcc(*'DIVX'), 15, size)

  for i in range(len(img_array)):
    out.write(img_array[i])
  out.release()
# ============================================================
def display_video(vid_path):
  """
  Displays a video in an Ipynb file.

  Args:
    vid_path: The path to the video file.

  Returns:
    None.

  **Optimizations:**
    - Use `display.HTML` to display the video in Ipynb files.
  """

  out_path = vid_path.split('.')[0] + '.mp4'
  # convert video extenstion from .avi to .mp4
  os.system(f"ffmpeg -i {vid_path} -y {out_path}")

  mp4 = open(out_path, 'rb').read()
  data_url = "data:video/mp4;base64," + base64.b64encode(mp4).decode()

  display(HTML("""
  <video controls width="1080" height="576">
        <source src="%s" type="video/mp4">
  </video>
  """ % data_url))
# ============================================================
