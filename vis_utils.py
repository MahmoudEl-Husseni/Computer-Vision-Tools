# ===================================================================================================================== #
# -------------------------------------------------- Visualizations --------------------------------------------------- #
# ===================================================================================================================== #
# Plot images: 
def plot_images(images, titles=None, rows=-1, cols=-1):
  if titles is None:
    titles = np.arange(len(images))

  n_images = len(images) 
  if rows==-1 and cols==-1:
    rows = np.sqrt(n_images).__floor__()
    cols = (n_images / rows).__ceil__()
  fig, axs = plt.subplots(rows, cols, figsize=(20, 12))
  if axs.ndim==1: axs = np.expand_dims(axs, 0)
  for row in range(rows):
    for col in range(cols):
      i = row*cols + col
      if(i>=len(images)): break
      img = images[i]
      ax = axs[row][col]
      ax.imshow(img)
      ax.set_title(f'{titles[i]}')
      ax.set_xticks([])
      ax.set_yticks([])
  plt.show()
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

