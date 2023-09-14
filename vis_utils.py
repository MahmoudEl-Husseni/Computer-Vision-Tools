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
