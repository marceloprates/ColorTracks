import os
import re
import cv2
import numpy as np
import gpxpy

# import osmnx as ox
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFilter
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.animation import PillowWriter
from scipy.integrate import solve_ivp
from scipy.interpolate import (
    splev,
    splprep,
    interp1d,
    make_interp_spline,
    CloughTocher2DInterpolator,
)
from shapely.geometry import LineString
from celluloid import Camera

# from vsketch import Vsketch
from PIL import Image
import hashlib
import pickle
import os
from PIL import Image, ImageChops, ImageDraw, ImageFont


np.random.seed(3)


def get_transparent_bbox(image):

    # Get the alpha channel (transparency mask)
    alpha = image.split()[-1]

    # Invert the alpha channel
    inverted_alpha = ImageChops.invert(alpha)

    # Get the bounding box of the non-zero regions in the inverted alpha channel
    bbox = inverted_alpha.getbbox()

    return bbox


def path2curve(graph, path):
    """
    Convert a path in the graph to a curve by extracting the x and y coordinates of the edges.

    Parameters:
    graph (networkx.Graph): The graph containing the nodes and edges.
    path (list): A list of nodes representing the path.

    Returns:
    tuple: Three lists containing the x coordinates, y coordinates, and cumulative distances along the path.
    """
    x = []
    y = []
    for u, v in zip(path[:-1], path[1:]):
        # if there are parallel edges, select the shortest in length
        data = min(graph.get_edge_data(u, v).values(), key=lambda d: d["length"])
        if "geometry" in data:
            # if geometry attribute exists, add all its coords to list
            xs, ys = data["geometry"].xy
            x.extend(xs)
            y.extend(ys)
        else:
            # otherwise, the edge is a straight line from node to node
            x.extend((graph.nodes[u]["x"], graph.nodes[v]["x"]))
            y.extend((graph.nodes[u]["y"], graph.nodes[v]["y"]))

    # Calculate cumulative distances along the path
    x, y = map(np.array, [x, y])
    t = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))

    return x, y, t


def process_and_smooth_path(
    path,
    show=False,
    noise_level=1e-6,
    spline_smoothing=1e-6,
    spline_degree=3,
    num_points=10**4,
    sinusoidal_amplitude=1e-4,
    sinusoidal_frequency=8,
    seed=0,
):
    """
    Process and smooth a given path by adding noise, fitting a B-spline, and adding sinusoidal noise.

    Parameters:
    path (numpy.ndarray): The input path as an array of coordinates.
    show (bool): Flag to indicate whether to show plots of the original and smoothed curves. Default is False.
    noise_level (float): The standard deviation of the Gaussian noise added to the path. Default is 1e-9.
    spline_smoothing (float): The smoothing factor for the B-spline. Default is 1e-7.
    spline_degree (int): The degree of the B-spline. Default is 3.
    num_points (int): The number of points to evaluate the B-spline. Default is 10**4.
    sinusoidal_amplitude (float): The amplitude of the sinusoidal noise added to the smoothed path. Default is 1e-4.
    sinusoidal_frequency (float): The frequency of the sinusoidal noise added to the smoothed path. Default is 8.

    Returns:
    tuple: The smoothed x, y coordinates and the cumulative distances along the smoothed path.
    """

    np.random.seed(seed)

    x, y = path[:, 1], path[:, 0]
    t = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))

    # Add a little noise to remove duplicates
    x += np.random.normal(0, noise_level, x.shape)
    y += np.random.normal(0, noise_level, y.shape)

    # Fit a B-spline representation to the path
    tck, u = splprep([x, y], s=spline_smoothing, k=spline_degree)

    # Evaluate the B-spline at a set of points
    u_fine = np.linspace(0, 1, num_points)
    x, y = splev(u_fine, tck)
    t = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
    x, y = x[1:], y[1:]
    k = len(x) // 4
    x, y, t = x[:-k], y[:-k], t[:-k]

    # Add sinusoidal noise to the smoothed curve
    x += sinusoidal_amplitude * np.sin(
        np.linspace(0, sinusoidal_frequency, len(x)) * 2 * np.pi
    )
    y += sinusoidal_amplitude * np.cos(
        np.linspace(0, sinusoidal_frequency, len(x)) * 2 * np.pi
    )

    if show:
        # Plot the original and smoothed curves in separate subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.patch.set_facecolor("#000")

        # Plot the original curve
        ax1.plot(path[:, 1], path[:, 0], c="w", linewidth=2, alpha=0.7)
        ax1.set_title("Original Curve", color="w")
        ax1.axis("off")
        ax1.axis("equal")

        # Plot the smoothed curve
        ax2.plot(x, y, c="r", linewidth=2, alpha=0.7)
        ax2.set_title("Smoothed Curve", color="w")
        ax2.axis("off")
        ax2.axis("equal")

        # Explanation
        fig.suptitle(
            "Comparison of Original and Smoothed Curves", color="w", fontsize=16
        )
        plt.show()

    return x, y, t


def compute_vector_field(x, y, t, smooth_factor=1e-5, show=True, seed=0):
    """
    Compute the vector field for a given path.

    Parameters:
    x (numpy.ndarray): The x-coordinates of the path.
    y (numpy.ndarray): The y-coordinates of the path.
    t (numpy.ndarray): The time values of the path.
    smooth_factor (float): The smoothing factor for the interpolator. Default is 1e-1.

    Returns:
    tuple: The interpolated time function, vector field function for plotting, and differential equation function for integration.
    """

    # Compute the derivatives of x and y with respect to time
    dx = np.gradient(x)
    dy = np.gradient(y)

    # Express t as a smooth function over x, y
    # Add some noise to avoid exact duplicates in the points
    xy_points = np.stack(
        [
            x,
            y,
        ],
        1,
    )

    t_func = CloughTocher2DInterpolator(
        xy_points, t, fill_value=0, tol=1e-6, maxiter=10**1
    )

    # Express dx as a smooth function over t
    dx_func = make_interp_spline(t, dx, k=3)
    # Express dy as a smooth function over t
    dy_func = make_interp_spline(t, dy, k=3)

    def vector_field_for_plotting(x_, y_):
        """
        Compute the vector field at given coordinates for plotting.

        Parameters:
        x_ (float): The x-coordinate.
        y_ (float): The y-coordinate.

        Returns:
        numpy.ndarray: The vector field components at the given coordinates.
        """
        t = t_func(x_, y_)
        dx = -4 * dx_func(t)
        dy = -4 * dy_func(t)

        mag = np.sqrt(dx**2 + dy**2)
        return np.where(mag > 1e-4, 0, 1) * np.array([dx * t, dy * t])

    def differential_equation_for_integration(t, y):
        """
        Compute the differential equation for the vector field for numerical integration.

        Parameters:
        t (float): The time value.
        y (list): The coordinates [x, y].

        Returns:
        list: The derivatives [dx, dy].
        """
        x, y = y
        dx, dy = vector_field_for_plotting(x, y)
        # Replace NaNs with 0
        dx = 0 if np.isnan(dx) else dx
        dy = 0 if np.isnan(dy) else dy
        return [dx, dy]

    if show:
        plot_vector_field(x, y, vector_field_for_plotting)

    return t_func, vector_field_for_plotting, differential_equation_for_integration


def plot_vector_field(
    x,
    y,
    vector_field_func,
    grid_size=40,
    plot_path=True,
    path_color="r",
    vector_color="w",
    alpha=0.7,
    title="Vector Field and Path",
    title_color="w",
    title_fontsize=16,
    text_color="w",
    text_fontsize=12,
    show_plot=True,
):
    """
    Plot the vector field as a quiver map along with the path.

    Parameters:
    x (numpy.ndarray): The x-coordinates of the original path.
    y (numpy.ndarray): The y-coordinates of the original path.
    x_ (numpy.ndarray): The x-coordinates of the smoothed path.
    y_ (numpy.ndarray): The y-coordinates of the smoothed path.
    vector_field_func (function): The function to compute the vector field.
    grid_size (int): The number of points in the grid for the quiver plot. Default is 40.
    plot_path (bool): Flag to indicate whether to plot the path. Default is True.
    path_color (str): The color of the path. Default is "r".
    vector_color (str): The color of the vectors in the quiver plot. Default is "w".
    alpha (float): The alpha transparency for the plot elements. Default is 0.7.
    title (str): The title of the plot. Default is "Vector Field and Path".
    title_color (str): The color of the title text. Default is "w".
    title_fontsize (int): The font size of the title text. Default is 16.
    text_color (str): The color of the explanatory text. Default is "w".
    text_fontsize (int): The font size of the explanatory text. Default is 12.
    show_plot (bool): Flag to indicate whether to show the plot. Default is True.

    Returns:
    matplotlib.figure.Figure: The figure object containing the plot.
    """
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_range = np.linspace(x_min, x_max, grid_size)
    y_range = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_range, y_range)
    U, V = vector_field_func(X, Y)

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("#000")

    # Plot the vector field
    quiver = ax.quiver(X, Y, U, V, color=vector_color, alpha=alpha)

    if plot_path:
        ax.plot(x, y, c=path_color, linewidth=2, alpha=alpha)

    # Add title and explanation
    ax.set_title(title, color=title_color, fontsize=title_fontsize)
    ax.text(
        0.5,
        -0.1,
        "The quiver plot shows the vector field along the path, with the path highlighted in red.",
        color=text_color,
        fontsize=text_fontsize,
        ha="center",
        transform=ax.transAxes,
    )

    ax.axis("off")

    if show_plot:
        plt.show()

    return fig


def standarize_path(x, y, interval=1e-6):
    """
    Standardize the path by resampling the x and y coordinates at uniform distance intervals.

    Parameters:
    x (numpy.ndarray): The x-coordinates of the path.
    y (numpy.ndarray): The y-coordinates of the path.
    interval (float): The uniform distance interval for resampling. Default is 1e-6.

    Returns:
    tuple: The resampled x and y coordinates, and the cumulative distances along the path.
    """
    # Calculate cumulative distances along the path
    t = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
    t = np.insert(t, 0, 0)  # Add the starting point

    # Create interpolating functions for x and y coordinates
    x_func = interp1d(t, x, kind="linear")
    y_func = interp1d(t, y, kind="linear")

    # Define the new uniform distance intervals
    t_uniform = np.arange(0, t[-1], interval)

    # Interpolate x and y coordinates at the new uniform intervals
    x_uniform = x_func(t_uniform)
    y_uniform = y_func(t_uniform)

    return x_uniform, y_uniform


def compute_streamlines(
    x,
    y,
    t,
    differential_equation_for_integration,
    frames=400,
    num_deviations=20,
    deviation_scale=0.0004,
    tmax=4 * 10**4,
    t_eval_points=500,
    method="RK23",
    show=True,
    memoize=False,
):
    """
    Compute streamlines for a given path and differential equation.

    Parameters:
    x (numpy.ndarray): The x-coordinates of the path.
    y (numpy.ndarray): The y-coordinates of the path.
    t (numpy.ndarray): The time values of the path.
    differential_equation_for_integration (function): The differential equation function for the vector field.
    frames (int): The number of frames for which to compute streamlines. Default is 100.
    num_deviations (int): The number of deviations to sample for each frame. Default is 10.
    deviation_scale (float): The scale of the random deviations. Default is 0.0001.
    tmax (float): The maximum time value for the integration. Default is 10**4.
    t_eval_points (int): The number of points at which to evaluate the solution. Default is 5000.
    method (str): The integration method to use. Default is "RK23".
    memoize (bool): Flag to indicate whether to memoize the results. Default is False.

    Returns:
    list: A list of streamlines for each frame.
    """
    ## Create a hash of the inputs
    # input_hash = hashlib.md5(
    #    pickle.dumps(
    #        (
    #            x,
    #            y,
    #            t,
    #            frames,
    #            num_deviations,
    #            deviation_scale,
    #            tmax,
    #            t_eval_points,
    #            method,
    #        )
    #    )
    # ).hexdigest()
    # cache_file = f"streamlines_{input_hash}.pkl"

    # if memoize and os.path.exists(cache_file):
    #    print(f"Loading streamlines from cache: {cache_file}")
    #    with open(cache_file, "rb") as f:
    #        streamlines = pickle.load(f)

    streamlines = []
    for t_ in np.linspace(t.min(), t.max(), frames)[::1]:
        mask = t <= t_
        initial_point = np.array([x[mask][-1], y[mask][-1]])
        dx, dy = np.random.randn(2, num_deviations) * deviation_scale
        for i in range(num_deviations):
            initial_point = np.array([x[mask][-1] + dx[i], y[mask][-1] + dy[i]])
            x0, y0 = initial_point

            sol = solve_ivp(
                differential_equation_for_integration,
                [0, tmax],
                initial_point,
                t_eval=np.linspace(0, tmax, t_eval_points),
                method=method,
                vectorized=True,
            )

            streamline = np.stack(standarize_path(*sol.y), axis=1)
            streamlines.append(
                {
                    "path": streamline,
                    "start_time": t_,
                    "deviation": np.linalg.norm(np.array([dx[i], dy[i]])),
                    "dilation": 3 * np.random.normal(1, 1),
                }
            )

    # if memoize:
    #    with open(cache_file, "wb") as f:
    #        pickle.dump(streamlines, f)

    if show:
        # Plot the streamlines
        print("Plotting streamlines...")
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.patch.set_facecolor("#000")
        ax.axis("off")

        # Collect all paths for LineCollection
        paths = [path["path"] for path in streamlines]
        colors = [plt.cm.plasma(i / len(streamlines)) for i in range(len(streamlines))]

        # Create a LineCollection
        lc = LineCollection(paths, colors=colors, linewidths=1, alpha=0.7)
        ax.add_collection(lc)
        ax.autoscale()

    return streamlines


def stylize(
    img,
    sigma=5,
    bilateral_d=40,
    bilateral_sigma_color=225,
    bilateral_sigma_space=25,
    emboss_kernel=None,
):
    """
    Apply stylization effects to an image.

    Parameters:
    img (PIL.Image or numpy.ndarray): The input image.
    sigma (float): Standard deviation for Gaussian noise. Default is 5.
    bilateral_d (int): Diameter of each pixel neighborhood for bilateral filter. Default is 40.
    bilateral_sigma_color (float): Filter sigma in the color space for bilateral filter. Default is 225.
    bilateral_sigma_space (float): Filter sigma in the coordinate space for bilateral filter. Default is 25.
    emboss_kernel (numpy.ndarray): Kernel for emboss effect. Default is None, which uses a predefined kernel.

    Returns:
    numpy.ndarray: The stylized image.
    """
    if emboss_kernel is None:
        emboss_kernel = np.array([[0, -1, -1], [1, 0.1, -1], [1, 1, 1]])

    # Convert PIL image to OpenCV format if necessary
    if isinstance(img, np.ndarray):
        img_cv = img
    else:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2RGB)

    # Convert the image to HSV color space
    img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)

    # Apply Gaussian noise to the saturation and value channels
    gaussian_noise_s = np.random.normal(0, sigma, img_hsv[:, :, 1].shape).astype(
        np.uint8
    )
    gaussian_noise_v = np.random.normal(0, sigma, img_hsv[:, :, 2].shape).astype(
        np.uint8
    )

    # Add the noise to the saturation and value channels
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] + gaussian_noise_s, 0, 255)
    img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] + gaussian_noise_v, 0, 255)

    # Convert the image back to RGB color space
    noisy_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    noisy_img = np.clip(noisy_img, 0, 255)

    # Apply Bilateral Filter
    blurred = cv2.bilateralFilter(
        noisy_img,
        d=bilateral_d,
        sigmaColor=bilateral_sigma_color,
        sigmaSpace=bilateral_sigma_space,
    )

    # Apply Emboss Effect
    stylized_img = cv2.filter2D(blurred, -1, emboss_kernel)

    return Image.fromarray(stylized_img)


def load_track(gpx_path):
    """
    Load a GPX file and extract the path.

    Parameters:
    gpx_path (str): The path to the GPX file.

    Returns:
    numpy.ndarray: The extracted path as an array of coordinates [latitude, longitude].
    """
    # Load GPX file
    with open(gpx_path, "r") as gpx_file:
        gpx = gpxpy.parse(gpx_file)

    # Extract path from GPX file
    path = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                path.append([point.latitude, point.longitude])

    path = np.array(path)
    return path


def add_border(img, border, user_name, event_date, font):
    # Get the transparent area bbox
    bbox = get_transparent_bbox(border)
    # Resize img to fit exactly in the transparent area
    img_resized = img.resize((bbox[2] - bbox[0], bbox[3] - bbox[1]))

    # Paste the resized image into the transparent area
    border.paste(img_resized, bbox)

    # Add text to the image
    draw = ImageDraw.Draw(border)

    # Add statistics text to the image
    if user_name:
        draw.text(
            (0.25 * border.width, 0.19 * border.height),
            f"By {user_name}",
            fill="black",
            font=font,
        )

    if event_date:
        draw.text(
            (0.1 * border.width, 0.75 * border.height),
            f"{event_date}",
            fill="black",
            font=font,
        )

    return border


def draw_frame(
    fig,
    ax,
    t,
    tmax,
    streamlines,
    palette,
    background_color="#000",
):
    """
    Draw a frame with streamlines and apply stylization effects.

    Parameters:
    fig (matplotlib.figure.Figure): The figure object.
    ax (matplotlib.axes.Axes): The axes object.
    t (float): The current time value.
    tmax (float): The maximum time value.
    streamlines (list): The list of streamlines.
    palette (list): The list of colors for the streamlines.

    Returns:
    None
    """
    # Pre-allocate lists with estimated capacity
    n = len(streamlines)
    patches = []
    colors = []
    linewidths = []

    palette_len = len(palette)

    # Vectorize calculations where possible
    if t is not None:
        start_times = np.array([p["start_time"] for p in streamlines])
        valid_paths = start_times <= t
        streamlines = [s for i, s in enumerate(streamlines) if valid_paths[i]]

    for path_i, path_data in enumerate(streamlines):
        path = path_data["path"]

        if path.size < 2:
            continue

        if t is not None:
            # Vectorized length calculation
            delta = np.diff(path, axis=0)
            length = np.cumsum(np.sqrt(np.sum(delta**2, axis=1)))
            length = np.insert(length, 0, 0)

            delta_time = t - path_data["start_time"]
            min_len = 0.2 * delta_time
            max_len = 0.8 * delta_time * (1 - 0.9 * t / tmax)

            mask = (length >= min_len) & (length <= max_len)

            if not mask.any():
                continue

            path = path[mask]
            if len(path) < 2:
                continue

        patches.append(path)
        colors.append(palette[path_i % palette_len])
        linewidths.append(path_data["dilation"])

    if patches:
        ax.add_collection(LineCollection(patches, colors=colors, linewidths=linewidths))
        ax.autoscale()


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def main(
    gpx_path,
    palette=["#000", "#00A3E0", "#D0D3D4", "#73FDEA", "#BFDD0D"],
    # palette=["#73FDEA", "#BFD000", "#E85B00", "#9B62F5"],
    show=True,
    animate=False,
    frames=400,
    figsize=(16, 9),
    dpi=300,
    apply_style=False,
    background="#000",
    memoize=False,
):
    """
    Main function to process a GPX file, compute vector fields, and generate streamlines.

    Parameters:
    gpx_path (str): The path to the GPX file.
    palette (list): The list of colors for the streamlines. Default is a predefined list of colors.
    show (bool): Flag to indicate whether to show plots. Default is True.
    animate (bool): Flag to indicate whether to create an animation. Default is False.
    frames (int): The number of frames for the animation. Default is 400.
    figsize (tuple): The size of the figure. Default is (16, 9).

    Returns:
    None
    """
    # Use cached values if the input file hasn't changed
    if not hasattr(main, "last_gpx_path") or main.last_gpx_path != gpx_path:
        # Load the GPX track
        path = load_track(gpx_path)

        # Process and smooth the path
        print("Processing and smoothing the path...")
        x, y, t = process_and_smooth_path(path, show=show)

        # Compute vector field
        print("Computing the vector field...")
        t_func, vector_field_for_plotting, differential_equation_for_integration = (
            compute_vector_field(x, y, t, show=show, seed=0)
        )

        # Compute streamlines
        print("Computing streamlines...")
        streamlines = compute_streamlines(
            x,
            y,
            t,
            differential_equation_for_integration,
            frames=frames,
            memoize=memoize,
            show=show,
        )

        # Cache the results
        main.last_gpx_path = gpx_path
        main.last_path = path
        main.last_x = x
        main.last_y = y
        main.last_t = t
        main.last_t_func = t_func
        main.last_vector_field = vector_field_for_plotting
        main.last_diff_eq = differential_equation_for_integration
        main.last_streamlines = streamlines
    else:
        # Use cached values
        path = main.last_path
        x = main.last_x
        y = main.last_y
        t = main.last_t
        t_func = main.last_t_func
        vector_field_for_plotting = main.last_vector_field
        differential_equation_for_integration = main.last_diff_eq
        streamlines = main.last_streamlines

    if animate:

        # Create frames and animations directory if it doesn't exist
        os.makedirs("frames", exist_ok=True)
        os.makedirs("animations", exist_ok=True)

        # Clean frames folder
        for f in os.listdir("frames"):
            os.remove(os.path.join("frames", f))

        # Create an animation
        print("Creating an animation...")
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ax.axis("off")
        fig.patch.set_facecolor("#111")
        camera = Camera(fig)

        dt = t.max() - t.min()
        for t_ in tqdm(
            2 * np.linspace(t.min(), t.max() - 0.2 * dt, frames)[: 3 * frames // 5]
        ):
            draw_frame(fig, ax, t_, t.max(), streamlines, palette)
            camera.snap()

        animation = camera.animate()
        animation_file_gif = os.path.join(
            "animations", os.path.splitext(os.path.basename(gpx_path))[0] + ".gif"
        )
        animation_file_mp4 = os.path.join(
            "animations", os.path.splitext(os.path.basename(gpx_path))[0] + ".mp4"
        )
        animation.save(animation_file_gif, writer=PillowWriter(fps=30))
        animation.save(animation_file_mp4, writer="ffmpeg", fps=30)
        plt.close()

        if apply_style:
            # Apply stylization effects to each frame of the animation
            print("Applying stylization effects...")

            # Extract frames from the GIF at original resolution
            os.system(f"ffmpeg -i {animation_file_gif} frames/frame_%04d.png")

            # Apply stylization to each frame
            frame_files = sorted(
                [
                    f
                    for f in os.listdir("frames")
                    if f.startswith("frame_") and f.endswith(".png")
                ]
            )
            for frame_file in tqdm(frame_files):
                img = Image.open(os.path.join("frames", frame_file))
                stylized_img = stylize(img)
                stylized_img.save(os.path.join("frames", frame_file))

            # Combine frames back into a GIF at original resolution
            stylized_animation_file = os.path.join(
                "animations",
                os.path.splitext(os.path.basename(gpx_path))[0] + "_stylized.gif",
            )
            stylized_animation_file_mp4 = os.path.join(
                "animations",
                os.path.splitext(os.path.basename(gpx_path))[0] + "_stylized.mp4",
            )
            os.system(
                f"ffmpeg -framerate 40 -i frames/frame_%04d.png {stylized_animation_file} -y"
            )
            os.system(
                f"ffmpeg -framerate 40 -i frames/frame_%04d.png -c:v libx264 -crf 18 -preset slow -r 30 -pix_fmt yuv420p {stylized_animation_file_mp4} -y"
            )

            # Clean up frame files
            for frame_file in frame_files:
                os.remove(os.path.join("frames", frame_file))

            # Display the stylized animation
            display(IImage(filename=stylized_animation_file))
        else:
            # Display the animation
            display(IImage(filename=animation_file_gif))

    else:
        # Draw a frame
        print("Drawing a frame...")
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True, dpi=dpi)
        ax.axis("off")
        fig.patch.set_facecolor(background)
        draw_frame(fig, ax, 0.5 * t.max(), t.max(), streamlines, palette)
        return fig, ax
        # Convert the matplotlib figure to a PIL image
        plt.savefig("final.png", dpi=dpi)
        img = fig2img(fig)
        plt.close()
        os.makedirs("prints", exist_ok=True)
        img.save(
            os.path.join(
                "prints", os.path.splitext(os.path.basename(gpx_path))[0] + ".png"
            )
        )

        img = img.resize(
            (int(img.width / 2), int(img.height / 2)), Image.Resampling.NEAREST
        )
        stylized_img = stylize(
            img,
            sigma=0,
            bilateral_d=60,
            bilateral_sigma_color=225,
            bilateral_sigma_space=25,
        )
        stylized_img.save(
            os.path.join(
                "prints",
                os.path.splitext(os.path.basename(gpx_path))[0] + "_stylized.png",
            )
        )
