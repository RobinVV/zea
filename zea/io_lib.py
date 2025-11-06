"""Input / output functions for reading and writing files.

Use to quickly read and write files or interact with file system.
"""

import functools
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Generator

import imageio
import numpy as np
from PIL import Image, ImageSequence

from zea import log

_SUPPORTED_VID_TYPES = [".mp4", ".gif"]
_SUPPORTED_IMG_TYPES = [".jpg", ".png", ".JPEG", ".PNG", ".jpeg"]
_SUPPORTED_ZEA_TYPES = [".hdf5", ".h5"]


def load_video(filename, mode="L"):
    """Load a video file and return a numpy array of frames.

    Supported file types: mp4, gif.

    Args:
        filename (str): The path to the video file.
        mode (str, optional): Color mode: "L" (grayscale) or "RGB".
            Defaults to "L".

    Returns:
        numpy.ndarray: Array of frames (num_frames, H, W) or (num_frames, H, W, C)

    Raises:
        ValueError: If the file extension or mode is not supported.
    """
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"File {filename} does not exist")
    ext = filename.suffix.lower()

    if ext not in _SUPPORTED_VID_TYPES:
        raise ValueError(f"Unsupported file extension: {ext}")

    if mode not in {"L", "RGB"}:
        raise ValueError(f"Unsupported mode: {mode}")

    frames = []

    if ext == ".gif":
        with Image.open(filename) as im:
            for frame in ImageSequence.Iterator(im):
                frames.append(_convert_image_mode(frame, mode=mode))
    elif ext == ".mp4":
        # Use imageio with FFMPEG format for MP4 files
        try:
            reader = imageio.get_reader(filename, format="FFMPEG")
        except (ImportError, ValueError) as exc:
            raise ImportError(
                "FFMPEG plugin is required to load MP4 files. "
                "Please install it with 'pip install imageio-ffmpeg'."
            ) from exc

        try:
            for frame in reader:
                img = Image.fromarray(frame)
                frames.append(_convert_image_mode(img, mode=mode))
        finally:
            reader.close()
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    return np.stack(frames, axis=0)


def load_image(filename, mode="L"):
    """Load an image file and return a numpy array.

    Supported file types: jpg, png.

    Args:
        filename (str): The path to the image file.
        mode (str, optional): Color mode: "L" (grayscale) or "RGB".
            Defaults to "L".

    Returns:
        numpy.ndarray: A numpy array of the image.

    Raises:
        ValueError: If the file extension or mode is not supported.
    """
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"File {filename} does not exist")
    extension = filename.suffix.lower()
    allowed_exts = {ext.lower() for ext in _SUPPORTED_IMG_TYPES}
    if extension not in allowed_exts:
        raise ValueError(f"File extension {extension} not supported")

    if mode not in {"L", "RGB"}:
        raise ValueError(f"Unsupported mode: {mode}")

    with Image.open(filename) as img:
        return _convert_image_mode(img, mode=mode)


def save_video(images, filename, fps=20, **kwargs):
    """Saves a sequence of images to a video file.

    Supported file types: mp4, gif.

    Args:
        images (list or np.ndarray): List or array of images. Must have shape
            (n_frames, height, width, channels) or (n_frames, height, width).
            If channel axis is not present, or is 1, grayscale image is assumed,
            which is then converted to RGB. Images should be uint8.
        filename (str or Path): Filename to which data should be written.
        fps (int): Frames per second of rendered format.
        **kwargs: Additional keyword arguments passed to the specific save function.
            For GIF files, this includes `shared_color_palette` (bool).

    Raises:
        ValueError: If the file extension is not supported.

    """
    filename = Path(filename)
    ext = filename.suffix.lower()

    if ext == ".mp4":
        return save_to_mp4(images, filename, fps=fps)
    elif ext == ".gif":
        return save_to_gif(images, filename, fps=fps, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def save_to_gif(images, filename, fps=20, shared_color_palette=False):
    """Saves a sequence of images to a GIF file.

    .. note::
        It's recommended to use :func:`save_video` for a more general interface
        that supports multiple formats.

    Args:
        images (list or np.ndarray): List or array of images. Must have shape
            (n_frames, height, width, channels) or (n_frames, height, width).
            If channel axis is not present, or is 1, grayscale image is assumed,
            which is then converted to RGB. Images should be uint8.
        filename (str or Path): Filename to which data should be written.
        fps (int): Frames per second of rendered format.
        shared_color_palette (bool, optional): If True, creates a global
            color palette across all frames, ensuring consistent colors
            throughout the GIF. Defaults to False, which is default behavior
            of PIL.Image.save. Note: True can cause slow saving for longer
            sequences, and also lead to larger file sizes in some cases.

    """
    images = preprocess_for_saving(images)

    if fps > 50:
        log.warning(f"Cannot set fps ({fps}) > 50. Setting it automatically to 50.")
        fps = 50

    duration = int(round(1000 / fps))  # milliseconds per frame

    pillow_imgs = [Image.fromarray(img) for img in images]

    if shared_color_palette:
        # Apply the same palette to all frames without dithering for consistent color mapping
        # Convert all images to RGB and combine their colors for palette generation
        all_colors = np.vstack([np.array(img.convert("RGB")).reshape(-1, 3) for img in pillow_imgs])
        combined_image = Image.fromarray(all_colors.reshape(-1, 1, 3))

        # Generate palette from all frames
        global_palette = combined_image.quantize(
            colors=256,
            method=Image.MEDIANCUT,
            kmeans=1,
        )

        # Apply the same palette to all frames without dithering
        pillow_imgs = [
            img.convert("RGB").quantize(
                palette=global_palette,
                dither=Image.NONE,
            )
            for img in pillow_imgs
        ]

    pillow_img, *pillow_imgs = pillow_imgs

    pillow_img.save(
        fp=filename,
        format="GIF",
        append_images=pillow_imgs,
        save_all=True,
        loop=0,
        duration=duration,
        interlace=False,
        optimize=False,
    )
    log.success(f"Successfully saved GIF to -> {log.yellow(filename)}")


def save_to_mp4(images, filename, fps=20):
    """Saves a sequence of images to an MP4 file.

    .. note::
        It's recommended to use :func:`save_video` for a more general interface
        that supports multiple formats.

    Args:
        images (list or np.ndarray): List or array of images. Must have shape
            (n_frames, height, width, channels) or (n_frames, height, width).
            If channel axis is not present, or is 1, grayscale image is assumed,
            which is then converted to RGB. Images should be uint8.
        filename (str or Path): Filename to which data should be written.
        fps (int): Frames per second of rendered format.

    Raises:
        ImportError: If imageio-ffmpeg is not installed.

    Returns:
        str: Success message.

    """
    images = preprocess_for_saving(images)

    filename = str(filename)

    parent_dir = Path(filename).parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    # Use imageio with FFMPEG format for MP4 files
    try:
        writer = imageio.get_writer(
            filename, fps=fps, format="FFMPEG", codec="libx264", pixelformat="yuv420p"
        )
    except (ImportError, ValueError) as exc:
        raise ImportError(
            "FFMPEG plugin is required to save MP4 files. "
            "Please install it with 'pip install imageio-ffmpeg'."
        ) from exc

    try:
        for image in images:
            writer.append_data(image)
    finally:
        writer.close()

    return log.success(f"Successfully saved MP4 to -> {filename}")


def search_file_tree(directory, filetypes=None, verbose=True, relative=False) -> Generator:
    """Traverse a directory tree and yield file paths matching specified file types.

    Args:
        directory (str or Path): The root directory to start the search.
        filetypes (list of str, optional): List of file extensions to match.
            If None, file types supported by `zea` are matched. Defaults to None.
        verbose (bool, optional): If True, logs the search process. Defaults to True.
        relative (bool, optional): If True, yields file paths relative to the
            root directory. Defaults to False.

    Yields:
        Path: Paths of files matching the specified file types.
    """
    # Traverse file tree to index all files from filetypes
    if verbose:
        log.info(f"Searching {log.yellow(directory)} for {filetypes} files...")

    for dirpath, _, filenames in os.walk(directory):
        for file in filenames:
            # Append to file_paths if it is a filetype file
            if Path(file).suffix in filetypes:
                file_path = Path(dirpath) / file
                if relative:
                    file_path = file_path.relative_to(directory)
                yield file_path


def matplotlib_figure_to_numpy(fig, **kwargs):
    """Convert matplotlib figure to numpy array.

    Args:
        fig (matplotlib.figure.Figure): figure to convert.

    Returns:
        np.ndarray: numpy array of figure.

    """
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", **kwargs)
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    image = np.array(image)[..., :3]
    buf.close()
    return image


def retry_on_io_error(max_retries=3, initial_delay=0.5, retry_action=None):
    """Decorator to retry functions on I/O errors with exponential backoff.

    Args:
        max_retries (int): Maximum number of retry attempts.
        initial_delay (float): Initial delay between retries in seconds.
        retry_action (callable, optional): Optional function to call before each retry attempt.
            If decorating a method: ``retry_action(self, exception, attempt, *args, **kwargs)``
            If decorating a function: ``retry_action(exception, attempt, *args, **kwargs)``

    Returns:
        callable: Decorated function with retry logic.

    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (OSError, IOError) as e:
                    last_exception = e

                    # if args exist and first arg is a class, update retry count of that method
                    if args and hasattr(args[0], "retry_count"):
                        args[0].retry_count = attempt + 1

                    if attempt < max_retries - 1:
                        # Execute custom retry action if provided
                        if retry_action:
                            # Pass all original arguments to retry_action
                            retry_action(
                                *args,
                                exception=e,
                                retry_count=attempt,
                                **kwargs,
                            )

                        time.sleep(delay)

                    else:
                        # Last attempt failed
                        log.error(f"Failed after {max_retries} attempts: {e}")

            # If we've exhausted all retries
            raise ValueError(
                f"Failed to complete operation after {max_retries} attempts. "
                f"Last error: {last_exception}"
            )

        return wrapper

    return decorator


def _convert_image_mode(img, mode="L"):
    """Convert a PIL Image to the specified mode and return as numpy array."""
    if mode not in {"L", "RGB"}:
        raise ValueError(f"Unsupported mode: {mode}, must be one of: L, RGB")
    if mode == "L":
        img = img.convert("L")
        arr = np.array(img)
    elif mode == "RGB":
        img = img.convert("RGB")
        arr = np.array(img)
    return arr


def grayscale_to_rgb(image):
    """Converts a grayscale image to an RGB image.

    Args:
        image (ndarray): Grayscale image. Must have shape (height, width).

    Returns:
        ndarray: RGB image.
    """
    assert image.ndim == 2, "Input image must be grayscale."
    # Stack the grayscale image into 3 channels (RGB)
    return np.stack([image] * 3, axis=-1)


def _assert_uint8_images(images: np.ndarray):
    """
    Asserts that the input images have the correct properties.

    Args:
        images (np.ndarray): The input images.

    Raises:
        AssertionError: If the dtype of images is not uint8.
        AssertionError: If the shape of images is not (n_frames, height, width, channels)
            or (n_frames, height, width) for grayscale images.
        AssertionError: If images have anything other than 1 (grayscale),
            3 (rgb) or 4 (rgba) channels.
    """
    assert images.dtype == np.uint8, f"dtype of images should be uint8, got {images.dtype}"

    assert images.ndim in (3, 4), (
        "images must have shape (n_frames, height, width, channels),"
        f" or (n_frames, height, width) for grayscale images. Got {images.shape}"
    )

    if images.ndim == 4:
        assert images.shape[-1] in (1, 3, 4), (
            "Grayscale images must have 1 channel, "
            "RGB images must have 3 channels, and RGBA images must have 4 channels. "
            f"Got shape: {images.shape}, channels: {images.shape[-1]}"
        )


def preprocess_for_saving(images):
    """Preprocesses images for saving to GIF or MP4.

    Args:
        images (ndarray, list[ndarray]): Images. Must have shape (n_frames, height, width, channels)
            or (n_frames, height, width).
    """
    images = np.array(images)
    _assert_uint8_images(images)

    # Remove channel axis if it is 1 (grayscale image)
    if images.ndim == 4 and images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)

    # convert grayscale images to RGB
    if images.ndim == 3:
        images = [grayscale_to_rgb(image) for image in images]
        images = np.array(images)

    # drop alpha channel if present (RGBA -> RGB)
    if images.ndim == 4 and images.shape[-1] == 4:
        images = images[..., :3]

    return images
