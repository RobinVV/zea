"""Pixel grid calculation for ultrasound beamforming."""

import numpy as np

from zea import log

eps = 1e-10


def check_for_aliasing(scan):
    """Checks if the scan class parameters will cause spatial aliasing due to a too low pixel
    density. If so, a warning is printed with a suggestion to increase the pixel density by either
    increasing the number of pixels, or decreasing the pixel spacing, depending on which parameter
    was set by the user."""
    width = scan.xlims[1] - scan.xlims[0]
    depth = scan.zlims[1] - scan.zlims[0]
    wvln = scan.wavelength

    if width / scan.grid_size_x > wvln / 2:
        log.warning(
            f"width/grid_size_x = {width / scan.grid_size_x:.7f} < wavelength/2 = {wvln / 2}. "
            f"Consider either increasing scan.grid_size_x to {int(np.ceil(width / (wvln / 2)))} "
            "or more, or increasing scan.pixels_per_wavelength to 2 or more."
        )
    if depth / scan.grid_size_z > wvln / 2:
        log.warning(
            f"depth/grid_size_z = {depth / scan.grid_size_z:.7f} < wavelength/2 = {wvln / 2:.7f}. "
            f"Consider either increasing scan.grid_size_z to {int(np.ceil(depth / (wvln / 2)))} "
            "or more, or increasing scan.pixels_per_wavelength to 2 or more."
        )


def cartesian_pixel_grid(
    xlims,
    zlims,
    ylims=None,
    grid_size_x=None,
    grid_size_y=None,
    grid_size_z=None,
    dx=None,
    dy=None,
    dz=None,
):
    """Generate a Cartesian pixel grid.

    Behaviour:
      - If ylims has zero extent (abs(ymax - ymin) < eps) the function returns a 2D grid
        with shape (nz, nx, 3) that contains (x, y=0, z) per-pixel (y omitted as a dimension).
      - If ylims has non-zero extent the function returns a 3D grid with shape
        (nz, nx, ny, 3) containing (x, y, z) per-voxel.

    Args:
        xlims (tuple): [xmin, xmax]
        ylims (tuple): [ymin, ymax] — if ymax == ymin (within tol) treated as "no y extent"
        zlims (tuple): [zmin, zmax]
        grid_size_x, grid_size_y, grid_size_z (int): number of samples along each axis.
            For 2D (no y extent) only grid_size_x and grid_size_z are required if using sizes.
        dx, dy, dz (float): spacings along axes.
            For 2D, only dx and dz are required if using spacings.

    Returns:
        np.ndarray:
            - 2D: shape (nz, nx, 3) with per-pixel [x, y, z] (y will be zeros)
            - 3D: shape (nz, nx, ny, 3) with per-voxel [x, y, z]
    """
    # decide whether y dimension exists
    is_3d = ylims is not None and abs(ylims[1] - ylims[0]) > eps

    # helper booleans
    sizes_all = (
        (grid_size_x is not None) and (grid_size_z is not None) and (grid_size_y is not None)
    )
    spacings_all = (dx is not None) and (dz is not None) and (dy is not None)

    # For 2D mode, we accept sizes/spacings that omit y
    if not is_3d:
        sizes_2d = (grid_size_x is not None) and (grid_size_z is not None)
        spacings_2d = (dx is not None) and (dz is not None)
        # exactly one of sizes_2d or spacings_2d must be True
        if sizes_2d == spacings_2d:
            raise ValueError(
                "For 2D (no y extent) either provide grid_size_x & grid_size_z "
                "OR provide dx & dz (but not both)."
            )
    else:
        # 3D: must provide either all three sizes OR all three spacings (exclusive)
        if sizes_all == spacings_all:
            raise ValueError(
                "For 3D (non-zero y extent) either provide grid_size_x/grid_size_y/grid_size_z "
                "OR provide dx/dy/dz (but not both)."
            )

    # Build coordinate vectors
    # X and Z (always present)
    if (
        (grid_size_x is not None)
        and (grid_size_z is not None)
        and (not is_3d or grid_size_y is not None)
    ):
        # size-based for whichever mode
        x = np.linspace(xlims[0], xlims[1] + eps, grid_size_x)
        z = np.linspace(zlims[0], zlims[1] + eps, grid_size_z)
        if is_3d:
            y = np.linspace(ylims[0], ylims[1] + eps, grid_size_y)
    else:
        # spacing-based: sign-aware arange
        sign_x = np.sign(xlims[1] - xlims[0]) if xlims[1] != xlims[0] else 1.0
        sign_z = np.sign(zlims[1] - zlims[0]) if zlims[1] != zlims[0] else 1.0
        x = np.arange(xlims[0], xlims[1] + sign_x * eps, sign_x * dx)
        z = np.arange(zlims[0], zlims[1] + sign_z * eps, sign_z * dz)
        if is_3d:
            sign_y = np.sign(ylims[1] - ylims[0]) if ylims[1] != ylims[0] else 1.0
            y = np.arange(ylims[0], ylims[1] + sign_y * eps, sign_y * dy)

    # Build grids
    if not is_3d:
        # 2D: ensure shapes (nz, nx)
        # meshgrid with z (first axis) and x (second axis) to get (nz, nx)
        z_grid, x_grid = np.meshgrid(z, x, indexing="ij")
        y_grid = np.zeros_like(x_grid)
        grid = np.stack((x_grid, y_grid, z_grid), axis=-1)
        return grid
    else:
        # 3D: user requested output shape (grid_size_z, grid_size_x, grid_size_y, 3)
        # so we meshgrid in the order (z, x, y) which yields arrays with shape (nz, nx, ny)
        z_grid, x_grid, y_grid = np.meshgrid(z, x, y, indexing="ij")
        grid = np.stack((x_grid, y_grid, z_grid), axis=-1)
        return grid


def radial_pixel_grid(rlims, dr, oris, dirs):
    """Generate a focused pixel grid based on input parameters.

    To accommodate the multitude of ways of defining a focused transmit grid, we define
    pixel "rays" or "lines" according to their origins (oris) and directions (dirs).
    The position along the ray is defined by its limits (rlims) and spacing (dr).

    Args:
        rlims (tuple): Radial limits of pixel grid ([rmin, rmax])
        dr (float): Pixel spacing in radius
        oris (np.ndarray): Origin of each ray in Cartesian coordinates (x, y, z)
            with shape (nrays, 3)
        dirs (np.ndarray): Steering direction of each ray in azimuth, in units of
            radians (nrays, 2)

    Returns:
        grid (np.ndarray): Pixel grid of size (nr, nrays, 3) in
            Cartesian coordinates (x, y, z), with nr being the number of radial pixels.
    """
    # Get focusing positions in rho-theta coordinates
    r = np.arange(rlims[0], rlims[1], dr)  # Depth rho
    t = dirs[:, 0]  # Use azimuthal angle theta (ignore elevation angle)
    tt, rr = np.meshgrid(t, r, indexing="ij")

    # Convert the focusing grid to Cartesian coordinates
    xx = rr * np.sin(tt) + oris[:, [0]]
    zz = rr * np.cos(tt) + oris[:, [2]]
    yy = 0 * xx
    grid = np.stack((xx, yy, zz), axis=-1)
    return grid


def polar_pixel_grid(polar_limits, zlims, num_radial_pixels: int, num_polar_pixels: int):
    """Generate a polar grid.

    Uses radial_pixel_grid but based on parameters that are present in the scan class.

    Args:
        polar_limits (tuple): Polar limits of pixel grid ([polar_min, polar_max])
        zlims (tuple): Depth limits of pixel grid ([zmin, zmax])
        num_radial_pixels (int, optional): Number of depth pixels.
        num_polar_pixels (int, optional): Number of polar pixels.

    Returns:
        grid (np.ndarray): Pixel grid of size (num_radial_pixels, num_polar_pixels, 3)
        in Cartesian coordinates (x, y, z)
    """
    assert len(polar_limits) == 2, "polar_limits must be a tuple of length 2."
    assert len(zlims) == 2, "zlims must be a tuple of length 2."

    dr = (zlims[1] - zlims[0]) / num_radial_pixels

    oris = np.array([0, 0, 0])
    oris = np.tile(oris, (num_polar_pixels, 1))
    dirs_az = np.linspace(*polar_limits, num_polar_pixels)

    dirs_el = np.zeros(num_polar_pixels)
    dirs = np.vstack((dirs_az, dirs_el)).T
    return radial_pixel_grid(zlims, dr, oris, dirs).transpose(1, 0, 2)
