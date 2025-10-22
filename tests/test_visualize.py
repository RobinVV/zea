"""Tests for the visualize module."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from zea.visualize import (
    pad_or_crop_extent,
    plot_biplanes,
    plot_frustum_vertices,
    plot_image_grid,
    plot_quadrants,
    set_mpl_style,
    visualize_matrix,
)

# Use non-interactive backend for testing
matplotlib.use("Agg")


class TestPlotImageGrid:
    """Tests for the plot_image_grid function."""

    def test_plot_image_grid_basic(self):
        """Test basic functionality of plot_image_grid."""
        images = [np.random.rand(10, 10) for _ in range(4)]
        fig, fig_contents = plot_image_grid(images)

        assert isinstance(fig, plt.Figure)
        assert isinstance(fig_contents, list)
        assert len(fig_contents) == 4
        plt.close(fig)

    def test_plot_image_grid_with_titles(self):
        """Test plot_image_grid with titles."""

        images = [np.random.rand(10, 10) for _ in range(4)]
        titles = ["Image 1", "Image 2", "Image 3", "Image 4"]
        fig, fig_contents = plot_image_grid(images, titles=titles)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 4
        plt.close(fig)

    def test_plot_image_grid_custom_cmap(self):
        """Test plot_image_grid with custom colormap."""

        images = [np.random.rand(10, 10) for _ in range(2)]
        fig, fig_contents = plot_image_grid(images, cmap=["viridis", "plasma"])

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_image_grid_vmin_vmax(self):
        """Test plot_image_grid with vmin and vmax."""

        images = [np.random.rand(10, 10) for _ in range(2)]
        fig, fig_contents = plot_image_grid(images, vmin=[0, 0.2], vmax=[0.8, 1.0])

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotBiplanes:
    """Tests for the plot_biplanes function."""

    def test_plot_biplanes_basic(self):
        """Test basic functionality of plot_biplanes."""

        volume = np.random.rand(20, 20, 20)
        fig, ax = plot_biplanes(volume, slice_x=10, slice_y=10, slice_z=10)

        assert isinstance(fig, plt.Figure)
        assert ax is not None
        plt.close(fig)

    def test_plot_biplanes_single_slice(self):
        """Test plot_biplanes with single slice."""

        volume = np.random.rand(20, 20, 20)
        fig, ax = plot_biplanes(volume, slice_x=10)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_biplanes_no_slice_raises(self):
        """Test that plot_biplanes raises error when no slice is specified."""

        volume = np.random.rand(20, 20, 20)
        with pytest.raises(AssertionError):
            plot_biplanes(volume)

    def test_plot_biplanes_with_resolution(self):
        """Test plot_biplanes with custom resolution."""

        volume = np.random.rand(20, 20, 20)
        fig, ax = plot_biplanes(volume, resolution=0.5, slice_x=10)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_biplanes_reuse_fig_ax(self):
        """Test plot_biplanes with reused figure and axes."""

        volume = np.random.rand(20, 20, 20)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        fig_out, ax_out = plot_biplanes(volume, slice_x=10, fig=fig, ax=ax)

        assert fig_out is fig
        assert ax_out is ax
        plt.close(fig)


class TestPlotQuadrants:
    """Tests for the plot_quadrants function."""

    def test_plot_quadrants_x_coord(self):
        """Test plot_quadrants with fixed x coordinate."""
        volume = np.random.rand(20, 20, 20)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax_out = plot_quadrants(ax, volume, "x", "gray", slice_index=10)

        assert ax_out is ax
        # Should have 4 surfaces plotted (one for each quadrant)
        assert len(ax.collections) == 4
        plt.close(fig)

    def test_plot_quadrants_y_coord(self):
        """Test plot_quadrants with fixed y coordinate."""
        volume = np.random.rand(20, 20, 20)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax_out = plot_quadrants(ax, volume, "y", "viridis", slice_index=10)

        assert ax_out is ax
        assert len(ax.collections) == 4
        plt.close(fig)

    def test_plot_quadrants_z_coord(self):
        """Test plot_quadrants with fixed z coordinate."""
        volume = np.random.rand(20, 20, 20)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax_out = plot_quadrants(ax, volume, "z", "plasma", slice_index=10)

        assert ax_out is ax
        assert len(ax.collections) == 4
        plt.close(fig)

    def test_plot_quadrants_none_slice_index(self):
        """Test plot_quadrants with None slice_index (should use middle)."""
        volume = np.random.rand(20, 20, 20)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax_out = plot_quadrants(ax, volume, "x", "gray", slice_index=None)

        assert ax_out is ax
        assert len(ax.collections) == 4
        plt.close(fig)

    def test_plot_quadrants_custom_stride(self):
        """Test plot_quadrants with custom stride."""
        volume = np.random.rand(20, 20, 20)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax_out = plot_quadrants(ax, volume, "z", "gray", slice_index=10, stride=2)

        assert ax_out is ax
        assert len(ax.collections) == 4
        plt.close(fig)

    def test_plot_quadrants_custom_centroid(self):
        """Test plot_quadrants with custom centroid."""
        volume = np.random.rand(20, 20, 20)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        centroid = [15, 15, 15]
        ax_out = plot_quadrants(ax, volume, "x", "gray", slice_index=10, centroid=centroid)

        assert ax_out is ax
        assert len(ax.collections) == 4
        plt.close(fig)

    def test_plot_quadrants_with_kwargs(self):
        """Test plot_quadrants with additional kwargs."""
        volume = np.random.rand(20, 20, 20)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax_out = plot_quadrants(
            ax, volume, "y", "gray", slice_index=10, alpha=0.5, antialiased=False
        )

        assert ax_out is ax
        assert len(ax.collections) == 4
        plt.close(fig)


class TestPlotFrustumVertices:
    """Tests for the plot_frustum_vertices function."""

    def test_plot_frustum_basic_phi_plane(self):
        """Test basic functionality with phi plane."""

        rho_range = [0.1, 10]
        theta_range = [-0.6, 0.6]
        phi_range = [-0.6, 0.6]

        fig, ax = plot_frustum_vertices(
            rho_range,
            theta_range=theta_range,
            phi_range=phi_range,
            phi_plane=0,
        )

        assert isinstance(fig, plt.Figure)
        assert ax is not None
        plt.close(fig)

    def test_plot_frustum_basic_theta_plane(self):
        """Test basic functionality with theta plane."""

        rho_range = [0.1, 10]
        theta_range = [-0.6, 0.6]
        phi_range = [-0.6, 0.6]

        fig, ax = plot_frustum_vertices(
            rho_range,
            theta_range=theta_range,
            phi_range=phi_range,
            theta_plane=0.2,
        )

        assert isinstance(fig, plt.Figure)
        assert ax is not None
        plt.close(fig)

    def test_plot_frustum_basic_rho_plane(self):
        """Test basic functionality with rho plane."""

        rho_range = [0.1, 10]
        theta_range = [-0.6, 0.6]
        phi_range = [-0.6, 0.6]

        fig, ax = plot_frustum_vertices(
            rho_range,
            theta_range=theta_range,
            phi_range=phi_range,
            rho_plane=5.0,
        )

        assert isinstance(fig, plt.Figure)
        assert ax is not None
        plt.close(fig)

    def test_plot_frustum_multiple_planes(self):
        """Test with multiple planes of each type."""

        rho_range = [0.1, 10]
        theta_range = [-0.6, 0.6]
        phi_range = [-0.6, 0.6]

        fig, ax = plot_frustum_vertices(
            rho_range,
            theta_range=theta_range,
            phi_range=phi_range,
            phi_plane=[0, 0.3],
            theta_plane=[0.2, -0.2],
            rho_plane=[2.0, 5.0, 8.0],
        )

        assert isinstance(fig, plt.Figure)
        assert ax is not None
        plt.close(fig)

    def test_plot_frustum_no_plane_raises(self):
        """Test that plot_frustum_vertices raises error when no plane is specified."""

        rho_range = [0.1, 10]
        theta_range = [-0.6, 0.6]
        phi_range = [-0.6, 0.6]

        with pytest.raises(ValueError, match="At least one plane must be specified"):
            plot_frustum_vertices(
                rho_range,
                theta_range=theta_range,
                phi_range=phi_range,
            )

    def test_plot_frustum_custom_frustum_style(self):
        """Test custom frustum style."""

        rho_range = [0.1, 10]
        theta_range = [-0.6, 0.6]
        phi_range = [-0.6, 0.6]

        frustum_style = {"color": "cyan", "linestyle": "-", "linewidth": 3, "alpha": 0.8}

        fig, ax = plot_frustum_vertices(
            rho_range,
            theta_range=theta_range,
            phi_range=phi_range,
            phi_plane=0,
            frustum_style=frustum_style,
        )

        assert isinstance(fig, plt.Figure)
        # Verify that lines were plotted (frustum edges + plane edges)
        assert len(ax.lines) > 0
        plt.close(fig)

    def test_plot_frustum_custom_phi_style(self):
        """Test custom phi plane style."""

        rho_range = [0.1, 10]
        theta_range = [-0.6, 0.6]
        phi_range = [-0.6, 0.6]

        phi_style = {"color": "red", "linestyle": "--", "linewidth": 2, "alpha": 0.7}

        fig, ax = plot_frustum_vertices(
            rho_range,
            theta_range=theta_range,
            phi_range=phi_range,
            phi_plane=0,
            phi_style=phi_style,
        )

        assert isinstance(fig, plt.Figure)
        assert len(ax.lines) > 0
        plt.close(fig)

    def test_plot_frustum_custom_theta_style(self):
        """Test custom theta plane style."""

        rho_range = [0.1, 10]
        theta_range = [-0.6, 0.6]
        phi_range = [-0.6, 0.6]

        theta_style = {"color": "green", "linestyle": ":", "linewidth": 1.5, "alpha": 0.9}

        fig, ax = plot_frustum_vertices(
            rho_range,
            theta_range=theta_range,
            phi_range=phi_range,
            theta_plane=0.2,
            theta_style=theta_style,
        )

        assert isinstance(fig, plt.Figure)
        assert len(ax.lines) > 0
        plt.close(fig)

    def test_plot_frustum_custom_rho_style(self):
        """Test custom rho plane style."""

        rho_range = [0.1, 10]
        theta_range = [-0.6, 0.6]
        phi_range = [-0.6, 0.6]

        rho_style = {"color": "magenta", "linestyle": "-.", "linewidth": 2.5}

        fig, ax = plot_frustum_vertices(
            rho_range,
            theta_range=theta_range,
            phi_range=phi_range,
            rho_plane=5.0,
            rho_style=rho_style,
        )

        assert isinstance(fig, plt.Figure)
        assert len(ax.lines) > 0
        plt.close(fig)

    def test_plot_frustum_all_custom_styles(self):
        """Test with all custom styles applied simultaneously."""

        rho_range = [0.1, 10]
        theta_range = [-0.6, 0.6]
        phi_range = [-0.6, 0.6]

        frustum_style = {"color": "blue", "linewidth": 1.5, "alpha": 0.6}
        phi_style = {"color": "red", "linestyle": "--", "linewidth": 2}
        theta_style = {"color": "green", "linestyle": ":", "alpha": 0.7}
        rho_style = {"color": "yellow", "linestyle": "-.", "linewidth": 2.5}

        fig, ax = plot_frustum_vertices(
            rho_range,
            theta_range=theta_range,
            phi_range=phi_range,
            phi_plane=0,
            theta_plane=0.2,
            rho_plane=5.0,
            frustum_style=frustum_style,
            phi_style=phi_style,
            theta_style=theta_style,
            rho_style=rho_style,
        )

        assert isinstance(fig, plt.Figure)
        assert ax is not None
        # Should have many lines: frustum edges + 3 sets of plane edges
        assert len(ax.lines) > 10
        plt.close(fig)

    def test_plot_frustum_reuse_fig_ax(self):
        """Test plot_frustum_vertices with reused figure and axes."""

        rho_range = [0.1, 10]
        theta_range = [-0.6, 0.6]
        phi_range = [-0.6, 0.6]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        fig_out, ax_out = plot_frustum_vertices(
            rho_range,
            theta_range=theta_range,
            phi_range=phi_range,
            phi_plane=0,
            fig=fig,
            ax=ax,
        )

        assert fig_out is fig
        assert ax_out is ax
        plt.close(fig)

    def test_plot_frustum_custom_num_points(self):
        """Test with custom number of points."""

        rho_range = [0.1, 10]
        theta_range = [-0.6, 0.6]
        phi_range = [-0.6, 0.6]

        fig, ax = plot_frustum_vertices(
            rho_range,
            theta_range=theta_range,
            phi_range=phi_range,
            phi_plane=0,
            num_points=50,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestVisualizationHelpers:
    """Tests for helper visualization functions."""

    def test_set_mpl_style_default(self):
        """Test set_mpl_style with default style."""

        set_mpl_style()
        # Should not raise an error

    def test_visualize_matrix(self):
        """Test visualize_matrix function."""

        matrix = np.random.rand(5, 5)
        fig = visualize_matrix(matrix)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_visualize_matrix_custom_colors(self):
        """Test visualize_matrix with custom colors."""

        matrix = np.random.rand(3, 3)
        fig = visualize_matrix(matrix, font_color="black", cmap="viridis")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPadOrCropExtent:
    """Tests for the pad_or_crop_extent function."""

    def test_pad_only(self):
        """Test padding an image to a larger extent."""

        image = np.ones((10, 10))
        extent = (0, 10, 0, 10)  # x_min, x_max, z_min, z_max
        target_extent = (-5, 15, -5, 15)  # Larger extent

        result = pad_or_crop_extent(image, extent, target_extent)

        # Should be larger than original
        assert result.shape[0] > image.shape[0]
        assert result.shape[1] > image.shape[1]

    def test_crop_only(self):
        """Test cropping an image to a smaller extent."""

        image = np.ones((20, 20))
        extent = (0, 20, 0, 20)  # x_min, x_max, z_min, z_max
        target_extent = (5, 15, 5, 15)  # Smaller extent

        result = pad_or_crop_extent(image, extent, target_extent)

        # Should be smaller than original
        assert result.shape[0] < image.shape[0]
        assert result.shape[1] < image.shape[1]

    def test_pad_and_crop(self):
        """Test both padding and cropping."""

        image = np.ones((10, 10))
        extent = (5, 15, 5, 15)  # x_min, x_max, z_min, z_max
        target_extent = (0, 20, 3, 17)  # Expand in x, mixed in z

        result = pad_or_crop_extent(image, extent, target_extent)

        # Should have specific dimensions based on the extent changes
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2

    def test_same_extent(self):
        """Test with same extent (no change expected)."""

        image = np.ones((10, 10))
        extent = (0, 10, 0, 10)
        target_extent = (0, 10, 0, 10)

        result = pad_or_crop_extent(image, extent, target_extent)

        # Should be same size
        assert result.shape == image.shape
        np.testing.assert_array_equal(result, image)
