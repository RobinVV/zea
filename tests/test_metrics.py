"""Module for testing loss functions"""

import inspect

import numpy as np
import pytest
from keras import ops

from zea import metrics
from zea.backend.tensorflow.losses import SMSLE
from zea.internal.registry import metrics_registry

from . import backend_equality_check


def test_smsle():
    """Test SMSLE loss function"""
    # Create random y_true and y_pred data
    y_true = np.random.rand(1, 11, 128, 512, 2).astype(np.float32)
    y_pred = np.random.rand(1, 11, 128, 512, 2).astype(np.float32)

    # Calculate SMSLE loss
    smsle = SMSLE()
    loss = smsle(y_true, y_pred)

    # Check if loss is a scalar
    assert loss.shape == ()


@pytest.mark.parametrize("metric_name", metrics_registry.registered_names())
@backend_equality_check(decimal=3)
def test_metrics(metric_name):
    """Test all losses and metrics"""
    if metric_name == "lpips":
        metric = metrics.get_metric(metric_name, image_range=[0, 255])
    else:
        metric = metrics.get_metric(metric_name)
    paired = metrics_registry.get_parameter(metric_name, "paired")

    rng = np.random.default_rng(42)
    y_true = rng.random((2, 16, 16, 3)).astype(np.float32) * 255.0
    y_pred = rng.random((2, 16, 16, 3)).astype(np.float32) * 255.0
    y_true = ops.convert_to_tensor(y_true)
    y_pred = ops.convert_to_tensor(y_pred)

    if paired:
        metric_value = metric(y_true, y_pred)
    else:
        metric_value = metric(y_pred)

    return metric_value


def test_metrics_class():
    """Test Metrics class"""
    rng = np.random.default_rng(42)
    y_true = rng.random((2, 16, 16, 3)).astype(np.float32) * 255.0
    y_pred = rng.random((2, 16, 16, 3)).astype(np.float32) * 255.0
    y_true = ops.convert_to_tensor(y_true)
    y_pred = ops.convert_to_tensor(y_pred)

    METRIC_NAMES = ["mse", "psnr", "ssim"]
    metrics_instance = metrics.Metrics(METRIC_NAMES, [0, 255])

    results = metrics_instance(y_true, y_pred, average_batch=True)
    assert all(name in results for name in METRIC_NAMES)
    assert all(np.isscalar(value.item()) for value in results.values())

    results_no_avg = metrics_instance(y_true, y_pred, average_batch=False, batch_axes=0)
    assert all(name in results_no_avg for name in METRIC_NAMES)
    assert all(value.shape[0] == 2 for value in results_no_avg.values())


def test_metrics_registry():
    """Test if all metrics are in the registry"""
    metrics_module = inspect.getmodule(metrics)
    metrics_funcs = inspect.getmembers(metrics_module, inspect.isfunction)
    metrics_func_names = [func[0] for func in metrics_funcs]

    for metric in metrics_func_names:
        if metric == "get_metric" or metric.startswith("_"):
            continue
        assert metric in metrics_registry, f"{metric} is not in the metrics registry"


def test_sector_reweight_image():
    """Test sector reweight util function"""
    # TODO: redo this test to not reimplement the function
    # arrange
    cube_of_ones = np.ones((3, 3, 3))

    # act
    reweighted_cube = metrics._sector_reweight_image(cube_of_ones, 180, axis=1)

    # assert
    # depths are set at the 'center' of each pixel index
    expected_depths = np.array([0.5, 1.5, 2.5])
    expected_reweighting_per_depth = np.pi  # (180 / 360) * 2 * pi = pi
    expected_result = cube_of_ones * expected_depths[:, None] * expected_reweighting_per_depth
    assert np.all(expected_result == reweighted_cube)
