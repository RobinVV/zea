"""Test the file operations module."""

import os
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from zea.data.data_format import (
    generate_example_dataset,
    load_additional_elements,
    load_description,
)
from zea.data.file import File, load_file, validate_file
from zea.data.file_operations import (
    compound_frames,
    compound_transmits,
    extract_frames_transmits,
    resave,
    sum_raw_data,
)

n_frames = 3
n_tx = 4
n_el = 16
n_ax = 128
n_ch = 1

DATASET_PARAMETERS = {
    "raw_data": np.ones((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32),
    "probe_geometry": np.zeros((n_el, 3), dtype=np.float32),
    "sampling_frequency": 30e6,
    "center_frequency": 6e6,
    "initial_times": np.zeros((n_tx), dtype=np.float32),
    "t0_delays": np.zeros((n_tx, n_el), dtype=np.float32),
    "sound_speed": 1540.0,
    "probe_name": "generic",
    "description": "Dataset parameters for testing",
    "focus_distances": np.zeros((n_tx,), dtype=np.float32),
    "polar_angles": np.linspace(-np.pi / 2, np.pi / 2, n_tx, dtype=np.float32),
    "azimuth_angles": np.zeros((n_tx), np.float32),
    "tx_apodizations": np.ones((n_tx, n_el), dtype=np.float32),
    "time_to_next_transmit": np.ones((n_frames, n_tx), dtype=np.float32),
    "bandwidth_percent": 200.0,
    "waveforms_one_way": [np.zeros((512), dtype=np.float32)],
    "waveforms_two_way": [np.zeros((512,), dtype=np.float32)],
    "tx_waveform_indices": np.zeros((n_tx,), dtype=np.int32),
}


@pytest.fixture
def tmp_hdf5_path(tmp_path) -> Generator[Path, None, None]:
    """Fixture to create a temporary HDF5 file."""
    yield Path(tmp_path, "test_case_dataset.hdf5")


def test_file_operations_sum(tmp_hdf5_path):
    """Tests the sum_raw_data function by creating two example datasets,
    summing them and checking if the result is correct."""

    # Create two example datasets
    input_path1 = tmp_hdf5_path.parent / "test_case_dataset1.hdf5"
    input_path2 = tmp_hdf5_path.parent / "test_case_dataset2.hdf5"
    generate_example_dataset(input_path1)
    generate_example_dataset(input_path2)

    data1, scan1, probe1 = load_file(input_path1)
    data2, scan2, probe2 = load_file(input_path2)

    # Sum the datasets
    output_path = tmp_hdf5_path.parent / "summed_dataset.hdf5"

    sum_raw_data([input_path1, input_path2], output_path)

    _assert_descriptions_and_additional_elements_equal(input_path1, output_path)

    # Load the summed dataset and check if the data is correct
    with File(output_path) as f:
        raw_data = f["data/raw_data"][:]
        assert raw_data[0, 0, 0, 0, 0] == data1[0, 0, 0, 0, 0] + data2[0, 0, 0, 0, 0]


def test_file_operations_extract(tmp_hdf5_path):
    """Tests the load_data function by creating an example dataset and
    loading a subset of the data."""

    input_path = tmp_hdf5_path.parent / "test_case_dataset.hdf5"
    output_path = tmp_hdf5_path.parent / "extracted_dataset.hdf5"

    # Create an example dataset
    generate_example_dataset(input_path)

    extract_frames_transmits(input_path, output_path, frame_indices=slice(2), transmit_indices=[0])
    data, scan, probe = load_file(output_path)

    _assert_descriptions_and_additional_elements_equal(input_path, output_path)

    assert data.shape[0] == 2
    assert data.shape[1] == 1


def test_file_operations_resave(tmp_hdf5_path):
    """Tests the resave operation by creating an example dataset and
    resaving it to a new file."""

    input_path = tmp_hdf5_path.parent / "test_case_dataset.hdf5"
    output_path = tmp_hdf5_path.parent / "resaved_dataset.hdf5"

    # Create an example dataset
    generate_example_dataset(input_path)

    resave(input_path, output_path)

    _assert_descriptions_and_additional_elements_equal(input_path, output_path)

    # Validate the resaved dataset
    validate_file(output_path)


def test_file_operations_compound_frames(tmp_hdf5_path):
    """Tests the compound_frames function by creating an example dataset and
    compounding frames."""

    input_path = tmp_hdf5_path.parent / "test_case_dataset.hdf5"
    output_path = tmp_hdf5_path.parent / "compounded_frames_dataset.hdf5"

    # Create an example dataset
    generate_example_dataset(input_path)

    compound_frames(input_path, output_path)

    _assert_descriptions_and_additional_elements_equal(input_path, output_path)

    data, scan, probe = load_file(output_path)
    assert data.shape[0] == 1  # Only one frame should remain


def test_file_operations_compound_transmits(tmp_hdf5_path):
    """Tests the compound_transmits function by creating an example dataset and
    compounding transmits."""

    input_path = tmp_hdf5_path.parent / "test_case_dataset.hdf5"
    output_path = tmp_hdf5_path.parent / "compounded_transmits_dataset.hdf5"

    # Create an example dataset
    generate_example_dataset(input_path)

    compound_transmits(input_path, output_path)

    _assert_descriptions_and_additional_elements_equal(input_path, output_path)

    data, scan, probe = load_file(output_path)
    assert data.shape[1] == 1  # Only one transmit should remain
    assert scan["initial_times"].shape[0] == 1
    assert scan["t0_delays"].shape[0] == 1
    assert scan["azimuth_angles"].shape[0] == 1
    assert scan["tx_apodizations"].shape[0] == 1


def test_file_operations_cli_sum(tmp_hdf5_path):
    """Tests the sum_raw_data function CLI by creating two example datasets,
    summing them and checking if the result is correct."""

    # Create two example datasets
    path1 = tmp_hdf5_path.parent / "test_case_dataset1.hdf5"
    path2 = tmp_hdf5_path.parent / "test_case_dataset2.hdf5"
    generate_example_dataset(path1)
    generate_example_dataset(path2)

    data1, scan1, probe1 = load_file(path1)
    data2, scan2, probe2 = load_file(path2)

    # Sum the datasets
    output_path = tmp_hdf5_path.parent / "summed_dataset.hdf5"

    os.system(
        "python -m zea.data.file_operations sum "
        + str(path1)
        + " "
        + str(path2)
        + " "
        + str(output_path)
    )

    # Load the summed dataset and check if the data is correct
    with File(output_path) as f:
        raw_data = f["data/raw_data"][:]
        assert raw_data[0, 0, 0, 0, 0] == data1[0, 0, 0, 0, 0] + data2[0, 0, 0, 0, 0]


def test_file_operations_cli_extract(tmp_hdf5_path):
    """Tests the load_data function CLI by creating an example dataset and
    loading a subset of the data."""

    input_path = tmp_hdf5_path.parent / "test_case_dataset.hdf5"
    output_path = tmp_hdf5_path.parent / "extracted_dataset.hdf5"

    # Create an example dataset
    generate_example_dataset(input_path)

    os.system(
        "python -m zea.data.file_operations extract "
        + str(input_path)
        + " "
        + str(output_path)
        + " --frames 0-1 --transmits 0 3 4"
    )

    data, scan, probe = load_file(output_path)
    assert data.shape[0] == 2
    assert data.shape[1] == 3


def _load_description_and_additional_elements(path: Path):
    description = load_description(path)
    additional_elements = load_additional_elements(path)
    return description, additional_elements


def _assert_descriptions_and_additional_elements_equal(path, other_path: Path):
    description, additional_elements = _load_description_and_additional_elements(path)
    other_description, other_additional_elements = _load_description_and_additional_elements(
        other_path
    )
    assert description == other_description
    assert len(additional_elements) == len(other_additional_elements)
