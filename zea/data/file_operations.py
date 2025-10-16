"""
This module provides some utilities to edit zea data files.

Available operations
--------------------

- `sum`: Sum multiple raw data files into one.

- `compound_frames`: Compound frames in a raw data file to increase SNR.

- `compound_transmits`: Compound transmits in a raw data file to increase SNR.

- `resave`: Resave a zea data file. This can be used to change the file format version.

- `extract`: extract frames and transmits in a raw data file.

Command-line usage
------------------

Sum two input files
^^^^^^^^^^^^^^^^^^^^^
.. code-block:: console

    python -m zea.data.file_operations sum input1.hdf5 input2.hdf5 output.hdf5

Compound frames/transmits
^^^^^^^^^^^^^^^
This can be used to increase the SNR of static acquisitions.

.. code-block:: console

    python -m zea.data.file_operations compound_frames input.hdf5 output.hdf5


.. code-block:: console

    python -m zea.data.file_operations compound_transmits input.hdf5 output.hdf5

    
Resave
^^^^^^
Loads a zea data file and saves it again. This can be used to change the file format version.

.. code-block:: console

    python -m zea.data.file_operations resave input.hdf5 output.hdf5

Extract frames and transmits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This can be used when you want to extract a subset of the data.

.. code-block:: console

    python -m zea.data.file_operations extract input.hdf5 output.hdf5 --frames 0-9 \
--transmits 0 2 4 6-8

"""

import argparse
from pathlib import Path

import numpy as np

from zea import Probe, Scan
from zea.data.data_format import generate_zea_dataset, load_additional_elements, load_description
from zea.data.file import load_file
from zea.log import logger

OPERATION_NAMES = [
    "sum",
    "compound_frames",
    "compound_transmits",
    "resave",
    "extract",
]


def save_file(
    path,
    data: np.ndarray,
    scan: Scan,
    probe: Probe,
    data_type="raw_data",
    additional_elements=None,
    description="",
):
    """Saves data to a zea data file (h5py file).

    Args:
        path (str, pathlike): The path to the hdf5 file.
        data (np.ndarray): The data to save.
        scan (Scan): The scan object containing the parameters of the acquisition.
        probe (Probe): The probe object containing the parameters of the probe.
        data_type (str, optional): The type of data to save. Defaults to
            'raw_data'. Other options are 'aligned_data', 'beamformed_data',
            'envelope_data', 'image' and 'image_sc'.
        additional_elements (list of DatasetElement, optional): Additional elements to save in the
            file. Defaults to None.
    """

    data_args = {data_type: data}

    generate_zea_dataset(
        path=path,
        **data_args,
        probe_name="generic",
        probe_geometry=probe.probe_geometry,
        sampling_frequency=scan.sampling_frequency,
        center_frequency=scan.center_frequency,
        initial_times=scan.initial_times,
        t0_delays=scan.t0_delays,
        sound_speed=scan.sound_speed,
        focus_distances=scan.focus_distances,
        polar_angles=scan.polar_angles,
        azimuth_angles=scan.azimuth_angles,
        tx_apodizations=scan.tx_apodizations,
        bandwidth_percent=scan.bandwidth_percent,
        time_to_next_transmit=scan.time_to_next_transmit,
        tgc_gain_curve=scan.tgc_gain_curve,
        element_width=scan.element_width,
        tx_waveform_indices=scan.tx_waveform_indices,
        waveforms_one_way=scan.waveforms_one_way,
        waveforms_two_way=scan.waveforms_two_way,
        description=description,
        additional_elements=additional_elements,
    )


def sum_raw_data(input_paths: list[Path], output_path: Path, overwrite=False):
    """
    Sums multiple raw data files and saves the result to a new file.

    Args:
        input_paths (list[Path]): List of paths to the input raw data files.
        output_path (Path): Path to the output file where the summed data will be saved.
        overwrite (bool, optional): Whether to overwrite the output file if it exists. Defaults to
            False.
    """

    data, scan, probe = load_file(input_paths[0])
    description = load_description(input_paths[0])
    additional_elements = load_additional_elements(input_paths[0])

    for file in input_paths[1:]:
        new_data, new_scan, new_probe = load_file(file)
        assert data.shape == new_data.shape, (
            f"Data shapes do not match. Got {data.shape} and {new_data.shape}."
        )
        data += new_data
        assert scan == new_scan, "Scan parameters do not match."
        assert probe == new_probe, "Probe parameters do not match."

    if overwrite:
        _delete_file_if_exists(output_path)

    save_file(
        output_path,
        data,
        scan,
        probe,
        additional_elements=additional_elements,
        description=description,
    )


def compound_frames(input_path: Path, output_path: Path, overwrite=False):
    """
    Compounds frames in a raw data file by averaging them.

    Args:
        input_path (Path): Path to the input raw data file.
        output_path (Path): Path to the output file where the compounded data will be saved.
        overwrite (bool, optional): Whether to overwrite the output file if it exists. Defaults to
            False.
    """

    data, scan, probe = load_file(input_path)
    additional_elements = load_additional_elements(input_path)
    description = load_description(input_path)

    # Assuming the first dimension is the frame dimension
    compounded_data = np.mean(data, axis=0, keepdims=True)

    scan = _scan_reduce_frames(scan, [0])

    if overwrite:
        _delete_file_if_exists(output_path)

    save_file(
        output_path,
        compounded_data,
        scan,
        probe,
        additional_elements=additional_elements,
        description=description,
    )


def compound_transmits(input_path: Path, output_path: Path, overwrite=False):
    """
    Compounds transmits in a raw data file by averaging them.

    Note
    ----
    This function assumes that all transmits are identical. If this is not the case the function
    will result in incorrect scan parameters.

    Args:
        input_path (Path): Path to the input raw data file.
        output_path (Path): Path to the output file where the compounded data will be saved.
        overwrite (bool, optional): Whether to overwrite the output file if it exists. Defaults to
            False.
    """

    data, scan, probe = load_file(input_path)
    additional_elements = load_additional_elements(input_path)
    description = load_description(input_path)

    if not _all_tx_are_identical(scan):
        logger.warning(
            "Not all transmits are identical. Compounding transmits may lead to unexpected results."
        )

    # Assuming the second dimension is the transmit dimension
    compounded_data = np.mean(data, axis=1, keepdims=True)

    scan.set_transmits([0])

    if overwrite:
        _delete_file_if_exists(output_path)

    save_file(
        output_path,
        compounded_data,
        scan,
        probe,
        additional_elements=additional_elements,
        description=description,
    )


def _all_tx_are_identical(scan: Scan):
    """Checks if all transmits in a Scan object are identical."""
    attributes_to_check = [
        scan.polar_angles,
        scan.azimuth_angles,
        scan.t0_delays,
        scan.tx_apodizations,
        scan.focus_distances,
        scan.initial_times,
    ]

    for attr in attributes_to_check:
        if attr is not None and not _check_all_identical(attr, axis=0):
            return False
    return True


def _check_all_identical(array, axis=0):
    """Checks if all elements along a given axis are identical."""
    first = array.take(0, axis=axis)
    return np.all(np.equal(array, first), axis=axis).all()


def resave(input_path: Path, output_path: Path, overwrite=False):
    """
    Resaves a zea data file to a new location.

    Args:
        input_path (Path): Path to the input zea data file.
        output_path (Path): Path to the output file where the data will be saved.
        overwrite (bool, optional): Whether to overwrite the output file if it exists. Defaults to
            False.
    """

    data, scan, probe = load_file(input_path)
    additional_elements = load_additional_elements(input_path)
    description = load_description(input_path)
    scan.set_transmits("all")

    if overwrite:
        _delete_file_if_exists(output_path)
    save_file(
        output_path,
        data,
        scan,
        probe,
        additional_elements=additional_elements,
        description=description,
    )


def extract_frames_transmits(
    input_path: Path, output_path: Path, frame_indices, transmit_indices, overwrite=False
):
    """
    extracts frames and transmits in a raw data file.

    Args:
        input_path (Path): Path to the input raw data file.
        output_path (Path): Path to the output file where the extracted data will be saved.
        frame_indices (list or array-like): Indices of the frames to keep.
        transmit_indices (list or array-like): Indices of the transmits to keep.
        overwrite (bool, optional): Whether to overwrite the output file if it exists. Defaults to
            False.
    """

    data, scan, probe = load_file(input_path, indices=(frame_indices, transmit_indices))
    additional_elements = load_additional_elements(input_path)
    description = load_description(input_path)

    scan = _scan_reduce_frames(scan, frame_indices)

    if overwrite:
        _delete_file_if_exists(output_path)

    save_file(
        output_path,
        data,
        scan,
        probe,
        additional_elements=additional_elements,
        description=description,
    )


def _delete_file_if_exists(path: Path):
    """Deletes a file if it exists."""
    if path.exists():
        path.unlink()


def _interpret_index(input_str):
    if "-" in input_str:
        start, end = map(int, input_str.split("-"))
        return list(range(start, end + 1))
    else:
        return [int(input_str)]


def _interpret_indices(input_str_list):
    if isinstance(input_str_list, str) and input_str_list == "all":
        return slice(None)

    if len(input_str_list) == 1 and "-" in input_str_list[0]:
        start, end = map(int, input_str_list[0].split("-"))
        return slice(start, end + 1)

    indices = []
    for part in input_str_list:
        indices.extend(_interpret_index(part))
    return indices


def _scan_reduce_frames(scan, frame_indices):
    transmit_indices = scan.selected_transmits
    scan.set_transmits("all")
    if scan.time_to_next_transmit is not None:
        scan.time_to_next_transmit = scan.time_to_next_transmit[frame_indices]
    scan.set_transmits(transmit_indices)
    return scan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sum multiple raw data files into one.")
    parser.add_argument(
        "operation",
        type=str,
        choices=OPERATION_NAMES,
        help="The operation to perform on the input files.",
    )
    parser.add_argument(
        "input_paths", type=Path, nargs="+", help="Paths to the input raw data files."
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to the output file where the summed data will be saved.",
    )

    parser.add_argument(
        "--transmits",
        type=str,
        nargs="*",
        default="all",
        help=("Target transmits. Can be a list of integers or ranges (e.g. 0-3)."),
    )

    parser.add_argument(
        "--frames",
        type=str,
        nargs="*",
        default="all",
        help=("Target frames. Can be a list of integers or ranges (e.g. 0-3)."),
    )

    parser.add_argument(
        "--overwrite", action="store_true", default=False, help="Overwrite existing output file."
    )

    args = parser.parse_args()

    if args.output_path.exists() and not args.overwrite:
        logger.error(
            f"Output file {args.output_path} already exists. Use --overwrite to overwrite it."
        )
        exit(1)

    if args.operation == "compound_frames":
        compound_frames(
            input_path=args.input_paths[0], output_path=args.output_path, overwrite=args.overwrite
        )
    elif args.operation == "compound_transmits":
        compound_transmits(
            input_path=args.input_paths[0], output_path=args.output_path, overwrite=args.overwrite
        )
    elif args.operation == "resave":
        resave(
            input_path=args.input_paths[0], output_path=args.output_path, overwrite=args.overwrite
        )
    elif args.operation == "extract":
        extract_frames_transmits(
            input_path=args.input_paths[0],
            output_path=args.output_path,
            frame_indices=_interpret_indices(args.frames),
            transmit_indices=_interpret_indices(args.transmits),
            overwrite=args.overwrite,
        )
    else:
        sum_raw_data(
            input_paths=args.input_paths, output_path=args.output_path, overwrite=args.overwrite
        )
