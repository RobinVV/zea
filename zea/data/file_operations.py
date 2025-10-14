"""
This module provides some utilities to edit zea data files.

Available operations
--------------------

- `sum`: Sum multiple raw data files into one.

- `compound_frames`: Compound frames in a raw data file to increase SNR.

- `compound_transmits`: Compound transmits in a raw data file to increase SNR.

- `resave`: Resave a zea data file. This can be useful to change the file format version.

- `subselect`: Subselect frames and transmits in a raw data file.

Command-line usage
------------------

.. code-block:: console

    # Sum two input files
    python -m zea.data.file_operations sum input1.hdf5 input2.hdf5 output.hdf5

    # Compound frames in an input file to increase SNR
    python -m zea.data.file_operations compound_frames input.hdf5 output.hdf5

    # Compound transmits in an input file to increase SNR
    python -m zea.data.file_operations compound_transmits input.hdf5 output.hdf5

    # Resave a zea data file
    python -m zea.data.file_operations resave input.hdf5 output.hdf5

    # Subselect frames and transmits in an input file
    python -m zea.data.file_operations subselect input.hdf5 output.hdf5 --frames 0-9 --transmits 0 2 4 6-8

"""

import argparse
from pathlib import Path

import numpy as np

from zea import Probe, Scan
from zea.data.data_format import generate_zea_dataset
from zea.data.file import load_file

OPERATION_NAMES = [
    "sum",
    "compound_frames",
    "compound_transmits",
    "resave",
    "subselect",
]


def save_file(path, data: np.ndarray, scan: Scan, probe: Probe, data_type="raw_data"):
    """Saves data to a zea data file (h5py file).

    Args:
        path (str, pathlike): The path to the hdf5 file.
        data (np.ndarray): The data to save.
        scan (Scan): The scan object containing the parameters of the acquisition.
        probe (Probe): The probe object containing the parameters of the probe.
        data_type (str, optional): The type of data to save. Defaults to
            'raw_data'. Other options are 'aligned_data', 'beamformed_data',
            'envelope_data', 'image' and 'image_sc'.
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
    )


def sum_raw_data(input_paths: list[Path], output_path: Path):
    """
    Sums multiple raw data files and saves the result to a new file.

    Args:
        input_paths (list[Path]): List of paths to the input raw data files.
        output_path (Path): Path to the output file where the summed data will be saved.
    """

    data, scan, probe = load_file(input_paths[0])

    for file in input_paths[1:]:
        new_data, new_scan, new_probe = load_file(file)
        data += new_data
        assert scan == new_scan, "Scan parameters do not match."
        assert probe == new_probe, "Probe parameters do not match."

    save_file(output_path, data, scan, probe)


def compound_frames(input_path: Path, output_path: Path):
    """
    Compounds frames in a raw data file by averaging them.

    Args:
        input_path (Path): Path to the input raw data file.
        output_path (Path): Path to the output file where the compounded data will be saved.
    """

    data, scan, probe = load_file(input_path)

    # Assuming the first dimension is the frame dimension
    compounded_data = np.mean(data, axis=0, keepdims=True)

    scan = _scan_reduce_frames(scan, [0])

    save_file(output_path, compounded_data, scan, probe)


def compound_transmits(input_path: Path, output_path: Path):
    """
    Compounds transmits in a raw data file by averaging them.

    Args:
        input_path (Path): Path to the input raw data file.
        output_path (Path): Path to the output file where the compounded data will be saved.
    """

    data, scan, probe = load_file(input_path)

    # Assuming the second dimension is the transmit dimension
    compounded_data = np.mean(data, axis=1, keepdims=True)

    # Update scan parameters
    scan.n_tx = 1
    scan.polar_angles = np.array([np.mean(scan.polar_angles)])
    scan.azimuth_angles = np.array([np.mean(scan.azimuth_angles)])
    scan.t0_delays = np.mean(scan.t0_delays, axis=0, keepdims=True)
    scan.tx_apodizations = np.mean(scan.tx_apodizations, axis=0, keepdims=True)
    scan.focus_distances = np.array([np.mean(scan.focus_distances)])
    scan.initial_times = np.array([np.mean(scan.initial_times)])

    save_file(output_path, compounded_data, scan, probe)


def resave(input_path: Path, output_path: Path):
    """
    Resaves a zea data file to a new location.

    Args:
        input_path (Path): Path to the input zea data file.
        output_path (Path): Path to the output file where the data will be saved.
    """

    data, scan, probe = load_file(
        input_path, scan_kwargs={"time_to_next_transmit": np.ones((800, 5))}
    )
    scan.set_transmits("all")
    save_file(output_path, data, scan, probe)


def subselect_frames_transmits(
    input_path: Path, output_path: Path, frame_indices, transmit_indices
):
    """
    Subselects frames and transmits in a raw data file.

    Args:
        input_path (Path): Path to the input raw data file.
        output_path (Path): Path to the output file where the subselected data will be saved.
        frame_indices (list or array-like): Indices of the frames to keep.
        transmit_indices (list or array-like): Indices of the transmits to keep.
    """

    data, scan, probe = load_file(input_path, indices=(frame_indices, transmit_indices))

    scan = _scan_reduce_frames(scan, frame_indices)

    save_file(output_path, data, scan, probe)


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

    if args.output_path.exists():
        if args.overwrite:
            args.output_path.unlink()
        else:
            raise FileExistsError(f"Output file {args.output_path} already exists.")

    if args.operation == "compound_frames":
        compound_frames(args.input_paths[0], args.output_path)
    elif args.operation == "compound_transmits":
        compound_transmits(args.input_paths[0], args.output_path)
    elif args.operation == "resave":
        resave(args.input_paths[0], args.output_path)
    elif args.operation == "subselect":
        subselect_frames_transmits(
            args.input_paths[0],
            args.output_path,
            _interpret_indices(args.frames),
            _interpret_indices(args.transmits),
        )
    else:
        sum_raw_data(args.input_paths, args.output_path)
