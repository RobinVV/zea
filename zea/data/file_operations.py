"""
This module provides some utilities to edit zea data files.

Available operations
--------------------

- `sum`: Sum multiple raw data files into one.

- `compound_frames`: Compound frames in a raw data file to increase SNR.

- `compound_transmits`: Compound transmits in a raw data file to increase SNR.

- `resave`: Resave a zea data file. This can be used to change the file format version.

- `extract`: extract frames and transmits in a raw data file.
"""

import argparse
from pathlib import Path

import numpy as np

from zea import Probe, Scan
from zea.data.data_format import generate_zea_dataset, load_additional_elements, load_description
from zea.data.file import File, load_file
from zea.internal.checks import _IMAGE_DATA_TYPES, _NON_IMAGE_DATA_TYPES
from zea.log import logger

ALL_DATA_TYPES_EXCEPT_RAW = set(_IMAGE_DATA_TYPES + _NON_IMAGE_DATA_TYPES) - {"raw_data"}

OPERATION_NAMES = [
    "sum",
    "compound_frames",
    "compound_transmits",
    "resave",
    "extract",
]


def save_file(
    path,
    raw_data: np.ndarray,
    scan: Scan,
    probe: Probe,
    additional_elements=None,
    description="",
    **kwargs,
):
    """Saves data to a zea data file (h5py file).

    Args:
        path (str, pathlike): The path to the hdf5 file.
        raw_data (np.ndarray): The data to save.
        scan (Scan): The scan object containing the parameters of the acquisition.
        probe (Probe): The probe object containing the parameters of the probe.
        additional_elements (list of DatasetElement, optional): Additional elements to save in the
            file. Defaults to None.
    """

    generate_zea_dataset(
        path=path,
        raw_data=raw_data,
        **kwargs,
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

    raw_data, scan, probe = load_file(input_paths[0])
    other_data = _load_all_data_types_except_raw(File(input_paths[0]))
    description = load_description(input_paths[0])
    additional_elements = load_additional_elements(input_paths[0])

    for file in input_paths[1:]:
        new_data, new_scan, new_probe = load_file(file)
        assert raw_data.shape == new_data.shape, (
            f"Data shapes do not match. Got {raw_data.shape} and {new_data.shape}."
        )
        raw_data += new_data
        assert scan == new_scan, "Scan parameters do not match."
        assert probe == new_probe, "Probe parameters do not match."

    if overwrite:
        _delete_file_if_exists(output_path)

    save_file(
        output_path,
        raw_data,
        scan,
        probe,
        additional_elements=additional_elements,
        description=description,
        **other_data,
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

    raw_data, scan, probe = load_file(input_path)
    other_data = _load_all_data_types_except_raw(File(input_path))
    additional_elements = load_additional_elements(input_path)
    description = load_description(input_path)

    # Assuming the first dimension is the frame dimension
    compounded_data = np.mean(raw_data, axis=0, keepdims=True)

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
        **other_data,
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

    raw_data, scan, probe = load_file(input_path)
    other_data = _load_all_data_types_except_raw(File(input_path))
    additional_elements = load_additional_elements(input_path)
    description = load_description(input_path)

    if not _all_tx_are_identical(scan):
        logger.warning(
            "Not all transmits are identical. Compounding transmits may lead to unexpected results."
        )

    # Assuming the second dimension is the transmit dimension
    compounded_data = np.mean(raw_data, axis=1, keepdims=True)

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
        **other_data,
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

    raw_data, scan, probe = load_file(input_path)
    other_data = _load_all_data_types_except_raw(File(input_path))
    additional_elements = load_additional_elements(input_path)
    description = load_description(input_path)
    scan.set_transmits("all")

    if overwrite:
        _delete_file_if_exists(output_path)
    save_file(
        output_path,
        raw_data,
        scan,
        probe,
        additional_elements=additional_elements,
        description=description,
        **other_data,
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
    indices = (frame_indices, transmit_indices)
    raw_data, scan, probe = load_file(input_path, indices=indices)
    other_data = _load_all_data_types_except_raw(File(input_path), indices=indices)

    additional_elements = load_additional_elements(input_path)
    description = load_description(input_path)

    scan = _scan_reduce_frames(scan, frame_indices)

    if overwrite:
        _delete_file_if_exists(output_path)

    save_file(
        output_path,
        raw_data,
        scan,
        probe,
        additional_elements=additional_elements,
        description=description,
        **other_data,
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


def _load_all_data_types_except_raw(file: File, indices=None):
    data = {}
    for data_type in ALL_DATA_TYPES_EXCEPT_RAW:
        try:
            data[data_type] = file.load_data(data_type, indices=indices)
        except AssertionError:
            pass
    return data


def get_parser():
    """Command line argument parser with subcommands"""

    parser = argparse.ArgumentParser(
        description="Manipulate zea data files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)
    _add_parser_sum(subparsers)
    _add_parser_compound_frames(subparsers)
    _add_parser_compound_transmits(subparsers)
    _add_parser_resave(subparsers)
    _add_parser_extract(subparsers)

    return parser


def _add_parser_sum(subparsers):
    sum_parser = subparsers.add_parser("sum", help="Sum the raw data of multiple files.")
    sum_parser.add_argument("input_paths", type=Path, nargs="+", help="Paths to the input files.")
    sum_parser.add_argument("output_path", type=Path, help="Output HDF5 file.")
    sum_parser.add_argument(
        "--overwrite", action="store_true", default=False, help="Overwrite existing output file."
    )


def _add_parser_compound_frames(subparsers):
    cf_parser = subparsers.add_parser("compound_frames", help="Compound frames to increase SNR.")
    cf_parser.add_argument("input_path", type=Path, help="Input HDF5 file.")
    cf_parser.add_argument("output_path", type=Path, help="Output HDF5 file.")
    cf_parser.add_argument(
        "--overwrite", action="store_true", default=False, help="Overwrite existing output file."
    )


def _add_parser_compound_transmits(subparsers):
    ct_parser = subparsers.add_parser(
        "compound_transmits", help="Compound transmits to increase SNR."
    )
    ct_parser.add_argument("input_path", type=Path, help="Input HDF5 file.")
    ct_parser.add_argument("output_path", type=Path, help="Output HDF5 file.")
    ct_parser.add_argument(
        "--overwrite", action="store_true", default=False, help="Overwrite existing output file."
    )


def _add_parser_resave(subparsers):
    resave_parser = subparsers.add_parser("resave", help="Resave a file to change format version.")
    resave_parser.add_argument("input_path", type=Path, help="Input HDF5 file.")
    resave_parser.add_argument("output_path", type=Path, help="Output HDF5 file.")
    resave_parser.add_argument(
        "--overwrite", action="store_true", default=False, help="Overwrite existing output file."
    )


def _add_parser_extract(subparsers):
    extract_parser = subparsers.add_parser("extract", help="Extract subset of frames or transmits.")
    extract_parser.add_argument("input_path", type=Path, help="Input HDF5 file.")
    extract_parser.add_argument("output_path", type=Path, help="Output HDF5 file.")
    extract_parser.add_argument(
        "--transmits",
        type=str,
        nargs="+",
        default="all",
        help="Target transmits. Can be a list of integers or ranges (e.g. 0-3 7).",
    )
    extract_parser.add_argument(
        "--frames",
        type=str,
        nargs="+",
        default="all",
        help="Target frames. Can be a list of integers or ranges (e.g. 0-3 7).",
    )
    extract_parser.add_argument(
        "--overwrite", action="store_true", default=False, help="Overwrite existing output file."
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.output_path.exists() and not args.overwrite:
        logger.error(
            f"Output file {args.output_path} already exists. Use --overwrite to overwrite it."
        )
        exit(1)

    if args.operation == "compound_frames":
        compound_frames(
            input_path=args.input_path, output_path=args.output_path, overwrite=args.overwrite
        )
    elif args.operation == "compound_transmits":
        compound_transmits(
            input_path=args.input_path, output_path=args.output_path, overwrite=args.overwrite
        )
    elif args.operation == "resave":
        resave(input_path=args.input_path, output_path=args.output_path, overwrite=args.overwrite)
    elif args.operation == "extract":
        extract_frames_transmits(
            input_path=args.input_path,
            output_path=args.output_path,
            frame_indices=_interpret_indices(args.frames),
            transmit_indices=_interpret_indices(args.transmits),
            overwrite=args.overwrite,
        )
    else:
        sum_raw_data(
            input_paths=args.input_paths, output_path=args.output_path, overwrite=args.overwrite
        )
