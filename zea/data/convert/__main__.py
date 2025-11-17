import argparse
import keras

from zea import init_device
from .camus import convert_camus
from .matlab import convert_matlab
from .picmus import convert_picmus
from .echonet import convert_echonet
from .echonetlvh.convert_raw_to_usbmd import convert_echonetlvh


def get_args():
    parser = argparse.ArgumentParser(description="Convert raw data to a USBMD dataset.")
    parser.add_argument(
        "dataset",
        choices=["echonet", "echonetlvh", "camus", "picmus", "matlab"],
        help="Raw dataset to convert",
    )
    parser.add_argument("src", type=str, help="Source folder path")
    parser.add_argument("dst", type=str, help="Destination folder path")
    parser.add_argument(
        "--dst_npz",
        type=str,
        default="None",
        help="Additional destination folder path if also saving to numpy",
    )
    parser.add_argument("--key", help="Key to access in the hdf5 files if necessary")
    parser.add_argument(
        "--split_path",
        type=str,
        help="Path to the yaml file containing the dataset split if a split should be copied",
    )
    parser.add_argument(
        "--no_hyperthreading",
        action="store_true",
        help="Disable hyperthreading for multiprocessing",
    )
    # Dataset specific arguments:

    # MATLAB
    parser.add_argument(
        "--frames",
        default=["all"],
        type=str,
        nargs="+",
        help="MATLAB: The frames to add to the file. This can be a list of integers, a range "
        "of integers (e.g. 4-8), or 'all'.",
    )
    # ECHONET_LVH
    parser.add_argument(
        "--batch",
        type=str,
        help="EchonetLVH: Specify which BatchX directory to process, e.g. --batch=Batch2",
    )
    parser.add_argument(
        "--convert_measurements",
        action="store_true",
        help="EchonetLVH: Only convert measurements CSV file",
    )
    parser.add_argument(
        "--convert_images", action="store_true", help="EchonetLVH: Only convert image files"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="EchonetLVH: Maximum number of files to process (for testing)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="EchonetLVH: Force recomputation even if parameters already exist",
    )
    args = parser.parse_args()
    return args


def main():
    init_device()
    args = get_args()
    if args.dataset == "echonet":
        convert_echonet(args)
    elif args.dataset == "echonetlvh":
        assert keras.backend.backend() == "jax", "EchonetLVH conversion requires the JAX backend."
        convert_echonetlvh(args)
    elif args.dataset == "camus":
        convert_camus(args)
    elif args.dataset == "picmus":
        convert_picmus(args)
    elif args.dataset == "matlab":
        convert_matlab(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


if __name__ == "__main__":
    main()
