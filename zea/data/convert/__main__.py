import argparse
import os

import keras.backend as K

from zea import log


def get_parser():
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
        default=None,
        help="Additional destination folder path if also saving to numpy",
    )
    parser.add_argument(
        "--split_path",
        type=str,
        help="Path to the split.yaml file containing the dataset split if a split should be copied",
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
        "--no_rejection",
        action="store_true",
        help="EchonetLVH: Do not reject sequences in manual_rejections.txt",
    )

    parser.add_argument(
        "--batch",
        type=str,
        default=None,
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
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.dataset == "echonet":
        from .echonet import convert_echonet

        convert_echonet(args)
    elif args.dataset == "echonetlvh":
        from .echonetlvh.convert_raw_to_usbmd import convert_echonetlvh

        convert_echonetlvh(args)
    elif args.dataset == "camus":
        from .camus import convert_camus

        convert_camus(args)
    elif args.dataset == "picmus":
        from .picmus import convert_picmus

        convert_picmus(args)
    elif args.dataset == "matlab":
        from .matlab import convert_matlab

        keras_backend = K.backend()
        cpu = os.environ.get("CUDA_VISIBLE_DEVICES") in ["", "-1"]
        if keras_backend == "jax" and cpu:
            log.error(
                "For MATLAB conversion with JAX, GPU is required. Consider a different backend."
            )
        convert_matlab(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


if __name__ == "__main__":
    main()
