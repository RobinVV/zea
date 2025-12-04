"""Test dataset conversion scripts"""
import subprocess
import zipfile
import pytest
import numpy as np
from pathlib import Path
import csv
import imageio
    
from zea.data.convert.utils import load_avi, unzip
from zea.data.convert.images import convert_image_dataset

from .. import DEFAULT_TEST_SEED

@pytest.mark.parametrize("dataset",
                         ["echonet", "echonetlvh", "camus", "picmus", "verasonics"])
def test_conversion_script(tmp_path_factory, dataset):
    """
    Function that given a dataset name creates some temporary data which is 
    similar to the real dataset, runs the corresponding conversion script,
    and verifies the output.
    """
    src = tmp_path_factory.mktemp("src")
    dst = tmp_path_factory.mktemp("dst")
    
    create_test_data_for_dataset(dataset, src)
    
    subprocess.run(
        ["python", "-m", "zea.data.convert", dataset, str(src), str(dst)],
        check=True,
    )
    
    verify_converted_test_dataset(dataset, dst)


def create_test_data_for_dataset(dataset, src):
    """
    Selects the function that generates test data based on the provided dataset

    Args:
        dataset (str): string containing name of the dataset
        src (Path): path to the source directory where test data will be created

    Raises:
        ValueError: If the dataset name is unknown
    """
    if dataset == "echonet":
        create_echonet_test_data(src)
    elif dataset == "echonetlvh":
        create_echonetlvh_test_data(src)
    elif dataset == "camus":
        create_camus_test_data(src)
    elif dataset == "picmus":
        create_picmus_test_data(src)
    elif dataset == "verasonics":
        create_verasonics_test_data(src)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return

def verify_converted_test_dataset(dataset, dst):
    """
    Selects the function that reads the converted test dataset based on the provided dataset

    Args:
        dataset (str): string containing name of the dataset
        dst (Path): path to the destination directory where converted test data is located

    Raises:
        ValueError: If the dataset name is unknown
    """
    
    if dataset == "echonet":
        verify_converted_echonet_test_data(dst)
    elif dataset == "echonetlvh":
        verify_converted_echonetlvh_test_data(dst)
    elif dataset == "camus":
        verify_converted_camus_test_data(dst)
    elif dataset == "picmus":
        verify_converted_picmus_test_data(dst)
    elif dataset == "verasonics":
        verify_converted_verasonics_test_data(dst)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return

def create_echonet_test_data(src):
    return

def create_echonetlvh_test_data(src):
    return

def create_camus_test_data(src):
    return

def create_picmus_test_data(src):
    return

def create_verasonics_test_data(src):
    return

def verify_converted_echonet_test_data(dst):
    return

def verify_converted_echonetlvh_test_data(dst):
    return

def verify_converted_camus_test_data(dst):
    return

def verify_converted_picmus_test_data(dst):
    return

def verify_converted_verasonics_test_data(dst):
    return

def test_convert_image_dataset(tmp_path):
    return


def test_load_avi(tmp_path):
    """Test the load_avi function from zea.data.convert.utils"""
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    # Create a temporary AVI file with known content
    avi_path = tmp_path / "test_video.avi"
    frames = [rng.integers(0, 256, (32, 32), dtype=np.uint8) for _ in range(10)]
    with imageio.get_writer(avi_path, fps=10, codec="ffv1") as writer:
        for frame in frames:
            writer.append_data(frame)

    # Load the AVI file using the function
    loaded_frames = load_avi(avi_path, mode="L")

    # Verify the shape and content
    assert loaded_frames.shape == (10, 32, 32)
    for i in range(10):
        np.testing.assert_allclose(loaded_frames[i], frames[i], atol=1)


@pytest.mark.parametrize("dataset", [
    ("picmus", "picmus.zip", "archive_to_download"),
    ("camus", "CAMUS_public.zip", "CAMUS_public"),
    ("echonet", "EchoNet-Dynamic.zip", "echonet"),
    ("echonetlvh", "EchoNet-LVH.zip", "Batch1")]
)
def test_unzip(tmp_path, dataset):
    """Test the unzip function from zea.data.convert.utils for all dataset-name pairs"""
    dataset_name, zip_filename, folder_name = dataset
    # Create a dummy zip file
    zip_path = tmp_path / zip_filename
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # Add a dummy file to the zip
        zipf.writestr(f"{folder_name}/dummy.txt", "This is a test.")
        if dataset_name == "echonetlvh":
            # EchoNetLVH dataset unzips into four folders and a csv file.
            zipf.writestr("Batch2/dummy.txt", "This is a test.")
            zipf.writestr("Batch3/dummy.txt", "This is a test.")
            zipf.writestr("Batch4/dummy.txt", "This is a test.")
            
            with open(Path(f"{tmp_path}/MeasurementsList.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["frame", "mean_value"])
                for i in range(3):
                    writer.writerow([i, i * 2])  # example data
            zipf.write(f"{tmp_path}/MeasurementsList.csv", "MeasurementsList.csv")


    # Call the unzip function twice:
    # Once to initialize from zip, once to initialize from folder
    unzip(tmp_path, dataset_name)
    unzip(tmp_path, dataset_name)

    # Verify that the folder was created and contains the dummy file
    extracted_folder = tmp_path / folder_name
    assert extracted_folder.exists()
    assert (extracted_folder / "dummy.txt").exists()
