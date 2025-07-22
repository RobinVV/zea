"""Test example notebooks in docs/source/notebooks.

Tests if notebooks run without errors using papermill. Generally these notebooks
are a bit heavy, so we mark the tests with the `notebook` marker, and also run
only on self-hosted runners. Run with:

.. code-block:: bash
    pytest -s -m 'notebook'

"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
from pathlib import Path

import papermill as pm
import pytest

# Automatically discover notebooks
NOTEBOOKS_DIR = Path("docs/source/notebooks")
NOTEBOOKS = list(NOTEBOOKS_DIR.rglob("*.ipynb"))

# Per-notebook parameters for CI testing (faster execution)
# these overwrite the default parameters in the notebooks
NOTEBOOK_PARAMETERS = {
    "diffusion_model_example.ipynb": {
        "n_unconditional_samples": 2,
        "n_unconditional_steps": 2,
        "n_conditional_samples": 2,
        "n_conditional_steps": 2,
    },
    "custom_models_example.ipynb": {
        "n_x": 10,
        "n_z": 10,
    },
    "agent_example.ipynb": {
        "n_prior_samples": 2,
        "n_unconditional_steps": 2,
        "n_initial_conditonal_steps": 1,
        "n_conditional_steps": 2,
        "n_conditional_samples": 2,
    },
    "zea_sequence_example.ipynb": {
        "n_frames": 15,
        "n_tx": 1,
        "n_tx_total": 3,
    },
    # Add more notebooks and their parameters here as needed
    # "other_notebook.ipynb": {
    #     "param1": value1,
    #     "param2": value2,
    # },
}


def pytest_sessionstart(session):
    print(f"📚 Preparing to test {len(NOTEBOOKS)} notebooks from {NOTEBOOKS_DIR}")
    print(f"📝 Using custom parameters for {len(NOTEBOOK_PARAMETERS)} notebooks")


@pytest.mark.notebook
@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda x: x.name)
def test_notebook_runs(notebook, tmp_path):
    print(f"\n📘 Starting notebook: {notebook.name}")

    output_path = tmp_path / notebook.name
    start = time.time()

    # Get custom parameters for this notebook if they exist
    notebook_params = NOTEBOOK_PARAMETERS.get(notebook.name, {})
    if notebook_params:
        print(f"🔧 Using custom parameters: {notebook_params}")

    pm.execute_notebook(
        input_path=str(notebook),
        output_path=str(output_path),
        kernel_name="python3",
        parameters=notebook_params,
    )

    duration = time.time() - start
    print(f"✅ Finished {notebook.name} in {duration:.1f}s")
