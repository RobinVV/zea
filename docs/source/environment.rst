Here are the environment variables that ``zea`` uses at runtime.

Environment variables
================================

.. list-table::
   :header-rows: 1
   :widths: 20 80 20 20

   * - **Variable**
     - **Description**
     - **Default**
     - **Options**
   * - ``ZEA_CACHE_DIR``
     - Directory to use for caching downloaded files, e.g. model weights or datasets from Hugging Face Hub.
     - ``~/.cache/zea``
     - ``str``
   * - ``ZEA_LOG_LEVEL``
     - Logging level for Zea.
     - ``DEBUG``
     - ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``
   * - ``ZEA_DISABLE_CACHE``
     - If set to ``1`` will write to a temporary cache directory that is deleted after the program exits.
     - 0
     - ``0``, ``1``
   * - ``ZEA_NVIDIA_SMI_TIMEOUT``
     - Timeout in seconds for calling ``nvidia-smi`` to get GPU information.
     - ``30``
     - Any positive integer, or ``<= 0`` to disable timeout.
