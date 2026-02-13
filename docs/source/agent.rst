.. _agent:

Agent
=====

The ``zea`` toolbox provides agent functionalities for implementing **cognitive ultrasound** imaging via active perception. This module enables intelligent, adaptive transmit design, where acquisition decisions are informed by the current belief state about the imaged tissue.

For background on cognitive ultrasound imaging and perception-action loops, see the introductory paper by van Sloun :cite:t:`about-van2024active`.

-------------------------------
Overview
-------------------------------

Active perception in ultrasound involves iteratively:

1. **Perceiving** the current state of tissue from sparse acquisitions (e.g., using generative models)
2. **Selecting actions** (e.g., which scan lines to acquire next) based on the current belief
3. **Acquiring** new data and looping back to the perception step

The :mod:`zea.agent` module provides the building blocks for implementing such perception-action loops. The functions available currently focus on action selection strategies for focused transmit steering.

-------------------------------
Action Selection Strategies
-------------------------------

Action selection strategies determine which focused transmits to fire next, given some belief about the tissue state.

The following strategies are available in :mod:`zea.agent.selection`:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - **Strategy**
     - **Description**
   * - :class:`~zea.agent.selection.GreedyEntropy`
     - Selects lines that maximize entropy reduction. Uses a Gaussian reweighting scheme to approximate the decrease in uncertainty from observing each line.
   * - :class:`~zea.agent.selection.UniformRandomLines`
     - Randomly samples scan lines with uniform probability. Useful as a baseline.
   * - :class:`~zea.agent.selection.EquispacedLines`
     - Selects equispaced lines that sweep across the image. Deterministic and reproducible.
   * - :class:`~zea.agent.selection.CovarianceSamplingLines`
     - Models line-to-line correlation to select masks with highest entropy.
   * - :class:`~zea.agent.selection.TaskBasedLines`
     - Selects lines to maximize information gain with respect to a downstream task. Uses gradient-based saliency to identify regions contributing most to task uncertainty.

**Basic Usage:**

.. code-block:: python

    from zea.agent.selection import GreedyEntropy

    # Initialize the action selection strategy
    action_selector = GreedyEntropy(
        n_actions=8,              # Number of lines to select per step
        n_possible_actions=64,    # Total number of possible scan lines
        img_width=128,
        img_height=128,
    )

    # Sample actions given a set of belief particles
    # particles: tensor of shape (n_particles, height, width) representing beliefs
    selected_lines, masks = action_selector.sample(particles)

-------------------------------
Masks
-------------------------------

The :mod:`zea.agent.masks` module provides utilities for converting action representations (e.g., selected line indices) to image-sized masks that can be applied to observations.

.. code-block:: python

    from zea.agent import masks

    # Convert k-hot encoded lines to image masks
    image_masks = masks.lines_to_im_size(lines, (img_height, img_width))

-------------------------------
Example Notebooks
-------------------------------

We provide example notebooks demonstrating perception-action loops in practice, as companions to recently published papers on the topic:

**Patient-Adaptive Echocardiography**

This tutorial implements a basic perception-action loop using diffusion models for perception-as-inference and greedy entropy minimization for action selection.

- :doc:`notebooks/agent/agent_example`

  - Uses :class:`~zea.agent.selection.GreedyEntropy` to select informative scan lines
  - Demonstrates iterative belief refinement with sparse acquisitions
  - Visualizes the reconstruction process over multiple acquisition steps

**Task-Based Transmit Beamforming**

This tutorial implements a task-driven perception-action loop where acquisition decisions are optimized to gain information about a specific downstream task (e.g., left-ventricular dimension measurement).

- :doc:`notebooks/agent/task_based_perception_action_loop`

  - Uses :class:`~zea.agent.selection.TaskBasedLines` for task-aware line selection
  - Computes saliency maps via uncertainty propagation through downstream task models
  - Demonstrates how to integrate domain-specific measurement tasks

-------------------------------
API Reference
-------------------------------

For complete API documentation, see:

- :mod:`zea.agent.selection` — Action selection strategies
- :mod:`zea.agent.masks` — Mask utilities

-------------------------------
References
-------------------------------

.. bibliography:: ../../paper/paper.bib
   :style: unsrt
   :keyprefix: about-
   :labelprefix: A-

   van2024active