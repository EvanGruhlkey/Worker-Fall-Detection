"""Stages 2, 6 and 7 of the fall detection pipeline.

- ``sampler``:   stage 2, drop frames to the classifier's frame rate
- ``features``:  joint positions, velocities and accelerations from a window
- ``model``:     stage 6, the GRU / TCN classifier and its checkpoint format
- ``gate``:      stage 7, threshold + debounce + cooldown
- ``detector``:  drop-in replacement for the rule-based ``FallDetector``
- ``synthetic``, ``dataset``, ``train``: building a training set and training
- ``visualize``: plots of the motion signals the classifier sees
"""
