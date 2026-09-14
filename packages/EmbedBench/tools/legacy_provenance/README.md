# Legacy provenance tools

This directory contains verifiers needed only to authenticate historical
training evidence. These tools are not part of current IsingFold data
generation, model training, architecture selection, or evaluation.

`freeze_legacy_training_snapshot.py` reads immutable staged artifacts and
constructs a digest-bound record for the superseded 80-cell Quality V2 screen.
It must not be used to launch or select a current RL experiment.
