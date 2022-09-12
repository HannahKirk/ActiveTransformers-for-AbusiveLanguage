# ActiveTransformers-for-AbusiveLanguage

This repo contains the code for our paper "Is More Data Better? Re-thinking the Importance of Efficiency in Abusive
Language Detection with Transformers-Based Active Learning". 

The core scripts are in four groups:
* `0_data_prep` contains the scripts for loading, cleaning and preparing the data for modelling.
* `1_experiment_utils` contains the scripts defining key functions for the training and evaluation processes.
* `2_experiment_runners` contains the base python scripts and slurm scripts for running each family of experiments. A family consists of the base classification framework -- transformers (distil-roBERTa) vs traditional/sklearn (SVM) and the modelling approach -- supervised "passive" learning over the full dataset (SL) vs active learning in iterations (AL).
* `3_submit_jobs` contains the scripts to create a parameter grid of all the possible experimental runs and launch them on a server.

For more information, feel free to contact me: hannah.kirk@oii.ox.ac.uk. 

