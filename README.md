# hmd_gait
repository for processing gait for evaluating HMD

# Upper-Body HMD Movement Modelling

This section summarises the scripts used for the upper-body movement analysis done during the associated Master's project. The pipeline processes upper-body IMU data collected during gait and functional tasks performed under three conditions: Normal (no HMD), AR (optical see-through Augmented Reality HMD), and VR (video see-through Virtual Reality HMD).

The goal of this workflow was to:
* extract time-series features from the upper-body IMU signals
* identify informative features for recognising movement events and determine a minimal upper-body IMU configuration
* construct event- and condition-specific normative PCA models
* quantify deviations in AR and VR conditions relative to normal movement.

The scripts below implement different stages of this analysis pipeline.

## 1. Motion Capture and IMU Pre-processing

These scripts prepare the raw motion capture and IMU datasets before feature extraction and modelling.

/scripts/preprocessing

`check_imu_data.py`
Performs quality checks on IMU recordings to identify missing or corrupted sensor data.

`clean_trc.py`
Cleans .trc motion capture files by correcting formatting issues and removing inconsistent marker trajectories.

`delete_marker_from_frame.py`
Removes incorrect marker detectioons from individual frames in motion capture datasets.

`export_trc_from_c3d.py`
Converts .c3d motion capture files into .trc format for downstream biomechanical processing.

`ik_trc.py`
Runs inverse kinematics processing using marker trajectories stored in .trc files.

`MissingHipMarkerReconstruction.py`
Reconstructs missing hip marker positions when markers are occluded during capture.

`pattern_fill.py`
Fills missing marker data segments using pattern-based interpolation (similar to Vicon Nexus gap-filling, but for the start and end of trials).

## 2. Signal Synchronisation

These scripts align signals recorded from different sensing systems (IMU and optical motion capture).

/scripts/synchronisation

`arm_sync.py`
Synchronises arm IMU signals (from T-pose arm raise at start and end of trials) with motion capture recordings.

`imu_upsample.py`
Upsamples IMU signals to match the sampling rate of motion capture data.

`knee_angle_sync.py`
Aligns IMU signals with knee joint angle measurements derived from motion capture.

`sync_utils.py`
Utility functions supporting synchronisation tasks.

## 3. Event Segmentation

These scripts prepare segmented data windows and compute time-series features used for machine learning models.

`trim_events.py`
Segments the dataset into labelled movement events used for feature extraction and model training.

`event_constants.py`
Stores constants and definitions for movement events analysed in the study.

## 4. Feature Engineering & Event Classification Models

These scripts extract/select features from IMU time-series, and train and evaluate machine learning (Random Forest) models that recognise events from IMU data.

`upper_body_classifier.py`
Main classifier training script for recognising movement events using extracted and selected upper-body IMU features.

`upper_body_classifier_minimal_imu.py`
Classifier training using the reduced minimal IMU configurations identified during sensor importance analysis.

`classifier_predict.py`
Applies trained classifiers to new data (different events) for event prediction.

## 5. IMU Importance

`imu_score.py`
Evaluates the contribution of individual IMU sensors to classification performance and helps determine a minimal IMU configuration.

## 6. Normative Movement Modelling

These scripts implement PCA-based normative modelling used to quantify movement deviations under HMD conditions.

`pca_upper_body.py`
Builds event- and condition-specific PCA models describing typical upper-body movement patterns.

`pca_fit_newdata.py`
Projects new datasets (e.g. AR and VR trials) into trained PCA spaces to compute reconstruction errors.

## 7. Additional Analysis Scripts

These scripts support supplementary analyses and experiments.

`t_test.py`
Performs statistical testing of classification performances between different IMU sets.

# Requirements
Python 3.11 or greater
ptb package - https://github.com/tedcty/ptb/tree/main/python_lib/ptb_src/dist

`requirements.py`
Lists Python dependencies required to run the pipelines.

`install_requirements.py`
Script to install required dependencies.
