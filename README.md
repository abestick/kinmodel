Kinmodel ROS Package
====================
This package contains tools for creating, fitting, and tracking models of rigid kinematic tree structures, including human limbs and articulated objects like furniture, boxes, etc. It can:
- Fit the parameters of a rigid kinematic tree model to a specific person/object using motion capture data
- Use a UKF to track the state of a kinematic model in real time using motion capture data
- Stream the real time state estimate of a model on a ROS topic
- Save the identified kinematic tree models to JSON files for later editing and use

Key Pieces
----------
Particularly useful files in this repo include:
- `src/kinmodel/kinmodel.py`: Core library for defining, saving, fitting, and tracking kinematic models
- `src/fit_kinmodel.py`: Fit a kinematic model's joint parameters to a pre-recorded mocap sequence
- `src/track_kinmodel.py`: Track a fitted kinematic model online and output estimated joint angles to a ROS topic
- `src/track_kinmodel_offline.py`: Run kinematic model tracking on a pre-recorded mocap sequence and plot the resulting joint trajectory
- `src/mocap_generate_marker_assignments.py`: RViz GUI for generating a mapping between mocap marker indices and named features attached to a kinematic model
- `src/kinmodel/mocap_recorder.py`: Records mocap sequences for later fitting of kinematic models

This package depends on the [phasespace](https://github.com/abestick/phasespace) ROS package to interface with the mocap systems we have in the TeleImmersion Lab.
