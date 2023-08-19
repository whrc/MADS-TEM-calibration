This directory contains scripts and notebooks used to investigate the 'deep_soil' branch of dvm-dos-tem

	deep_soil_uitls.py - contains plotting functions for analysis
	deep_soil_comparison.ipynb contains code used for plotting/visualizing results, this uses deep_soil_utils.py
	dsl_depth_exploration - script used to explore differences in results with the dsl module turned on/off
	experiment.sh - bash script containing commands to set up and run dvm-dos-tem runs in both the master and deep soil branch. 		NOTE do not run this bash script directly, use line by line. file names and folders need to be adjusted.
	Layer_var_synth.py - called by Output_synthesis_Valeria.sh
	Output_synthesis_Valeria.sh - use this to generate depth interpolated output data, will save in a new folder in the docker 			data folder. Make sure to edit the stage and depths needed

	Obs (folder) - contains some data/observations for MD3
