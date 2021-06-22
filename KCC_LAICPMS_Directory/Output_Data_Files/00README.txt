LA-ICP-MS Input Software

Use: Creates excel and pdf documents containing compiled and edited data from information and raw data provided by the output of the LA-ICP-MS system.

Open to directory containing LAICPMS_Input.py and {}*_LAICPMS_Directory

Input into terminal: python LAICPMS_Input.py {}*_LAICPMS_Directory
*Core Abbreviation (ex. Colle Gnifetti's abbreviation is KCC->KCC_LAICPMS_Directory)

For User Input
Inside LAICPMS_Directory
	- Sections Folder
		contains section folders that hold raw data in text files exported from the LA-ICP-MS system

	- Input_Files Folder
		contains input excel files for each section with information about the section and core provided by the user 

Outputs: Found in Output_Data_Files

- {}_Core_LAICPMS_Data_02152018
	contains compiled input information, compiled section/run information, compiled data by depth for all elements, runs and sections LR & MR stand for different elements ran together

- {}_TuningParameters_02152018
	contains compiled tunning parameters from the input files


- Background Information
	contains excel and pdf documents containing statistical data for the first 12 sections for each run of each section lasered

- Section Plots
	contains PDF documents of plots for each run of each section and each section





