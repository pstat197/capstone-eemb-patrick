# Project Title

Mantis Shrimp Behavior Analysis

## Description

This project will focus on using video data analysis to automate the detection of mantis shrimp behaviors. From a starting point of video recordings, the projects aims to identify time points in videos where two individuals come into close contact. These timestamps will be used to simplify later behavioral analysis. Gather tracking data on uniquely-marked individuals, including, e.g., how often and how far individuals move over the course of the experiment. Automate the data collection process for future use. E.g., from a starting point of video recordings, generate an output product containing: 
* timestamps of relevant interactions
* movement data from each individual, and 
* residence data from each individual. 

Automate the behavioral analysis of fights themselves, e.g., finding automated methods of counting the number of strikes each individual delivers. 

## Getting Started

### Prerequisites

* Python - Version 3.10.0 or higher.
* Conda - With Jupyter Notebook installed.
* OpenCV - Version 4.7.0. [Download and Install OpenCV](https://pypi.org/project/opencv-python/), or use the following code in your Conda command prompt.
```bash
$ pip install opencv-python
```
* Tracktor - Tracktor is an OpenCV based object tracking software. [See the instructions.](https://github.com/vivekhsridhar/tracktor). 
* Other packages - Numpy, Pandas, Scipy, Sys 

### Repository contents

* `data` contains

   - Short raw videos of mantis shrimp for code testing. 
   - Scripts that link to larger videos on other drives.
   
* `scripts` contains

   - `tracktor.py` tracktor code.
   - Video object tracking scripts.
   - etc.
   
* `writeups` contains 

   - contains the write-ups for finalized work, results, and guides to execute scripts.
   
### Executing program

* Describe how to run the program. 

## Contributors

* Tianhong Liu 
* Ashley Son
* Brian Fan
* Luke Fields

## Version History

* Version update description

## References

* Any reference you want to add.
