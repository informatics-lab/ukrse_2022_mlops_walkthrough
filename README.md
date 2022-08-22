# MLOps for RSEs
This repository contains material first presented at the 2022 conference of the Society of Research Software Engineering as a walkthrough on *MLOps for RSEs*, covering the key aspect of MLOps that RSEs will be expected to support and engage with broadly. The material is primarily a series of Jupyter notebooks.

## Running the material

To run the notebooks, you will need to go through the following steps (see subsections below for more details on each step):
* Clone the repository
* Set up a conda environments using the supplied conda requirements YAML files.
* Get the sample data
* Run Jupyter Lab

### Clone the repository

From the command line, run the following command:

```
git clone https://github.com/informatics-lab/ukrse_2022_mlops_walkthrough.git
```
then navigate to the root irectory of your local copy of the repository 

### Set up a conda environment

First [install anaconda or miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) if you do not already have them installed. Folloow the instruction on the conda website to do this.

Once you have installed conda (via anaconda or miniconda), you then need to set up the relvant conda environments on your machine. For this walktrhough we have several different conda environments, representing good practice to create seperate environments for different stages and tasks in a typical machine learning learning project or pipeline.

You can create the environment it describes with the following command:
```conda env create --file requirements.yml```

The following environments should be installed for this walkthrough:
* Data Preparation and Exploration - `requirements_data_prep.yml`
* Model Development and Training - `requiremments_model_development.yml`
* Model Inference, Evaluation and Explainability (XAI) - `requirements_model_evaluation.yml`

### Get the data

The data used in this walkthrough as an [archive on Zenodo](https://doi.org/10.5281/zenodo.6966936). Unformatuantely you can't easily just download all the files in a zendodo record, so you will need to download each. This can be done with the wget commands listed below. It is recommended that these be placed in a directory in ~/data/ukrse2022. 
* Raw Rotors dataset `wget https://zenodo.org/record/6966937/files/2021_met_office_aviation_rotors.csv ~/data/ukrse2022/`
* Preprocessed rotors dataset `wget XXX ~/data/ukrse2022/`
* Intake catalog file `wget catalog.yml ~/data/ukrse2022/`
* Preprocessed UK cutout of ERA5 Mean Sea-level Pressure dataset for 2017-2021 `wget https://zenodo.org/record/6966937/files/2021_met_office_aviation_rotors.csv ~/data/ukrse2022/`

### Run Jupyter Lab

Navigate to the repository root, if you are not already there. Activate one of the condas environements, for example 
```
conda activate ukrse2022_mlops_data_prep
```
Then run jupyter lab with the following command
```
jupyter lab
```
The Jupyter Lab interface will pop up in your default browser.


### Links
* [UK RSE Conference 2022](https://rsecon2022.society-rse.org/)
* [RSE Association](https://society-rse.org/)
