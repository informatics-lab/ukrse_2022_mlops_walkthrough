#!/bin/bash

export UKRSE_DATA_ROOT=~/data/ukrse2022/

mkdir $UKRSE_DATA_ROOT

# Raw Rotors dataset 
wget https://zenodo.org/record/6966937/files/2021_met_office_aviation_rotors.csv -P  ${UKRSE_DATA_ROOT}

# Preprocessed rotors dataset 
wget https://zenodo.org/record/7022648/files/2021_met_office_aviation_rotors_preprocessed.csv -P ${UKRSE_DATA_ROOT}

# Intake catalog file 
wget https://raw.githubusercontent.com/informatics-lab/ukrse_2022_mlops_walkthrough/main/rotors_catalog.yml -P  ${UKRSE_DATA_ROOT}

# Preprocessed UK cutout of ERA5 Mean Sea-level Pressure dataset for 2017-2021
wget https://zenodo.org/record/7022648/files/era5_mslp_UK_2017_2020.nc -P ${UKRSE_DATA_ROOT}


