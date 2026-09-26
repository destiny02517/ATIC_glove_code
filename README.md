# ATIC Glove Code

This repository contains the code and data supporting the ATIC glove study, including multimodal tactile sensing, geometry-defined in-sensor computing (GDIC), spatial recognition, sign language recognition, object discrimination, physical inference, and edge inference.

## Repository structure

- `01_Sensor_Characterization/`
  - Code and data for characterization of the hydrogel–Ecoflex tactile sensing unit
  - Related to Fig. 2 and corresponding Supplementary Figures

- `02_Spatial_Recognition/`
  - Spatial position discrimination using the 40-channel baseline configuration and TCO & ISO Outputs
  - Includes Scenario A and Scenario B
  - Related to Fig. 3 and corresponding Supplementary Figures

- `03_Sign_Language/`
  - 46-word sign language recognition
  - Pearson correlation analysis
  - Sentence-level recognition
  - Related to Fig. 4 and corresponding Supplementary Figures

- `04_Object_Discrimination/`
  - 21-class object discrimination using the 40-channel baseline configuration, TCO & ISO Outputs, and edge inference
  - Related to Fig. 5 and corresponding Supplementary Figures

- `05_Physical_Inference/`
  - Code and data for physical-property estimation, including object weight prediction
  - Related to Fig. 5f and corresponding Supplementary Figures

- `06_Application_Demonstrations/`
  - Code and data for application-level demonstrations, including sign language communication and digital-twin interaction
  - Related to Fig. 6 and corresponding Supplementary Figures

- `utilities/`
  - Common scripts for data reading, preprocessing, and other shared functions

## System requirements

- MATLAB >= R2022a
- Python >= 3.8

## Data organization

Each task folder contains corresponding `code/` and `data/` subfolders where applicable.

Example:

```text
02_Spatial_Recognition/
├── Scenario_A_20Class/
│   ├── code/
│   └── data/
└── Scenario_B_30Class/
    ├── 40CH/
    │   ├── code/
    │   └── data/
    └── TCO_ISO/
        ├── code/
        └── data/
