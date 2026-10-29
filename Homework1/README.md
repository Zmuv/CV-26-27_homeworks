# Requirements
python 3.x (any version should be fine, i used 3.14.6)
numpy, matplotlib, opencv-python

## Make sure to use a python environment (conda or venv)
```bash
python -m venv .venv
```

## Activate the environment
(macOS)
```bash 
source .venv/bin/activate
```

(Windows)
```bash
.\venv\Scripts\Activate.ps1
```


## Install the required libraries with the following command:
```bash
pip install -r requirements.txt
```

# Creating degraded images
1. First Upload your images in the images/ folder
2. then modify degrade.py to add your image names to IMAGE_NAME
3. Make sure the folder images/degraded exists
4. execute:
```bash
python degrade.py
```

Some test images are already provided.


# Pipeline
To execute the pipeline simply run:
```bash
python homework1.py
```