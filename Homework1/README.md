# Requirements
Python 3.14.6 (other 3.x versions should work too)

Dependencies (exact versions in requirements.txt): numpy, matplotlib, opencv-python

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
.\.venv\Scripts\Activate.ps1
```


## Install the required libraries with the following command:
```bash
pip install -r requirements.txt
```

# Creating degraded images
1. First Upload your images in the images/ folder
2. then modify degrade.py to add your image names to IMAGES_NAMES
3. execute:
```bash
python degrade.py
```

The contrast is reduced around the mean (alpha = 0.6) and gaussian noise is added (std = 25, seed 42).

Some test images are already provided, together with their degraded version.

# Pipeline
To execute the pipeline simply run:
```bash
python homework1.py
```

The metrics are printed and saved in images/output/metrics.csv, the processed images and the comparison figures are saved in images/output/

# File structure
```
Homework1/
└── images/
    ├── *.png                   # original images
    ├── degraded/               # degraded images (.png used by the code)
    └── output/
        ├── metrics.csv                 # PSNR and MSE for each image and method
        ├── linear_stretch/             # rgb_*.png and hls_*.png
        ├── histogram_equalization/
        ├── smoothing_gamma/            # bilateral filter + gamma correction
        └── comparison/                 # original / degraded / result figures with histograms
```

Apart from output/comparison the other results are not uploaded due to file size constraints, you can reach the same results by executing homework1.py

# Images
All images are photographs taken by the author.

