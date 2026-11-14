# Requirements
Python 3.14.6 (other 3.x versions should work too)

Dependencies (exact versions in requirements.txt): numpy, torch, torchvision, scikit-learn

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

## Dataset
The dataset requires no extra steps, it will download when the script is executed

## Execution
```bash
python main.py
```