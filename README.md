# Anomaly Detection

## Initialize virtual environment
Create virtual environment and install dependencies.
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Preprocess Data
```bash
python preprocess.py -d <path_to_dataset> -o <path_to_save_output>

```

For example to preprocess data for fan 00, use the following
```bash
python preprocess.py -d data/0db/fan/id_00 -o .
```

## Training Script
```bash
python train.py -dev gpu -d . -e 50
```
