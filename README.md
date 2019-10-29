# Anomaly Detection

## Initialize virtual environment
(Contains all the dependencies)
```bash
source env/bin/activate
```

## Preprocess Data
```bash
python preprocess.py -d data/0db/fan/id_00 -o .
```

## Training Script
```bash
python train.py -dev gpu -d . -e 50
```
