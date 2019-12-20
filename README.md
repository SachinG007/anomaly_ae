# Anomaly Detection

| Machine type | Model ID | AUC | 
| ------------ |:--------:| ---:|
| Valve        | 00       | 0.55| 
|              | 02       | 0.59|
|              | 04       | 0.65|
|              | 06       | 0.66|
| Pump         | 00       | 0.65|
|              | 02       | 0.46|
|              | 04       | 0.95|
|              | 06       | 0.76|
|Fan           | 00       | 0.63|
|              | 02       | 0.83|
|              | 04       | 0.75|
|              | 06       | 0.97|
|Slide rail    | 00       | 0.99|
|              | 02       | 0.79|
|              | 04       | 0.78|
|              | 06       | 0.56|

[Paper](https://arxiv.org/abs/1909.09347)

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