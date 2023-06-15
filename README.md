# Prediction religions in India

## Setup
It's always a good idea to create a virtual environment for your project. Navigate to your project directory and run:

```commandline
python -m venv venv
```

To activate your virtual environment on linux run the following from your project directory:

```commandline
source venv\bin\activate
```
To activate your virtual environment on Windows  run the following from your project directory:

```shell
.\venv\Scripts\activate
```
Now install the religion prediction package in your library

```shell
pip install git+https://github.com/gufranpathan/religion_prediction.git#egg=religion_prediction
```

## Execution

```python
import pandas as pd

# Sample Data Frame
names_df = pd.DataFrame({'person_name':['ahmed khan', 'Kumar Vishwas', 'Rabindranath Tagore','Razia Khatoon', 'Yusuf Khan', 'Dilip Kumar'],
                         'age':[23,35,57,12,32,32],
                         'gender':['M','M','M','F','M','M']})
```
This is what a sample dataframe might look like:
```
>>> names_df
           person_name  age gender
0           ahmed khan   23      M
1        Kumar Vishwas   35      M
2  Rabindranath Tagore   57      M
3        Razia Khatoon   12      F
4           Yusuf Khan   32      M
5          Dilip Kumar   32      M
```

Running the prediction algorithm:
```python
from religion_prediction.prediction import ReligionPrediction

religion_prediction = ReligionPrediction()
religion_prediction.clean_and_score(names_df,'person_name')
names_df
```

Three new columns with the 'clean' name, prediction, and probability score
```
>>> names_df

person_name         age gender  person_name_clean       person_name_clean_multi_pred    person_name_clean_multi_score
ahmed khan          23  M       {AHMED}{KHAN}           Muslim                          0.99687
Kumar Vishwas       35  M       {KUMAR}{VISHWAS}        Hindu                           0.98280
Rabindranath Tagore 57  M       {RABINDRANATH}{TAGORE}  Hindu                           0.99499
Razia Khatoon       12  F       {RAZIA}{KHATOON}        Muslim                          0.98558
Yusuf Khan          32  M       {YUSUF}{KHAN}           Muslim                          0.99986
Dilip Kumar         32  M       {DILIP}{KUMAR}          Jain                            0.97413
```