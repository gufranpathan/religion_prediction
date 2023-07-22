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

two_class = ReligionPrediction(model_class='two_class')
multi_class = ReligionPrediction(model_class='multi_class')
two_class.clean_and_score(names_df,'person_name')
multi_class.clean_and_score(names_df,'person_name')
names_df

```

Three new columns with the 'clean' name, prediction, and probability score
```
>>> names_df

person_name         age gender  person_name_clean       person_name_clean_two_pred  person_name_clean_two_pred_score    person_name_clean_multi_pred    person_name_clean_multi_score
ahmed khan	    23	M	{AHMED}{KHAN}		1	                    0.99977	                        Muslim	                        0.99687
Kumar Vishwas	    35	M	{KUMAR}{VISHWAS}	0	                    0.99963                     	Hindu	                        0.9828
Rabindranath Tagore 57	M	{RABINDRANATH}{TAGORE}	0	                    0.99744	                        Hindu                           0.99499
Razia Khatoon       12	F	{RAZIA}{KHATOON}	1	                    0.99478                     	Muslim	                        0.98558
Yusuf Khan          32	M	{YUSUF}{KHAN}		1	                    0.99987	                        Muslim	                        0.99986
Dilip Kumar         32	M	{DILIP}{KUMAR}		0	                    0.99731	                        Jain                            0.97413
```