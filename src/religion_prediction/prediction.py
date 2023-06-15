
import os
import pickle

import pandas as pd
from pydload import dload
import zipfile
import numpy as np
from unidecode import unidecode

class ReligionPrediction:
    def __init__(self,base_dir="models/",type="multi_class"):
        model_name = "sepri_concat_False" if type=="multi_class" else ""
        self.model_dir = os.path.join(base_dir,type,model_name)
        self.classifier = "LOGIT"
        self.concat_model = False
        self.classifier_dir = os.path.join(self.model_dir, f'classifier_model_multiclass_{self.classifier}_concat_{self.concat_model}.sav')
        self.vectorizer_dir = os.path.join(self.model_dir, f'vectorizer_model_multiclass_{self.classifier}_concat_{self.concat_model}.sav')
        self.encoder_dir = os.path.join(self.model_dir, f'encoder_model_multiclass_{self.classifier}_concat_{self.concat_model}.sav')

        models_downloaded=all([os.path.isfile(model_file_path) for model_file_path in [self.classifier_dir, self.vectorizer_dir, self.encoder_dir]])
        if not models_downloaded:
            self.download_models(self.model_dir,download_url="")
        self.load_model()

    def download_models(self, models_path, download_url):

        model_file_path = self.model_dir
        if not os.path.isfile(model_file_path):
            print('Downloading Multilingual model for transliteration')
            remote_url = download_url
            downloaded_zip_path = os.path.join(models_path, 'model.zip')

            dload(url=remote_url, save_to_path=downloaded_zip_path, max_time=None)

            if not os.path.isfile(downloaded_zip_path):
                exit(f'ERROR: Unable to download model from {remote_url} into {models_path}')

            with zipfile.ZipFile(downloaded_zip_path, 'r') as zip_ref:
                zip_ref.extractall(models_path)

            if os.path.isfile(model_file_path):
                os.remove(downloaded_zip_path)
            else:
                exit(f'ERROR: Unable to find models in {models_path} after download')

            print("Models downloaded to:", models_path)
            print("NOTE: When uninstalling this library, REMEMBER to delete the models manually")
        return model_file_path

    def load_model(self):
        with open(self.vectorizer_dir, 'rb') as f:
            self.multi_vectorizer = pickle.load(f)

        with open(self.classifier_dir, 'rb') as f:
            self.multi_clf = pickle.load(f)

        with open(self.encoder_dir, 'rb') as f:
            self.multi_model_id_to_category = pickle.load(f)

    def clean_names(self,data,name_col):
        new_name_col = name_col + "_clean"
        data[new_name_col] = data[name_col]
        data[new_name_col] = data[new_name_col].replace(np.nan, '', regex=True)
        # data[new_name_col] = data[new_name_col].apply(unidecode, meta=(new_name_col,'object'))
        data[new_name_col] = data[new_name_col].apply(lambda s: ''.join(x for x in s if x.isalpha() or x==" "))
        data[new_name_col] = data[new_name_col].str.upper()
        data[new_name_col] = data[new_name_col].replace(" ", "}{", regex=True)
        data[new_name_col] = "{" + data[new_name_col].astype(str) + "}"
        # return data

    def score(self, data, col_name):
        x = data[col_name]
        # print(x.head())
        # print(f'{datetime.datetime.now()}: Transforming Multi')
        multi_tfidf_matrix = self.multi_vectorizer.transform(x)
        # print(f'{datetime.datetime.now()}: Predicting Multi')
        data[col_name+'_multi_pred'] = self.multi_clf.predict(multi_tfidf_matrix)
        # print(f'{datetime.datetime.now()}: Mapping Multi')
        data[col_name+'_multi_pred'] = data[col_name+'_multi_pred'].map(self.multi_model_id_to_category)

        # print(f'{datetime.datetime.now()}: Predicting Proba Multi')
        multi_y_pred_score = self.multi_clf.predict_proba(multi_tfidf_matrix)
        data[col_name+'_multi_score'] = np.max(multi_y_pred_score, axis=1)

        # print(f'{datetime.datetime.now()}: Transforming Two')
        # two_tfidf_matrix = self.two_vectorizer.transform(x)

        # print(f'{datetime.datetime.now()}: Predicting Two')
        # data[col_name+'_two_pred'] = self.two_clf.predict(two_tfidf_matrix)
        # data[col_name+'_two_pred'] = data[col_name+'_two_pred'].map(lambda x: 'Muslim' if x == 1 else "Non-Muslim")
        # two_y_pred_score = self.two_clf.predict_proba(two_tfidf_matrix)
        # data[col_name+'_two_score'] = np.max(two_y_pred_score, axis=1)


        # return data

    def clean_and_score(self, data, col_name):
        self.clean_names(data, col_name)
        self.score(data, col_name+"_clean")
