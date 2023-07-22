
import os
import pickle

import pandas as pd
from pydload import dload
import zipfile
import numpy as np
from unidecode import unidecode

import requests

MODEL_URLS ={
    'multi_class_sepri_concat_False': 'https://api.github.com/repos/gufranpathan/religion_prediction/releases/assets/112815800'
}

'Authorization: token my_access_token' 'https://api.github.com/repos/:owner/:repo/releases/assets/:id'

mb = 1024 * 1024
# download_url = 'https://api.github.com/repos/gufranpathan/religion_prediction/releases/assets/112815800'
GITHUB_TOKEN = ''


class ReligionPrediction:
    def __init__(self,base_dir="models/",model_class="multi_class"):
        self.model_class = model_class
        self.classifier = "LOGIT"
        self.concat_model = False
        if model_class=="multi_class":
            model_name = "sepri_concat_False"
            self.model_dir = os.path.join(base_dir, self.model_class, model_name)
            self.classifier_dir = os.path.join(self.model_dir,
                                               f'classifier_model_multiclass_{self.classifier}_concat_{self.concat_model}.sav')
            self.vectorizer_dir = os.path.join(self.model_dir,
                                               f'vectorizer_model_multiclass_{self.classifier}_concat_{self.concat_model}.sav')
            self.encoder_dir = os.path.join(self.model_dir,
                                            f'encoder_model_multiclass_{self.classifier}_concat_{self.concat_model}.sav')
            models_downloaded=all([os.path.isfile(model_file_path) for model_file_path in [self.classifier_dir, self.vectorizer_dir, self.encoder_dir]])

        elif self.model_class=="two_class":
            model_name= "religion_muslim_non_hm_include_LOGIT_concat_False"
            self.model_dir = os.path.join(base_dir, self.model_class, model_name)
            self.classifier_dir = os.path.join(self.model_dir,
                                               'model_2class_muslim_non_hm_include_LOGIT_concat_False.sav')
            self.vectorizer_dir = os.path.join(self.model_dir,
                                               'vectorizer_2class_muslim_non_hm_include_LOGIT_concat_False.sav')
            models_downloaded = all([os.path.isfile(model_file_path) for model_file_path in
                                     [self.classifier_dir, self.vectorizer_dir]])

        print('Cehcking if models are downloaded')
        if not models_downloaded:
            self.download_models(model_class=self.model_class,model_name=model_name)
        self.load_model()

    def download_models(self, model_class, model_name):
        print('Downloading models')
        download_url = MODEL_URLS[model_class + "_" + model_name]
        model_zip_name = f'{model_class}_{model_name}.zip'

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
        downloaded_zip_path = os.path.join(self.model_dir + model_zip_name)
        print(f'Downloading model from {download_url}')
        # dload(url=download_url, save_to_path=downloaded_zip_path, max_time=None)

        headers = {'Authorization': f'Bearer {GITHUB_TOKEN}', 'Accept': 'application/octet-stream'}
        response = requests.get(download_url, headers=headers, stream=True, verify=True, allow_redirects=True)
        with open(downloaded_zip_path, 'wb') as f:
            for chunk in response.iter_content(mb):
                f.write(chunk)

        with zipfile.ZipFile(downloaded_zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.model_dir)

        models_downloaded=all([os.path.isfile(model_file_path) for model_file_path in [self.classifier_dir, self.vectorizer_dir, self.encoder_dir]])

        if models_downloaded:
            os.remove(downloaded_zip_path)
        else:
            exit(f'ERROR: Unable to find models in {self.model_dir} after download')

        print("Models downloaded to:", self.model_dir)

    def load_model(self):
        with open(self.vectorizer_dir, 'rb') as f:
            self.vectorizer = pickle.load(f)

        with open(self.classifier_dir, 'rb') as f:
            self.clf = pickle.load(f)
        if self.model_class=='multi_class':
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
        pred_col_suffix = '_multi_pred' if self.model_class=='multi_class' else '_two_pred'
        # print(x.head())
        # print(f'{datetime.datetime.now()}: Transforming Multi')
        multi_tfidf_matrix = self.vectorizer.transform(x)
        # print(f'{datetime.datetime.now()}: Predicting Multi')
        data[col_name+pred_col_suffix] = self.clf.predict(multi_tfidf_matrix)
        # print(f'{datetime.datetime.now()}: Mapping Multi')
        if self.model_class=='multi_class':
            data[col_name+'_multi_pred'] = data[col_name+'_multi_pred'].map(self.multi_model_id_to_category)

        # print(f'{datetime.datetime.now()}: Predicting Proba Multi')
        multi_y_pred_score = self.clf.predict_proba(multi_tfidf_matrix)
        data[col_name+pred_col_suffix+'_score'] = np.max(multi_y_pred_score, axis=1)

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

# import pandas as pd
#
# # Sample Data Frame
# names_df = pd.DataFrame({'person_name':['ahmed khan', 'Kumar Vishwas', 'Rabindranath Tagore','Razia Khatoon', 'Yusuf Khan', 'Dilip Kumar'],
#                          'age':[23,35,57,12,32,32],
#                          'gender':['M','M','M','F','M','M']})
# from religion_prediction.prediction import ReligionPrediction
#
# two_class = ReligionPrediction(model_class='two_class')
# multi_class = ReligionPrediction(model_class='multi_class')
# two_class.clean_and_score(names_df,'person_name')
# multi_class.clean_and_score(names_df,'person_name')
# names_df
#
