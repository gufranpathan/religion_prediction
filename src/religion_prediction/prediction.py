
import os
import pickle

class ReligionPrediction:
    def __init__(self,base_dir="models/",type="multi_class"):
        model_name = "sepri_concat_False" if type=="multi_class" else ""
        self.model_dir = os.path.join(base_dir,type,model_name)
        self.classifier = "LOGIT"
        self.concat_model = False
        self.classifier_dir = os.path.join(self.model_dir, f'classifier_model_multiclass_{self.classifier}_concat_{self.concat_model}.sav')
        self.vectorizer_dir = os.path.join(self.model_dir, f'vectorizer_model_multiclass_{self.classifier}_concat_{self.concat_model}.sav')
        self.encoder_dir = os.path.join(self.model_dir, f'encoder_model_multiclass_{self.classifier}_concat_{self.concat_model}.sav')
        self.load_model()

    def load_model(self):
        with open(self.vectorizer_dir, 'rb') as f:
            self.multi_vectorizer = pickle.load(f)

        with open(self.classifier_dir, 'rb') as f:
            self.multi_clf = pickle.load(f)

        with open(self.encoder_dir, 'rb') as f:
            self.multi_model_id_to_category = pickle.load(f)

    def clean_names(self):
        pass

    def score(self):
        pass


ReligionPrediction()