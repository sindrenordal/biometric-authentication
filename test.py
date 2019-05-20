from keras.models import load_model
import numpy as np
import random as r

from sklearn.metrics import roc_curve, auc
from train import find_subject_ids, load_dataframe, split
from train import load_dataframe

def validate_model():
    auc_tot = []
    eer_tot = []
    subject_ids = find_subject_ids("models/lstm/")
    for subject_id in subject_ids:
        subject = subject_id.split('_')[0]
        neg = subject_ids[r.randint(0, len(subject_ids)-1)]
        neg_subject = neg.split('_')[0]
        if(neg_subject == subject):
            neg = subject_ids[r.randint(0, len(subject_ids)-1)]
            neg_subject = neg.split('_')[0]

        df_pos = load_dataframe(subject, "val")
        df_neg = load_dataframe(neg_subject, "val")

        df = np.concatenate([df_pos, df_neg], axis=0)

        X, y = split(subject, df)

        model_path = "models/min_300/"+subject_id+".h5"
        model = load_model(model_path)

        y_pred = model.predict_proba(X)
        y_pred = y_pred[:,1]

        fpr, tpr, thresholds = roc_curve(y, y_pred)
        roc_auc = auc(fpr, tpr)

        fnr = 1-tpr
        EER = fpr[np.nanargmin(np.absolute(fnr-fpr))]
        eer_tot.append(EER)
        auc_tot.append(roc_auc)
    return eer_tot, auc_tot


eer, auc = validate_model()
print("EER: ", np.mean(eer))
print("AUC: ", np.mean(auc))