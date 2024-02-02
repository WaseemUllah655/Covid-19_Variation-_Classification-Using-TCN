import tensorflow as tf
from  Utils import *
import numpy as np
from random import shuffle
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def detect(inference_dict, thresholds_dict):
    # return results
    results = {'detect_entropy': {}}
    results['detect_entropy'] = {
        thresh: inference_dict['entropy_values'] >= thresh
        for thresh in thresholds_dict['entropy_scale']
    }

    return results
    


# Recreate the exact same model purely from the file
new_model = tf.keras.models.load_model(r'C:\Users\aminu\Documents\COVID-19\COVID19VariantClassification\Classification_nucleotide_Model_1')



all_data_class8 = read_seq_new(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (C.1.2)\ncbi_dataset\data\genomic.fna',7)
all_data_class9 = read_seq_new(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (P.2)\ncbi_dataset\data\genomic.fna',8)
all_data_class10 = read_seq_new(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (B.1.427)\ncbi_dataset\data\genomic.fna',9)



all_data=[]
for itm in all_data_class8:
    all_data.append(itm)
for itm in all_data_class9:
    all_data.append(itm)
for itm in all_data_class10:
    all_data.append(itm)

x=[]
y=[]
for itm in all_data:
    x.append(itm[0])
    y.append(np.array(itm[1])) 

x=np.asarray(x,dtype=np.float)

y_scores= new_model.predict(x)

ent1 = cal_entropy(y_scores)

# ent1 = np.std(y_scores, axis=1)

print(ent1, np.min(ent1), np.max(ent1))

clean_inference = {}
threshold_dict = {}
clean_inference['entropy_values'] = ent1
threshold_dict["entropy_scale"] = np.logspace(np.log10(0.02), np.log10(0.50), 60)
trueresults = detect(clean_inference, threshold_dict)







############ nucleotide seq data

all_data_class1 = read_seq_new(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (B)\ncbi_dataset\data\genomic.fna',0)
all_data_class2 = read_seq_new(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (B.1.1.7)\ncbi_dataset\data\genomic.fna',1)
all_data_class3 = read_seq_new(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (B.1.351)\ncbi_dataset\data\genomic.fna',2)
all_data_class4 = read_seq_new(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (B.1.617.2)\ncbi_dataset\data\genomic.fna',3)
all_data_class5 = read_seq_new(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (C.37)\ncbi_dataset\data\genomic.fna',4)
all_data_class6 = read_seq_new(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (P.1)\ncbi_dataset\data\genomic.fna',5)
all_data_class7 = read_seq_new(r'C:\Users\aminu\Documents\COVID-19\CovidVariantsDataset\SARS-CoV-2 (B.1.525)\ncbi_dataset\data\genomic.fna',6)





all_data=[]
for itm in all_data_class1:
    all_data.append(itm)
for itm in all_data_class2:
    all_data.append(itm)
for itm in all_data_class3:
    all_data.append(itm)
for itm in all_data_class4:
    all_data.append(itm)
for itm in all_data_class5:
    all_data.append(itm)
for itm in all_data_class6:
    all_data.append(itm)
for itm in all_data_class7:
    all_data.append(itm)
shuffle(all_data)

x=[]
y=[]
for itm in all_data:
    x.append(itm[0])
    y.append(np.array(itm[1])) 


x=np.asarray(x,dtype=np.float)

y_scores= new_model.predict(x)

ent2 = cal_entropy(y_scores)

# ent2 = np.std(y_scores, axis=1)

print(ent1, np.min(ent2), np.max(ent2))
clean_inference = {}
clean_inference['entropy_values'] = ent2
falseresults = detect(clean_inference, threshold_dict)




truep = {}
falsep = {}

for k in trueresults:    
    truep[k] = {}
    falsep[k] = {}
    for thresh in trueresults[k]:                
        truep[k][thresh] = str(trueresults[k][thresh].mean())
        falsep[k][thresh] = str(falseresults[k][thresh].mean())

tp = np.array( list(list(truep.values())[0].values()), dtype=np.float32)
fp = np.array( list(list(falsep.values())[0].values()), dtype=np.float32)

fp = np.concatenate((fp, [0]), axis=0)
tp = np.concatenate((tp, [0]), axis=0)
fp = fp/np.max(fp)
tp = tp/np.max(tp)
plt.plot(sorted(fp), sorted(tp), linewidth=4, label=' (AUC=%0.2f)' %auc(sorted(fp),sorted(tp)))


no_skill = len(fp[fp==1]) / len(fp)
plt.plot([0, 1], [0, 1], linestyle='--')



plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend()
plt.grid()
plt.show()
