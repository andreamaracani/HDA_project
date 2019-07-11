import util as u
import os
import numpy as np
input_path = "data/"
output_path = "data/26 silence/"
input_name = "noise"
output_name = "noise"
nFiles = 6
nSamples = 5000
duration = 1
seed = 30

#u.get_samples_from_noise(input_path, output_path, input_name, output_name, nFiles, nSamples, duration, seed)

#dataset = u.create_dataset(input_path,10)
tr,va,te,tr_l,va_l,te_l = u.create_dataset_and_split(input_path,  50, 0.7, (97, 40, 3), max_classes= 2)

print(tr.shape)
print(va.shape)
print(te.shape)
print(tr_l)
print(va_l)
print(te_l)
#feature1 = dataset[0,0]

# print(dataset.shape)
# print(feature1.shape)
# print(u.get_class_dirs(input_path))
