import util as u

input_path = "data/"
test_samples_per_class = 50
training_percentage = 0.7

dataset = u.create_dataset(input_path)
train, val, test, train_l, val_l, test_l = u.split_dataset(dataset,  test_samples_per_class, training_percentage)

print("training size = ", train.shape)
print("validation size = ", val.shape)
print("test size = ", test.shape)
print("training labels size = ", train_l.shape)
print("validation labels size = ", val_l.shape)
print("test labels size = ", test_l.shape)

# RESULTS
# In the dataset there are 110829 files divided in 27 classes
# Class 0 has 4052 samples.
# Class 1 has 3890 samples.
# Class 2 has 3880 samples.
# Class 3 has 3727 samples.
# Class 4 has 3728 samples.
# Class 5 has 4052 samples.
# Class 6 has 3860 samples.
# Class 7 has 3998 samples.
# Class 8 has 3787 samples.
# Class 9 has 3934 samples.
# Class 10 has 3880 samples.
# Class 11 has 4044 samples.
# Class 12 has 3941 samples.
# Class 13 has 3845 samples.
# Class 14 has 3745 samples.
# Class 15 has 1557 samples.
# Class 16 has 1664 samples.
# Class 17 has 3801 samples.
# Class 18 has 3778 samples.
# Class 19 has 3723 samples.
# Class 20 has 3917 samples.
# Class 21 has 3872 samples.
# Class 22 has 1592 samples.
# Class 23 has 1579 samples.
# Class 24 has 1575 samples.
# Class 25 has 20408 samples.
# Class 26 has 5000 samples.
#
# training size =  (76644,)
# validation size =  (32835,)
# test size =  (1350,)
# training labels size =  (76644,)
# validation labels size =  (32835,)
# test labels size =  (1350,)
#
# Process finished with exit code 0
