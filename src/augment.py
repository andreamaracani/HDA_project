import util as u

dataset_path = "data/"

u.augment_dataset(dataset_path, sample_rate=16000, maximum_length=17000,
                  folder_name="augmentation/", percentage=0.25,
                  pitch_change_min=-1, pitch_change_max=1,
                  speed_change_min=0.95, speed_change_max=1.05,
                  noise_max=0.005, seed=1)
