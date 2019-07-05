import util as u

input_path = "files/"
output_path = "data/26 silence/"
input_name = "noise"
output_name = "noise"
nFiles = 6          # number of input noise files
nSamples = 5000     # number of output noise files
duration = 1        # duration, in seconds, of each output file
seed = 30

u.get_samples_from_noise(input_path, output_path, input_name, output_name, nFiles, nSamples, duration, seed)
