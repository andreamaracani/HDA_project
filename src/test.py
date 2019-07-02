import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import sys
import features as f

PATH = 'files/'
FILE_NAME = '1.wav'

fs, data = wavfile.read(PATH+FILE_NAME)
print(len(data))
print('sample rate = ' + str(fs))

plt.figure(1)
plt.title('Signal Wave...')
plt.plot(data)
plt.show()

features = f.get_features(data,  fs, window_function=np.hamming, number_of_filters=40)

#print infos
print("features shape= ", features.shape)
print("static min = ", np.min(features[0]), "   static max = ", np.max(features[0]))
print("delta min = ", np.min(features[1]), "   delta max = ", np.max(features[1]))
print("delta-delta min = ", np.min(features[2]), "   delta-delta max = ", np.max(features[2]))


plt.figure(2)
plt.title('Static...')
plt.imshow(np.transpose(features[0]), extent=[0,97,0,40], cmap='jet', vmin=-2, vmax=22, origin='lowest', aspect='auto')
plt.colorbar()
plt.show()


plt.figure(3)
plt.title('Delta...')
plt.imshow(np.transpose(features[1]), extent=[0,97,0,40], cmap='jet', vmin=-2, vmax=22, origin='lowest', aspect='auto')
plt.colorbar()
plt.show()

plt.figure(4)
plt.title('Delta Delta...')
plt.imshow(np.transpose(features[2]), extent=[0,97,0,40], cmap='jet', vmin=-2, vmax=22, origin='lowest', aspect='auto')
plt.colorbar()
plt.show()






