from kivy.app import App
from kivy.lang import Builder
from kivy.properties import NumericProperty
from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.config import Config
import pyaudio
import wave
import keras.backend as K
from model import load

import matplotlib.pyplot as plt


import numpy as np
from scipy.io import wavfile
import features as f


Config.set('graphics', 'resizable', '0') #0 being off 1 being on as in true/false
Config.set('graphics', 'width', '1000')
Config.set('graphics', 'height', '800')


Builder.load_string('''

<AudioInterface>:

    BoxLayout:
        orientation:'vertical'
        size: root.width, root.height
        
        Image:
            id: wav_plot
            source: 'img/topl.png'
            size: self.texture_size
            pos: self.pos
            size: self.size
            allow_stretch: True
            keep_ratio: False
        Image:
            id: f_plot
            source: 'img/toph.png'
            size: self.texture_size
            pos: self.pos
            size: self.size
            allow_stretch: True
            keep_ratio: False
                
        GridLayout:
            cols:6
            Button:
                id: id0
                text: 'On'
                bold: True
                font_size: '25sp'
                color: 1,1,1,1
                background_normal: 'img/label_b.png'
                background_down: 'img/label_b.png'
                border: 0,0,0,0
                
             
            Button:
                id: id1
                text: 'Off'
                bold: True
                font_size: '25sp'
                color: 1,1,1,1
                background_normal: 'img/label_b.png'
                background_down: 'img/label_b.png'
                border: 0,0,0,0
              
                
            Button:
                id: id2
                text: 'Up'
                bold: True
                font_size: '25sp'
                color: 1,1,1,1
                background_normal: 'img/label_b.png'
                background_down: 'img/label_b.png'
                border: 0,0,0,0
               
                
            Button:
                id: id3
                text: 'Down'
                bold: True
                font_size: '25sp'
                color: 1,1,1,1
                background_normal: 'img/label_b.png'
                background_down: 'img/label_b.png'
                border: 0,0,0,0
              
                
            Button:
                id: id4
                text: 'Left'
                bold: True
                font_size: '25sp'
                color: 1,1,1,1
                background_normal: 'img/label_b.png'
                background_down: 'img/label_b.png'
                border: 0,0,0,0
                
                
            Button:
                id: id5
                text: 'Right'
                bold: True
                font_size: '25sp'
                color: 1,1,1,1
                background_normal: 'img/label_b.png'
                background_down: 'img/label_b.png'
                border: 0,0,0,0
                
                
            Button:
                id: id6
                text: 'Yes'
                bold: True
                font_size: '25sp'
                color: 1,1,1,1
                background_normal: 'img/label_b.png'
                background_down: 'img/label_b.png'
                border: 0,0,0,0
           
                
            Button:
                id: id7
                text: 'No'
                bold: True
                font_size: '25sp'
                color: 1,1,1,1
                background_normal: 'img/label_b.png'
                background_down: 'img/label_b.png'
                border: 0,0,0,0
         
                
            Button:
                id: id8
                text: 'Go'
                bold: True
                font_size: '25sp'
                color: 1,1,1,1
                background_normal: 'img/label_b.png'
                background_down: 'img/label_b.png'
                border: 0,0,0,0
            
                
            Button:
                id: id9
                text: 'Stop'
                bold: True
                font_size: '25sp'
                color: 1,1,1,1
                background_normal: 'img/label_b.png'
                background_down: 'img/label_b.png'
                border: 0,0,0,0
            
                
            Button:
                id: id10
                text: 'Silence'
                bold: True
                font_size: '25sp'
                color: 1,1,1,1
                background_normal: 'img/label_b.png'
                background_down: 'img/label_b.png'
                border: 0,0,0,0
               
                
            Button:
                id: id11
                text: 'Unknown'
                bold: True
                font_size: '25sp'
                color: 1,1,1,1
                background_normal: 'img/label_b.png'
                background_down: 'img/label_b.png'
                border: 0,0,0,0
         
                
        BoxLayout:
            orientation:'horizontal'
            
            Image:
                
                source: 'img/left.jpg'
                pos: self.pos
                size: self.size
                allow_stretch: True
                keep_ratio: False
            
            Button:
                id: record_button
                background_normal: 'img/button_up.png'
                background_down: 'img/button_down.png'
                size: 200,200
                size_hint: None, None
                on_release:  root.start_recording(wav_plot, f_plot, id0, id1, id2, id3, id4, id5, id6, id7, id8, id9, id10,id11)
                
            Image:
                source: 'img/right.jpg'
                pos: self.pos
                size: self.size
                allow_stretch: True
                keep_ratio: False
        
    
''')


class AudioInterface(Widget):

    WAVE_OUTPUT_FILENAME = "tmp/record.wav"
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 1
    MODEL_PATH = 'models/model'


    model = load(MODEL_PATH, custom_objects={"backend": K})

    def start_recording(self, wav_plot_instance, f_plot_instance, id0, id1, id2, id3, id4, id5, id6, id7, id8, id9, id10, id11):


        id0.background_normal = 'img/label_b.png'
        id0.background_down = 'img/label_b.png'
        id1.background_normal = 'img/label_b.png'
        id1.background_down = 'img/label_b.png'
        id2.background_normal = 'img/label_b.png'
        id2.background_down = 'img/label_b.png'
        id3.background_normal = 'img/label_b.png'
        id3.background_down = 'img/label_b.png'
        id4.background_normal = 'img/label_b.png'
        id4.background_down = 'img/label_b.png'
        id5.background_normal = 'img/label_b.png'
        id5.background_down = 'img/label_b.png'
        id6.background_normal = 'img/label_b.png'
        id6.background_down = 'img/label_b.png'
        id7.background_normal = 'img/label_b.png'
        id7.background_down = 'img/label_b.png'
        id8.background_normal = 'img/label_b.png'
        id8.background_down = 'img/label_b.png'
        id9.background_normal = 'img/label_b.png'
        id9.background_down = 'img/label_b.png'
        id10.background_normal = 'img/label_b.png'
        id10.background_down = 'img/label_b.png'
        id11.background_normal = 'img/label_b.png'
        id11.background_down = 'img/label_b.png'


        audio = pyaudio.PyAudio()

        # start Recording
        stream = audio.open(format=self.FORMAT, channels=self.CHANNELS,
                            rate=self.RATE, input=True,
                            frames_per_buffer=self.CHUNK)
        print("recording...")

        frames = []

        for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            data = stream.read(self.CHUNK)
            frames.append(data)

        print("finished recording")
        print("Save WAV file")

        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()

        waveFile = wave.open(self.WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(self.CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(self.FORMAT))
        waveFile.setframerate(self.RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

        print("Evaluating the file...")
        fs, data = wavfile.read(self.WAVE_OUTPUT_FILENAME)
        data = data[-16000:]


        #####################################     PLOT 1     ###########################################################
        # create and save WAV plot
        fig = plt.figure(figsize=(20, 4))
        ax = fig.add_subplot(111)
        ax.set_facecolor('black')
        ax.spines['bottom'].set_color('white')
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['top'].set_color('white')
        ax.spines['top'].set_linewidth(3)
        ax.spines['left'].set_color('white')
        ax.spines['left'].set_linewidth(3)
        ax.spines['right'].set_color('white')
        ax.spines['right'].set_linewidth(3)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white', length=10, width=3, labelsize=20)
        ax.tick_params(axis='y', colors='white', length=10, width=3, labelsize=20)
        plt.plot(data, color="white")
        plt.savefig("tmp/fig1.png", facecolor='black')
        plt.close(fig)

        # show WAV plot
        wav_plot_instance.source = "tmp/fig1.png"
        wav_plot_instance.reload()

        ####################################     FEATURES     ##########################################################


        # evaluate features
        features = f.get_time_padded_features(data, sample_rate=fs,
                                              # PADDING
                                              target_frame_number=135,
                                              random_time_shift=True,
                                              smooth=True,
                                              smooth_length=5,

                                              pre_emphasis_coef=0.95,
                                              # FRAMING PARAMETERS
                                              frame_length=1024,
                                              frame_step=128,
                                              window_function=np.hamming,

                                              # MEL FILTERS PARAMETERS
                                              hertz_from=300,
                                              hertz_to=None,
                                              number_of_filters=80,

                                              # FFT PARAMETERS
                                              power_of_2=True,

                                              # OUTPUT SETTINGS
                                              dtype='float32',
                                              use_dct=False,
                                              add_delta=False,

                                              # NORMALIZATION
                                              shift_static=0,
                                              scale_static=1,
                                              shift_delta=0,
                                              scale_delta=1,
                                              shift_delta_delta=0,
                                              scale_delta_delta=1)
        # max and min
        fmin = np.min(features[:, :, 0])
        fmax = np.max(features[:, :, 0])


        #####################################     PLOT 2     ###########################################################
        # create features plot
        fig = plt.figure(figsize=(20, 4))

        ax = fig.add_subplot(111)
        ax.set_facecolor('black')
        ax.spines['bottom'].set_color('white')
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['top'].set_color('white')
        ax.spines['top'].set_linewidth(3)
        ax.spines['left'].set_color('white')
        ax.spines['left'].set_linewidth(3)
        ax.spines['right'].set_color('white')
        ax.spines['right'].set_linewidth(3)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white', length=10, width=3, labelsize=20)
        ax.tick_params(axis='y', colors='white', length=10, width=3, labelsize=20)

        plt.imshow(np.transpose(features[:, :, 0]), cmap='jet', vmin=fmin, vmax=fmax, origin='lowest', aspect='auto')
        # plt.colorbar()

        plt.savefig("tmp/fig2.png", facecolor="black")
        plt.close(fig)

        # show feature plot
        f_plot_instance.source = "tmp/fig2.png"
        f_plot_instance.reload()

        ################################################################################################################
        #

        # adding batch dim
        prediction = self.model.predict(np.expand_dims(features, axis=0))

        print("prediction =", str(prediction))
        choice = np.argmax(prediction)

        if choice == 0:
            id0.background_normal = 'img/label_b_sel.png'
            id0.background_down = 'img/label_b_sel.png'
        if choice == 1:
            id1.background_normal = 'img/label_b_sel.png'
            id1.background_down = 'img/label_b_sel.png'
        if choice == 2:
            id2.background_normal = 'img/label_b_sel.png'
            id2.background_down = 'img/label_b_sel.png'
        if choice == 3:
            id3.background_normal = 'img/label_b_sel.png'
            id3.background_down = 'img/label_b_sel.png'
        if choice == 4:
            id4.background_normal = 'img/label_b_sel.png'
            id4.background_down = 'img/label_b_sel.png'
        if choice == 5:
            id5.background_normal = 'img/label_b_sel.png'
            id5.background_down = 'img/label_b_sel.png'
        if choice == 6:
            id6.background_normal = 'img/label_b_sel.png'
            id6.background_down = 'img/label_b_sel.png'
        if choice == 7:
            id7.background_normal = 'img/label_b_sel.png'
            id7.background_down = 'img/label_b_sel.png'
        if choice == 8:
            id8.background_normal = 'img/label_b_sel.png'
            id8.background_down = 'img/label_b_sel.png'
        if choice == 9:
            id9.background_normal = 'img/label_b_sel.png'
            id9.background_down = 'img/label_b_sel.png'
        if choice == 10:
            id10.background_normal = 'img/label_b_sel.png'
            id10.background_down = 'img/label_b_sel.png'
        if choice == 11:
            id11.background_normal = 'img/label_b_sel.png'
            id11.background_down = 'img/label_b_sel.png'


class HDAApp(App):

    def build(self):
        return AudioInterface()


if __name__ == "__main__":
    HDAApp().run()