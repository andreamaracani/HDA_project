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

                text: 'On'
                bold: True
                font_size: '25sp'
                color: 1,1,1,1
                background_normal: 'img/label_b.png'
                background_down: 'img/label_b.png'
                border: 0,0,0,0
                
             
            Button:

                text: 'Off'
                bold: True
                font_size: '25sp'
                color: 1,1,1,1
                background_normal: 'img/label_b.png'
                background_down: 'img/label_b.png'
                border: 0,0,0,0
              
                
            Button:

                text: 'Up'
                bold: True
                font_size: '25sp'
                color: 1,1,1,1
                background_normal: 'img/label_b.png'
                background_down: 'img/label_b.png'
                border: 0,0,0,0
               
                
            Button:

                text: 'Down'
                bold: True
                font_size: '25sp'
                color: 1,1,1,1
                background_normal: 'img/label_b.png'
                background_down: 'img/label_b.png'
                border: 0,0,0,0
              
                
            Button:

                text: 'Left'
                bold: True
                font_size: '25sp'
                color: 1,1,1,1
                background_normal: 'img/label_b.png'
                background_down: 'img/label_b.png'
                border: 0,0,0,0
                
                
            Button:

                text: 'Right'
                bold: True
                font_size: '25sp'
                color: 1,1,1,1
                background_normal: 'img/label_b.png'
                background_down: 'img/label_b.png'
                border: 0,0,0,0
                
                
            Button:

                text: 'Yes'
                bold: True
                font_size: '25sp'
                color: 1,1,1,1
                background_normal: 'img/label_b.png'
                background_down: 'img/label_b.png'
                border: 0,0,0,0
           
                
            Button:

                text: 'No'
                bold: True
                font_size: '25sp'
                color: 1,1,1,1
                background_normal: 'img/label_b.png'
                background_down: 'img/label_b.png'
                border: 0,0,0,0
         
                
            Button:

                text: 'Go'
                bold: True
                font_size: '25sp'
                color: 1,1,1,1
                background_normal: 'img/label_b.png'
                background_down: 'img/label_b.png'
                border: 0,0,0,0
            
                
            Button:

                text: 'Stop'
                bold: True
                font_size: '25sp'
                color: 1,1,1,1
                background_normal: 'img/label_b.png'
                background_down: 'img/label_b.png'
                border: 0,0,0,0
            
                
            Button:

                text: 'Silence'
                bold: True
                font_size: '25sp'
                color: 1,1,1,1
                background_normal: 'img/label_b.png'
                background_down: 'img/label_b.png'
                border: 0,0,0,0
               
                
            Button:

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
                on_press:  root.start_recording(wav_plot, f_plot)
                
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
    RECORD_SECONDS = 1.5

    def start_recording(self, wav_plot_instance, f_plot_instance):

        print("start...")
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

        # evaluate features
        features = f.get_time_padded_features(data, fs, 200, frame_length=400, frame_step=160, number_of_filters=40,
                                              add_delta=False, random_time_shift=False)
        # max and min
        fmin = np.min(features[:, :, 0])
        fmax = np.max(features[:, :, 0])

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



class HDAApp(App):

    def build(self):
        return AudioInterface()


if __name__ == "__main__":
    HDAApp().run()