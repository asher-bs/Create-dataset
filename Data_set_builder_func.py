# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 00:24:47 2023

@author: kids
"""
#audio_path=r"C:\temp\wn.wav"
#audio_in, sr = librosa.audio.load(audio_path, sr=None)  # Target signal 
#sf.write(r"C:\temp\after_HP.wav" ,audio_out, sr, subtype='PCM_24')
#https://hillel.org.il/podcast/se03ep05-nachaman-mishel-live-in-the-present-forgive-the-past/
#https://www.osimhistoria.com/ima-aba/ep07-udi

# https://github.com/SRPOL-AUI/storir
# pip install -e C:\Users\kids\anaconda3\Lib\site-packages\storir-master\storir-master

import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy 
import soundfile as sf
from scipy.io.wavfile import write
import glob
import pandas as pd
from storir import ImpulseResponse
import os


plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.xmargin"] = 0

def add_rir (audio_in,sr,rt60=600,edt=50,itdg=3,er_duration=80):
    # rt60 = 600
    # edt = 50
    # itdg = 3
    # er_duration = 80
    drr = int(rt60 * (- 1 / 100)) + np.random.randint(0, np.ceil(rt60 * (1 / 100)))

    rir = ImpulseResponse(rt60=rt60,
                          edt=edt,
                          itdg=itdg,
                          drr=drr,
                          er_duration=er_duration)


    rir_data = rir.generate(sampling_rate=sr)
    audio_rev=np.convolve(audio_in, rir_data,mode="valid")
    return audio_rev

def audio_norm (target_db,audio_in):
    audio_rms=20*np.log10(np.sqrt(np.mean(audio_in**2)))
    gain=target_db-audio_rms
    audio_out=audio_in*10**(gain/20)
    return audio_out

def audio_rms (audio_in):
    audio_rms=20*np.log10(np.sqrt(np.mean(audio_in**2))) 
    return audio_rms    

def front_back_mix (front_audio,back_audio,front_to_back_SNR):
        
#    sf.write(r"c:/temp/front.wav" ,front_audio, sr, subtype='PCM_24')

    
    # Add rir to back audio
    back_tmp=add_rir(back_audio,sr_back)
    
    # Calc front RMS for changing back RMS
    front_rms=audio_rms(front_audio)
    
    back_target_rms=front_rms-front_to_back_SNR
    back_audio=audio_norm(back_target_rms,back_tmp)
    
    mix_audio=front_audio
    mix_audio=front_audio[0:len(back_audio)]+back_audio
 #   sf.write(r"c:/temp/back_rir.wav" ,back_audio, sr, subtype='PCM_24')
    
    return mix_audio

def audio_compress (audio_in,sr,file_name):
    #sf.write(r"c:/temp/temp/audio_mix.wav" ,audio_in, sr, subtype='PCM_24')
    audio_resample=librosa.resample(audio_in,sr,8000)
    #sf.write(r"c:/temp/temp/audio_mix_resample.wav" ,audio_resample, 8000, subtype='PCM_24')
    audio_resample_int16=np.round(audio_resample*2**15)
    write(tmp_folder+"tmp.wav", 8000,  audio_resample_int16.astype(np.int16))
    cmd =exe_files_folder+ "ffmpeg.exe -i "+ tmp_folder+"tmp.wav -f s16le -ar 8000 -acodec pcm_s16le tmp.pcm"
    os.system(cmd)
    cmd = exe_files_folder+"coder.exe "+tmp_folder+"tmp.pcm " +tmp_folder+"tmp.g729 " 
    os.system(cmd)   
    cmd = exe_files_folder+"decoder.exe "+tmp_folder+"tmp.g729 " +tmp_folder+"tmp_g729.pcm " 
    os.system(cmd)  
    cmd =exe_files_folder+ "ffmpeg.exe -f s16le -ar 8000 -acodec pcm_s16le -i "+tmp_folder+"tmp_g729.pcm " + file_name
    os.system(cmd)
    cmd="del "+tmp_folder+"tmp.wav, tmp.pcm,tmp.g729,tmp_g729.pcm"
    os.system(cmd) 


SNR_front_range=[13,33]

front_to_back_SNR=np.random.randint(SNR_front_range[0],SNR_front_range[1])


front_files_path=r"E:\temp\dataset\front_set\*.wav"
back_files_path=r"E:\temp\dataset\back_set\*.wav"
tmp_folder=r"c:\temp\temp\\"
exe_files_folder=r"c:\temp\temp\\"
saved_files_folder="c:/temp/temp/data/"

front_files_list=glob.glob(front_files_path)
back_files_list=glob.glob(back_files_path)

# Load front random file from folder
front_rand=front_files_list[np.random.randint(0,len(front_files_list))]
front_tmp, sr = librosa.audio.load(front_rand, sr=None)  
front_name=front_rand.split("\\")[-1].split(".")[0]

# Load back random file from folder
back_rand=back_files_list[np.random.randint(0,len(back_files_list))]
back_tmp, sr_back = librosa.audio.load(back_rand, sr=None)  
back_name=back_rand.split("\\")[-1].split(".")[0]


assert sr == sr_back, "files sample rate should be the same !"

# Mix front and back with define SNR after adding rir to the back file
mix_audio=front_back_mix(front_tmp,back_tmp,front_to_back_SNR)

# Compressed and save the mixed audio file
file_name=str(front_name)+"__"+str(back_name)+"_SNR_-"+str(front_to_back_SNR)+".wav"
file_name=saved_files_folder+file_name

compressed_audio=audio_compress(mix_audio,sr,file_name)




