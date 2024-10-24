    


import requests
# from protocal import TTSRequest, TTSResponse
import base64
import numpy
import time
from scipy.io.wavfile import read, write
import base64
import numpy as np
import json


if __name__ == "__main__":

    # url="http://wa-tts.parametrix.cn/"
    url="http://0.0.0.0:63061/"


    text = "在遥远的星系中，银河帝国经历了繁荣和倾覆，造就了一批流浪者和机器人"
    temperature = 1.0
    top_k = 15
    params = {
        "text": text,
        "top_k": top_k,
        "temperature": temperature
    }
    
    repeat = 1
    start_time = time.time()


    for i in range(repeat):
        start_time2 = time.time()
        response = requests.post(url + "tts_torch", json=params)
        # print(response.json())
        response_dict = response.json()
        audio = response_dict['audio']
        audio = base64.b64decode(audio)
        audio_array = np.frombuffer(audio, dtype=np.int16)
        output_file = "v3.wav"
        sample_rate = response_dict['sampling_rate']
        write(output_file, sample_rate, audio_array)
    ed = time.time()
    print("cost time", ed - start_time)
