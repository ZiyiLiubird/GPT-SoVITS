    


import requests
# from protocal import TTSRequest, TTSResponse
import base64
import numpy
import time
from scipy.io.wavfile import read, write
import base64
import numpy as np
import json
import ray






def compute(index):

    # url="http://wa-tts.parametrix.cn/"
    url="http://0.0.0.0:9880/"

    # text = "在遥远的, 星系中, 银河帝国, 经历了繁荣, 和倾覆，造就了一批, 流浪者, 和机器人"
    text = "这个字的读音是角(jue2)色，而不是角(jiao3)色"
    # text = ""
    # for i in range(30):
    #     text += raw_text
    data = {
        "text": text,
        "text_language": "zh"
    }

    repeat = 1
    start_time = time.time()

    for i in range(repeat):
        start_time2 = time.time()
        audio = requests.post(url, json=data)
        # response = requests.post(url + "tts_torch", json=params)
        # print(response.json())
        # response_dict = response.json()
        # audio = response_dict['audio']
        # audio = base64.b64decode(audio)
        audio_array = np.frombuffer(audio, dtype=np.int16)
        output_file = "wa.wav"
        sample_rate = 32000
        write(output_file, sample_rate, audio_array)
    ed = time.time()
    print(ed - start_time)
    return {"index": index, "cost": ed - start_time}


if __name__ == "__main__":
    # st = time.time()
    # ref = [compute.remote(i) for i in range(30)]
    # cnt = 0
    # cost = 0
    # while cnt != 30:
    #     ready_futures, pending_futures = ray.wait(ref, timeout=1)
    #     ref = pending_futures
    #     for f in ready_futures:
    #         result = ray.get(f)
    #         print(f"index: {result['index']}, cost: {result['cost']}")
    #         cost = cost + result['cost']
    #     cnt = cnt + len(ready_futures)
    #     if len(pending_futures) == 0:
    #         break
    # ed = time.time()
    # print("cost time", ed - st)
    # print(cost / 30)
    compute(0)
