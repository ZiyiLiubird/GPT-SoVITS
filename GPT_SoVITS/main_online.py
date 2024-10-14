import os, sys
from fastapi import FastAPI
from typing import Optional, List, Union
import base64
import time
from time import time as ttime
from scipy.io.wavfile import read, write
from pydantic import BaseModel
import json


from tools.i18n.i18n import I18nAuto, scan_language_list


class TTSResponse(BaseModel):  # 定义一个类用作返回值
    #现在没有使用，因为audio太大会导致转pydantic速度太慢
    audio: str 
    sampling_rate: int

class TTSRequest(BaseModel):  # 定义一个类用作参数
    text: str
    top_k: int
    temperature: float

language = os.environ.get("language", "Auto")
language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else language

i18n = I18nAuto(language = language)

app = FastAPI()

from pytorch_tts_engine import PyTorchTTSEngine

ref_wav_path = "/data1/ziyiliu/tts/GPT-SoVITS/logs/Ningguang/raw/vo_dialog_DLEQ001_ningguang_01.wav"
prompt_text = "北斗正在孤云阁那边帮我打捞散落的群玉阁藏品。你们若有兴趣，可以去看看。"
prompt_language = i18n("中文")
how_to_cut = i18n("按中文句号。切")

torch_engine = PyTorchTTSEngine(ref_wav_path=ref_wav_path,
                                ref_text=prompt_text,
                                ref_language=prompt_language,
                                how_to_cut=how_to_cut)

text_language = i18n("中文")
how_to_cut = i18n("按中文句号。切")
top_k = 15
temperature = 1.0

sovits_path = "/data1/ziyiliu/tts/GPT-SoVITS/SoVITS_weights_v2/Ningguang_e15_s630.pth"
gpt_path = "/data1/ziyiliu/tts/GPT-SoVITS/GPT_weights_v2/Ningguang-e10.ckpt"

torch_engine.change_sovits_weights(sovits_path=sovits_path, prompt_language=prompt_language, text_language=text_language)
torch_engine.change_gpt_weights(gpt_path=gpt_path)

torch_engine.process_ref_audio()




@app.get("/")
async def read_root():
    return {"name": "GPT_SoVITS-Serving"}


@app.post("/tts_torch")
async def tts_torch(param: TTSRequest):
    #print("DEBUG tts_torch post",type(param))
    start_time = time.time()
    text = param.text
    temperature = param.temperature
    top_k = param.top_k
    top_p = 1

    print("INIT FILE TIME",time.time()-start_time)
    # res =  torch_engine.tts_fn(text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, language, reference_audio, emotion, prompt_mode, style_text, style_weight)

    sample_rate, audio_output = torch_engine.get_tts_wav(text=text, text_language=text_language,
                                                         top_k=top_k,
                                                         top_p=top_p,
                                                         temperature=temperature)

    print("TTS_FN TORCH INFER",time.time()-start_time)
    
    # res_str = base64.b64encode(res[1][1].tostring())
    res_str = base64.b64encode(audio_output.tostring())
    response = TTSResponse(audio=res_str, sampling_rate=sample_rate)
    #print("DEBUG TTSResponse")
    print("TTS TORCH cost time ",time.time()-start_time)
    print("*"*100)
    return response



