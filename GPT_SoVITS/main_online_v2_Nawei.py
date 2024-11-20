import os, sys
from os.path import dirname

from fastapi import FastAPI
from typing import Optional, List, Union
import base64
import time
from time import time as ttime
from scipy.io.wavfile import read, write
from pydantic import BaseModel
import json
sys.path.append(dirname(dirname(__file__)))


from tools.i18n.i18n import I18nAuto, scan_language_list


class TTSResponse(BaseModel):  # 定义一个类用作返回值
    #现在没有使用，因为audio太大会导致转pydantic速度太慢
    audio: str 
    sampling_rate: int

class TTSRequest(BaseModel):  # 定义一个类用作参数
    text: str
    speaker: str = ""
    speed_factor: float = 1.0
    sdp_ratio: float = 0.5
    noise_scale: float = 0.6
    noise_scale_w: float = 0.9
    language: str = "ZH"
    length_scale: float = 1.0
    top_k: int = 15
    temperature: float = 1.0
    emotion: str = "Happy"

language = os.environ.get("language", "Auto")
language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else language

i18n = I18nAuto(language = language)

app = FastAPI()

from pytorch_tts_engine_v2 import PyTorchTTSEngine

ref_wav_path = "/data1/ziyiliu/tts/GPT-SoVITS/refs/Nawei/vo_ARLQ003_4_neuvillette_09.wav"
prompt_text = "真是滴水不漏的发言。好吧，我原则上同意你的提案。"

# ref_wav_path_dict = {
#     "Happy": "/data1/ziyiliu/tts/GPT-SoVITS/refs/Xiangling/vo_card_xiangling_freetalk_01.wav",
#     "Sad": "/data1/ziyiliu/tts/GPT-SoVITS/refs/Xiangling/vo_card_xiangling_endOfGame_fail_01.wav"
# }
# ref_wav_prompt_path_dict = {
#     "Happy": "哇，这么多人都在这里玩牌…他们的肚子会不会饿了呀？",
#     "Sad": "你打牌的技术是找谁学的呀，也教教我嘛。"
# }

prompt_language = i18n("中文")
how_to_cut = i18n("按中文句号。切")

torch_engine = PyTorchTTSEngine(ref_wav_path=ref_wav_path,
                                ref_text=prompt_text,
                                ref_language=prompt_language,
                                how_to_cut=how_to_cut)

text_language = i18n("中文")
top_k = 15
temperature = 1.0

sovits_path = "/data1/ziyiliu/tts/GPT-SoVITS/SoVITS_weights_v2/Nawei_e16_s848.pth"
gpt_path = "/data1/ziyiliu/tts/GPT-SoVITS/GPT_weights_v2/Nawei-e20.ckpt"

torch_engine.change_sovits_weights(sovits_path=sovits_path, prompt_language=prompt_language, text_language=text_language)
torch_engine.change_gpt_weights(gpt_path=gpt_path)

# torch_engine.process_ref_audio()




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
    emotion = param.emotion
    speed_factor = param.speed_factor

    print("INIT FILE TIME",time.time()-start_time)
    # res =  torch_engine.tts_fn(text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, language, reference_audio, emotion, prompt_mode, style_text, style_weight)
    item = torch_engine.inference(text=text,
                                text_lang=text_language,
                                ref_audio_path=ref_wav_path,
                                prompt_text=prompt_text,
                                prompt_lang=prompt_language,
                                top_k=top_k,
                                top_p=1,
                                speed_factor=speed_factor,
                                batch_size=60,
                                temperature=temperature,
                                )
    item, seed = next(item)
    sample_rate, audio_output = item
    # sample_rate, audio_output = torch_engine.get_tts_wav(text=text, text_language=text_language,
    #                                                      top_k=top_k,
    #                                                      top_p=top_p,
    #                                                      temperature=temperature)

    print("TTS_FN TORCH INFER",time.time()-start_time)
    
    # res_str = base64.b64encode(res[1][1].tostring())
    res_str = base64.b64encode(audio_output.tostring())
    response = TTSResponse(audio=res_str, sampling_rate=sample_rate)
    #print("DEBUG TTSResponse")
    print("TTS TORCH cost time ",time.time()-start_time)
    print("*"*100)
    return response



