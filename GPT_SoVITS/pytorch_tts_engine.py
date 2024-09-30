import os
import sys
import json
import torch
import re
import traceback
from scipy.io.wavfile import read, write

import gradio as gr
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import librosa
from feature_extractor import cnhubert
from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence, chinese
from text.cleaner import clean_text

from time import time as ttime
from module.mel_processing import spectrogram_torch
from tools.my_utils import load_audio
from tools.i18n.i18n import I18nAuto, scan_language_list
import LangSegment, os, re, sys, json
import pdb


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

language = os.environ.get("language", "Auto")
language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else language

i18n = I18nAuto(language = language)

class PyTorchTTSEngine(object):
    def __init__(self, ref_wav_path, ref_text, ref_language, how_to_cut) -> None:
        self.ref_wav_path = ref_wav_path
        self.ref_text = ref_text
        self.ref_language = ref_language

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.how_to_cut = how_to_cut
        self.language = os.environ.get("language", "Auto")
        self.language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else self.language
        
        self.i18n = I18nAuto(language = self.language)

        if (how_to_cut == self.i18n("凑四句一切")):
            self.cut_method = self.cut1
        elif (how_to_cut == self.i18n("凑50字一切")):
            self.cut_method = self.cut2
        elif (how_to_cut == self.i18n("按中文句号。切")):
            self.cut_method = self.cut3
        elif (how_to_cut == self.i18n("按英文句号.切")):
            self.cut_method = self.cut4
        elif (how_to_cut == self.i18n("按标点符号切")):
            self.cut_method = self.cut5
        else:
            self.cut_method = self.cut3

        self.pretrained_sovits_name = ["GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
                                       "GPT_SoVITS/pretrained_models/s2G488k.pth"]
        self.pretrained_gpt_name = ["GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
                                    "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"]
        self.cnhubert_base_path = os.environ.get(
            "cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base"
        )
        self.bert_path = os.environ.get(
            "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
        )
        cnhubert.cnhubert_base_path = self.cnhubert_base_path
        self.version = "v2"
        self.is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()

        self.punctuation = set(['!', '?', '…', ',', '.', '-'," "])

        if os.path.exists(f"./weight.json"):
            pass
        else:
            with open(f"./weight.json", 'w', encoding="utf-8") as file:json.dump({'GPT':{},'SoVITS':{}},file)
        
        SoVITS_weight_root=["SoVITS_weights_v2", "SoVITS_weights"]
        GPT_weight_root=["GPT_weights_v2", "GPT_weights"]
        for path in SoVITS_weight_root + GPT_weight_root:
            os.makedirs(path, exist_ok=True)

        with open(f"./weight.json", 'r', encoding="utf-8") as file:
            weight_data = file.read()
            weight_data=json.loads(weight_data)
            self.gpt_path = os.environ.get(
                "gpt_path", weight_data.get('GPT',{}).get(self.version, self.pretrained_gpt_name))
            self.sovits_path = os.environ.get(
                "sovits_path", weight_data.get('SoVITS',{}).get(self.version, self.pretrained_sovits_name))
            if isinstance(self.gpt_path, list):
                self.gpt_path = self.gpt_path[0]
            if isinstance(self.sovits_path, list):
                self.sovits_path = self.sovits_path[0]
            
            self.dict_language = {
                                self.i18n("中文"): "all_zh",#全部按中文识别
                                self.i18n("英文"): "en",#全部按英文识别#######不变
                                self.i18n("日文"): "all_ja",#全部按日文识别
                                self.i18n("粤语"): "all_yue",#全部按中文识别
                                self.i18n("韩文"): "all_ko",#全部按韩文识别
                                self.i18n("中英混合"): "zh",#按中英混合识别####不变
                                self.i18n("日英混合"): "ja",#按日英混合识别####不变
                                self.i18n("粤英混合"): "yue",#按粤英混合识别####不变
                                self.i18n("韩英混合"): "ko",#按韩英混合识别####不变
                                self.i18n("多语种混合"): "auto",#多语种启动切分识别语种
                                self.i18n("多语种混合(粤语)"): "auto_yue",#多语种启动切分识别语种
                            }
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
            self.bert_model = AutoModelForMaskedLM.from_pretrained(self.bert_path)
            if self.is_half == True:
                self.bert_model = self.bert_model.half().to(self.device)
            else:
                self.bert_model = self.bert_model.to(self.device)

            self.ssl_model = cnhubert.get_model()
            if self.is_half == True:
                self.ssl_model = self.ssl_model.half().to(self.device)
            else:
                self.ssl_model = self.ssl_model.to(self.device)
    
            self.splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }
            self.dtype = torch.float16 if self.is_half == True else torch.float32
            # self.change_model_weights()

    def get_bert_feature(self, text, word2ph):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T

    def split(self, todo_text):
        todo_text = todo_text.replace("……", "。").replace("——", "，")
        if todo_text[-1] not in self.splits:
            todo_text += "。"
        i_split_head = i_split_tail = 0
        len_text = len(todo_text)
        todo_texts = []
        while 1:
            if i_split_head >= len_text:
                break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
            if todo_text[i_split_head] in self.splits:
                i_split_head += 1
                todo_texts.append(todo_text[i_split_tail:i_split_head])
                i_split_tail = i_split_head
            else:
                i_split_head += 1
        return todo_texts

    def cut1(self, inp):
        inp = inp.strip("\n")
        inps = self.split(inp)
        split_idx = list(range(0, len(inps), 4))
        split_idx[-1] = None
        if len(split_idx) > 1:
            opts = []
            for idx in range(len(split_idx) - 1):
                opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
        else:
            opts = [inp]
        opts = [item for item in opts if not set(item).issubset(self.punctuation)]
        return "\n".join(opts)


    def cut2(self, inp):
        inp = inp.strip("\n")
        inps = self.split(inp)
        if len(inps) < 2:
            return inp
        opts = []
        summ = 0
        tmp_str = ""
        for i in range(len(inps)):
            summ += len(inps[i])
            tmp_str += inps[i]
            if summ > 50:
                summ = 0
                opts.append(tmp_str)
                tmp_str = ""
        if tmp_str != "":
            opts.append(tmp_str)
        # print(opts)
        if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
            opts[-2] = opts[-2] + opts[-1]
            opts = opts[:-1]
        opts = [item for item in opts if not set(item).issubset(self.punctuation)]
        return "\n".join(opts)

    def cut3(self, inp):
        inp = inp.strip("\n")
        opts = ["%s" % item for item in inp.strip("。").split("。")]
        opts = [item for item in opts if not set(item).issubset(self.punctuation)]
        return  "\n".join(opts)

    def cut4(self, inp):
        inp = inp.strip("\n")
        opts = ["%s" % item for item in inp.strip(".").split(".")]
        opts = [item for item in opts if not set(item).issubset(self.punctuation)]
        return "\n".join(opts)

    # contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
    def cut5(self, inp):
        inp = inp.strip("\n")
        punds = {',', '.', ';', '?', '!', '、', '，', '。', '？', '！', ';', '：', '…'}
        mergeitems = []
        items = []

        for i, char in enumerate(inp):
            if char in punds:
                if char == '.' and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                    items.append(char)
                else:
                    items.append(char)
                    mergeitems.append("".join(items))
                    items = []
            else:
                items.append(char)

        if items:
            mergeitems.append("".join(items))

        opt = [item for item in mergeitems if not set(item).issubset(punds)]
        return "\n".join(opt)

    def change_sovits_weights(self, sovits_path, prompt_language=None, text_language=None):
        
        dict_s2 = torch.load(sovits_path, map_location="cpu")
        hps = dict_s2["config"]
        hps = DictToAttrRecursive(hps)
        hps.model.semantic_frame_rate = "25hz"
        if dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
            hps.model.version = "v1"
        else:
            hps.model.version = "v2"
        version = hps.model.version
        # print("sovits版本:",hps.model.version)
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model
        )
        if ("pretrained" not in sovits_path):
            del vq_model.enc_q
        if self.is_half == True:
            vq_model = vq_model.half().to(self.device)
        else:
            vq_model = vq_model.to(self.device)
        vq_model.eval()
        print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
        with open("./weight.json")as f:
            data=f.read()
            data=json.loads(data)
            data["SoVITS"][version]=sovits_path
        with open("./weight.json","w")as f:f.write(json.dumps(data))
        self.hps = hps
        self.vq_model = vq_model
        if prompt_language is not None and text_language is not None:
            if prompt_language in list(self.dict_language.keys()):
                prompt_text_update, prompt_language_update = {'__type__':'update'},  {'__type__':'update', 'value':prompt_language}
            else:
                prompt_text_update = {'__type__':'update', 'value':''}
                prompt_language_update = {'__type__':'update', 'value':self.i18n("中文")}
            if text_language in list(self.dict_language.keys()):
                text_update, text_language_update = {'__type__':'update'}, {'__type__':'update', 'value':text_language}
            else:
                text_update = {'__type__':'update', 'value':''}
                text_language_update = {'__type__':'update', 'value':self.i18n("中文")}
            return  {'__type__':'update', 'choices':list(self.dict_language.keys())}, {'__type__':'update', 'choices':list(self.dict_language.keys())}, prompt_text_update, prompt_language_update, text_update, text_language_update

    def change_gpt_weights(self, gpt_path):
        self.hz = 50
        dict_s1 = torch.load(gpt_path, map_location="cpu")
        self.config = dict_s1["config"]
        self.max_sec = self.config["data"]["max_sec"]
        t2s_model = Text2SemanticLightningModule(self.config, "****", is_train=False)
        t2s_model.load_state_dict(dict_s1["weight"])
        if self.is_half == True:
            t2s_model = t2s_model.half()
        t2s_model = t2s_model.to(self.device)
        t2s_model.eval()
        self.t2s_model = t2s_model
        total = sum([param.nelement() for param in t2s_model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))
        with open("./weight.json")as f:
            data=f.read()
            data=json.loads(data)
            data["GPT"][self.version]=gpt_path
        with open("./weight.json","w")as f: f.write(json.dumps(data))

    def change_model_weights(self, gpt_path, sovits_path, prompt_language=None, text_language=None):
        self.change_sovits_weights(sovits_path=sovits_path,
                                   prompt_language=prompt_language,
                                   text_language=text_language)
        self.change_gpt_weights(gpt_path=gpt_path)
    
    def get_spepc(self, hps, filename):
        audio = load_audio(filename, int(hps.data.sampling_rate))
        audio = torch.FloatTensor(audio)
        maxx = audio.abs().max()
        if(maxx>1): audio/=min(2,maxx)
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        )
        return spec

    def clean_text_inf(self, text, language, version):
        phones, word2ph, norm_text = clean_text(text, language, version)
        phones = cleaned_text_to_sequence(phones, version)
        return phones, word2ph, norm_text

    def get_bert_inf(self, phones, word2ph, norm_text, language):
        language=language.replace("all_","")
        if language == "zh":
            bert = self.get_bert_feature(norm_text, word2ph).to(self.device)#.to(dtype)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if self.is_half == True else torch.float32,
            ).to(self.device)

        return bert

    def get_first(self, text):
        pattern = "[" + "".join(re.escape(sep) for sep in self.splits) + "]"
        text = re.split(pattern, text)[0].strip()
        return text

    def get_phones_and_bert(self, text, language, version, final=False):
        if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
            language = language.replace("all_","")
            if language == "en":
                LangSegment.setfilters(["en"])
                formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
            else:
                # 因无法区别中日韩文汉字,以用户输入为准
                formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
            if language == "zh":
                if re.search(r'[A-Za-z]', formattext):
                    formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                    formattext = chinese.mix_text_normalize(formattext)
                    return self.get_phones_and_bert(formattext,"zh",version)
                else:
                    phones, word2ph, norm_text = self.clean_text_inf(formattext, language, version)
                    bert = self.get_bert_feature(norm_text, word2ph).to(self.device)
            elif language == "yue" and re.search(r'[A-Za-z]', formattext):
                    formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                    formattext = chinese.mix_text_normalize(formattext)
                    return self.get_phones_and_bert(formattext,"yue",version)
            else:
                phones, word2ph, norm_text = self.clean_text_inf(formattext, language, version)
                bert = torch.zeros(
                    (1024, len(phones)),
                    dtype=torch.float16 if self.is_half == True else torch.float32,
                ).to(self.device)
        elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
            textlist=[]
            langlist=[]
            LangSegment.setfilters(["zh","ja","en","ko"])
            if language == "auto":
                for tmp in LangSegment.getTexts(text):
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            elif language == "auto_yue":
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "zh":
                        tmp["lang"] = "yue"
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            else:
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "en":
                        langlist.append(tmp["lang"])
                    else:
                        # 因无法区别中日韩文汉字,以用户输入为准
                        langlist.append(language)
                    textlist.append(tmp["text"])
            phones_list = []
            bert_list = []
            norm_text_list = []
            for i in range(len(textlist)):
                lang = langlist[i]
                phones, word2ph, norm_text = self.clean_text_inf(textlist[i], lang, version)
                bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)
            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = ''.join(norm_text_list)

        if not final and len(phones) < 6:
            return self.get_phones_and_bert("." + text,language,version,final=True)

        return phones,bert.to(self.dtype),norm_text

    def merge_short_text_in_array(self, texts, threshold):
        if (len(texts)) < 2:
            return texts
        result = []
        text = ""
        for ele in texts:
            text += ele
            if len(text) >= threshold:
                result.append(text)
                text = ""
        if (len(text) > 0):
            if len(result) == 0:
                result.append(text)
            else:
                result[len(result) - 1] += text
        return result

    def process_ref_audio(self,):
        ref_wav_path, prompt_text, prompt_language = self.ref_wav_path, self.ref_text, self.ref_language
        prompt_language = self.dict_language[prompt_language]
        prompt_text = prompt_text.strip("\n")
        if (prompt_text[-1] not in self.splits): prompt_text += "。" if prompt_language != "en" else "."
        zero_wav = np.zeros(
            int(self.hps.data.sampling_rate * 0.3),
            dtype=np.float16 if self.is_half == True else np.float32,
        )
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
                raise OSError(self.i18n("参考音频在3~10秒范围外，请更换！"))
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            if self.is_half == True:
                wav16k = wav16k.half().to(self.device)
                zero_wav_torch = zero_wav_torch.half().to(self.device)
            else:
                wav16k = wav16k.to(self.device)
                zero_wav_torch = zero_wav_torch.to(self.device)
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))[
                "last_hidden_state"
            ].transpose(
                1, 2
            )  # .float()
            codes = self.vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            self.ref_prompt = prompt_semantic.unsqueeze(0).to(self.device)

        self.ref_phones1, self.ref_bert1, self.ref_norm_text1 = self.get_phones_and_bert(prompt_text, prompt_language, self.version)

    def process_text(self, texts):
        _text=[]
        if all(text in [None, " ", "\n",""] for text in texts):
            raise ValueError(self.i18n("请输入有效文本"))
        for text in texts:
            if text in  [None, " ", ""]:
                pass
            else:
                _text.append(text)
        return _text
    
    # def free_up_memory(self):
    # # Prior inference run might have large variables not cleaned up due to exception during the run.
    # # Free up as much memory as possible to allow this run to be successful.

    #     if np.random.rand() > 0.98:
    #         gc.collect()
    #         if torch.cuda.is_available():
    #             torch.cuda.empty_cache()

    def get_tts_wav(self,
                    text,
                    text_language,
                    top_k=20,
                    top_p=0.6,
                    temperature=0.6,
                    speed=1,
                    if_freeze=False,
                    inp_refs=None):
        text_language = self.dict_language[text_language]
        text = text.strip("\n")
        text = self.cut_method(text)
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        texts = text.split("\n")
        texts = self.process_text(texts)
        texts = self.merge_short_text_in_array(texts, 5)
        audio_opt = []
        t = []
        zero_wav = np.zeros(
            int(self.hps.data.sampling_rate * 0.3),
            dtype=np.float16 if self.is_half == True else np.float32,
        )

        for i_text, text in enumerate(texts):
            # 解决输入目标文本的空行导致报错的问题
            if (len(text.strip()) == 0):
                continue
            if (text[-1] not in self.splits): text += "。" if text_language != "en" else "."
            phones2, bert2, norm_text2 = self.get_phones_and_bert(text, text_language, self.version)

            bert = torch.cat([self.ref_bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(self.ref_phones1+phones2).to(self.device).unsqueeze(0)

            bert = bert.to(self.device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)

            t2 = ttime()
            # cache_key="%s-%s-%s-%s-%s-%s-%s-%s"%(ref_wav_path,prompt_text,prompt_language,text,text_language,top_k,top_p,temperature)
            # print(cache.keys(),if_freeze)
            # if(i_text in cache and if_freeze==True):pred_semantic=cache[i_text]
            # else:
            with torch.no_grad():
                pred_semantic, idx = self.t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    self.ref_prompt,
                    bert,
                    # prompt_phone_len=ph_offset,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=self.hz * self.max_sec,
                )
                pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                # cache[i_text]=pred_semantic
            t3 = ttime()
            refers=[]
            if(inp_refs):
                for path in inp_refs:
                    try:
                        refer = self.get_spepc(self.hps, path.name).to(self.dtype).to(self.device)
                        refers.append(refer)
                    except:
                        traceback.print_exc()
            if(len(refers)==0):refers = [self.get_spepc(self.hps, self.ref_wav_path).to(self.dtype).to(self.device)]
            audio = (self.vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(self.device).unsqueeze(0), refers,speed=speed).detach().cpu().numpy()[0, 0])
            max_audio=np.abs(audio).max()#简单防止16bit爆音
            if max_audio>1:audio/=max_audio
            audio_opt.append(audio)
            audio_opt.append(zero_wav)
            # t4 = ttime()
            # t.extend([t2 - t1,t3 - t2, t4 - t3])
            # t1 = ttime()

        return self.hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768 ).astype(
            np.int16
        )


if __name__ == "__main__":
    # ref_wav_path = "/data1/ziyiliu/tts/GPT-SoVITS/logs/Zhongli/raw/vo_card_zhongli_invite_easy_01.wav"
    # prompt_text = "这种卡牌的玩法…唔，应该与弈棋之法有共通之处。"

    ref_wav_path = "/data1/ziyiliu/tts/GPT-SoVITS/logs/Ningguang/raw/vo_dialog_DLEQ001_ningguang_01.wav"
    prompt_text = "北斗正在孤云阁那边帮我打捞散落的群玉阁藏品。你们若有兴趣，可以去看看。"
    prompt_language = i18n("中文")
    how_to_cut = i18n("按中文句号。切")
    infer_engine = PyTorchTTSEngine(ref_wav_path=ref_wav_path,
                                    ref_text=prompt_text,
                                    ref_language=prompt_language,
                                    how_to_cut=how_to_cut)
    
    # text = "若你困于无风之地，我便为你奏响高天之歌。以千年的流风，指引你前进的方向。当你重新踏上旅途之后，一定要记得旅途本身的意义。"
    text_language = i18n("中文")
    how_to_cut = i18n("按中文句号。切")
    top_k = 15
    temperature = 1.0

    sovits_path = "/data1/ziyiliu/tts/GPT-SoVITS/SoVITS_weights_v2/Ningguang_e15_s630.pth"
    gpt_path = "/data1/ziyiliu/tts/GPT-SoVITS/GPT_weights_v2/Ningguang-e10.ckpt"

    infer_engine.change_sovits_weights(sovits_path=sovits_path, prompt_language=prompt_language, text_language=text_language)
    infer_engine.change_gpt_weights(gpt_path=gpt_path)

    infer_engine.process_ref_audio()
    test_text_list = [
        "若你困于无风之地，我便为你奏响高天之歌。以千年的流风，指引你前进的方向。当你重新踏上旅途之后，一定要记得旅途本身的意义。",
        "我在明星斋的时候说，决定砸下群玉阁的时候，我已经算好了这笔交易的得与失。",
        "这两位，还是别相提并论为好。北斗每次回到璃月港，都只会让我头疼而已。",
        "如果早上没赚到钱，中午就会没饭吃。我对摩拉的执着，大概也是那时候形成的吧。",
        "我是七星之「天权」，凝光。要做个交易吗？你来做我的贴身侍卫，而我，会教导你在璃月出人头地的技巧。",
        "真想在复杂的商场中获利，还要看他们自己的见识与格局，我只是分享一些我的理解而已。",
        "那时候他们一定已经做好准备，有资格成为卓越的商人了。而在那之前，我会保护他们的。",
        "她刚刚说这些话的意思，大概也是想把她心中的那份热情传达给你。",
        "要是每天醒来，在饮食上只能感受到重复，那还没开始做事，疲劳就已经积累起来了。",
        "船越重，船体下沉就会越厉害，那么只要掌握船体浸入水的深度，就能估计船上的负载。"
    ]
    st = ttime()
    for i, text in enumerate(test_text_list):
        sample_rate, audio_output = infer_engine.get_tts_wav(text=text,
                                                            text_language=text_language,
                                                            top_k=top_k,
                                                            top_p=1,
                                                            temperature=temperature,
                                                            )

        audio = audio_output
        output_file = f"ningguang_infer_{i}.wav"
        # sample_rate = 44100
        write(output_file, sample_rate, audio)
    ed = ttime()
    print(f"cost {ed - st}s")
    print(f"average time: {(ed - st) / len(test_text_list)}s")