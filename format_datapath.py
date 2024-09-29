

ori_esd_path = "/data1/ziyiliu/tts/GPT-SoVITS/logs/Ningguang/esd.list"

new_esd_path = "/data1/ziyiliu/tts/GPT-SoVITS/logs/Ningguang/gpt_esd.list"

with open(ori_esd_path, 'r') as f:
    lines = f.readlines()
    with open(new_esd_path, 'w') as f2:
        for line in lines:
            new_line = line.split('wavs/')[1]
            f2.write(new_line)

