import os
import wave
import shutil
import random

def get_wav_file_size(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        # 获取文件大小（以字节为单位）
        file_size = os.path.getsize(file_path)
        return file_size


def filter_file_by_size(source_directory_list, destination_directory, metadata, random_meta_data, language='EN'):
    for source_directory in source_directory_list:
        wav_files = [f for f in os.listdir(source_directory) if f.endswith('.wav')]
        if not os.path.exists(destination_directory):
                os.makedirs(destination_directory)
        min_size = 500
        speaker_name = source_directory.split('/')[-1]

        filtered_files = [f for f in wav_files if get_wav_file_size(os.path.join(source_directory, f)) / 1024 >= min_size]
        print(f"Found {len(filtered_files)} WAV files with size greater than or equal to {min_size} KB.")
        
        for f in filtered_files:
            source_file = os.path.join(source_directory, f)
            destination_file = os.path.join(destination_directory, f)
            shutil.copy(source_file, destination_file)
        
        with open(metadata, 'a+') as metadata_file:
            for file in filtered_files:
                text_file = file.replace('.wav', '.lab')
                if os.path.exists(os.path.join(source_directory, text_file)):
                    data_piece = str(file)
                    if language == 'EN':
                        data_piece += f"|{speaker_name}|EN|"
                    else:
                        data_piece += f"|{speaker_name}|ZH|"
                    with open(os.path.join(source_directory, text_file), "r") as text_file:
                        data_piece += text_file.read().strip()
                    metadata_file.write(data_piece + "\n")

    with open(metadata, "r") as metadata_file:
        dataset =[line for line in metadata_file]
    
    len(dataset)
    random.shuffle(dataset)
    with open(metadata, "w") as metadata_file:
        metadata_file.writelines(dataset)


def merge_dataset():
    dataset_path_1 = ["/data1/ziyiliu/datasets/tts/chinese/Ningguang"]
    dataset_path_2 = ["/data1/ziyiliu/datasets/tts/chinese/Leidian"]
    destination_directory = "/data1/ziyiliu/datasets/tts/chinese/NingguangMergeLeidian"

    for source_directory in dataset_path_1:
        wav_files = [f for f in os.listdir(source_directory) if f.endswith('.wav')]
        if not os.path.exists(destination_directory):
                os.makedirs(destination_directory)
        min_size = 500

        filtered_files = [f for f in wav_files if get_wav_file_size(os.path.join(source_directory, f)) / 1024 >= min_size]
        filtered_texts = []
        for f in filtered_files:
            text_file = f.replace('.wav', '.lab')
            if os.path.exists(os.path.join(source_directory, text_file)):
                filtered_texts.append(text_file)
        
        print(f"Found {len(filtered_files)} WAV files with size greater than or equal to {min_size} KB.")
        assert len(filtered_files) == len(filtered_texts), "files and texts do not match."
        main_file_length = len(filtered_files)

        for f in filtered_files:
            source_file = os.path.join(source_directory, f)
            destination_file = os.path.join(destination_directory, f)
            shutil.copy(source_file, destination_file)

        for f in filtered_texts:
            source_file = os.path.join(source_directory, f)
            destination_file = os.path.join(destination_directory, f)
            shutil.copy(source_file, destination_file)


    for source_directory in dataset_path_2:
        wav_files = [f for f in os.listdir(source_directory) if f.endswith('.wav')]
        if not os.path.exists(destination_directory):
                os.makedirs(destination_directory)
        min_size = 500

        filtered_files = [f for f in wav_files if get_wav_file_size(os.path.join(source_directory, f)) / 1024 >= min_size]
        
        filtered_files = filtered_files[-int(main_file_length * 0.2):]

        filtered_texts = []
        for f in filtered_files:
            text_file = f.replace('.wav', '.lab')
            if os.path.exists(os.path.join(source_directory, text_file)):
                filtered_texts.append(text_file)

        print(f"Found {len(filtered_files)} WAV files with size greater than or equal to {min_size} KB.")
        assert len(filtered_files) == len(filtered_texts), "files and texts do not match."

        for f in filtered_files:
            source_file = os.path.join(source_directory, f)
            destination_file = os.path.join(destination_directory, f)
            shutil.copy(source_file, destination_file)

        for f in filtered_texts:
            source_file = os.path.join(source_directory, f)
            destination_file = os.path.join(destination_directory, f)
            shutil.copy(source_file, destination_file)


if __name__ == '__main__':
    directory = ["/data1/ziyiliu/datasets/tts/chinese/NingguangMergeLeidian"]

    metadata = "/data1/ziyiliu/tts/Bert-VITS2/data/NingguangMergeLeidian/esd.list"
    random_metadata = "/data1/ziyiliu/tts/Bert-VITS2/data/NingguangMergeLeidian/random_esd.list"
    data_save_path = "/data1/ziyiliu/tts/Bert-VITS2/data/NingguangMergeLeidian/raw"
    filtered_files = filter_file_by_size(directory, data_save_path, metadata, random_metadata, language='ZH')

    # merge_dataset()