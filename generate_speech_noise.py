from glob import glob
import csv
from random import random , shuffle
from numpy.lib.npyio import save
from scipy.io import wavfile
from tqdm import tqdm


def clip_noise():
    noise_path1 = "/mnt/data/xjw/data/musan/speech/librivox/"
    
    noise_list1 = glob(noise_path1 + '*.wav')

    noise_list = noise_list1 
    print(len(noise_list))

    save_path = "/mnt/data/xjw/data/musan/clip_noise_speech/"
    count = 0
    
    for line in tqdm(noise_list):
        index = 0
        wav_name = line.split('/')[-1]
        wav_name = wav_name.split('.')[0]

        sr, wav_data = wavfile.read(line)
        if wav_data.shape[0] < sr:
            continue

        begin = 0
        end = sr*10

        if end > wav_data.shape[0]:
            wavfile.write(save_path + wav_name + '-' + str(index) + '.wav', sr, wav_data)
        else:
            while begin < wav_data.shape[0]:
                tmp_wav = wav_data[begin : end]
                wavfile.write(save_path + wav_name + '-' + str(index) + '.wav', sr, tmp_wav)
                index += 1 
                begin = end
                end += sr*10

def split_dataset():
    data_path = "/mnt/data/xjw/data/musan/clip_noise_speech/"
    data_list = glob(data_path + '*.wav')

    print(len(data_list))

    shuffle(data_list)
    count = 0
    f_train = open('./noise_dataset/noise_speech_txt/train_list.txt', 'w')
    f_val = open('./noise_dataset/noise_speech_txt/val_list.txt', 'w')
    f_test = open('./noise_dataset/noise_speech_txt/test_list.txt', 'w')
    
    for line in data_list:
        if count < int(len(data_list)*0.7):
            f_train.writelines(line + '\n')
        elif count < int(len(data_list)*0.9):
            f_val.writelines(line + '\n')
        else:
            f_test.writelines(line + '\n')
        
        count += 1


if __name__ == '__main__':
    clip_noise()
    split_dataset()