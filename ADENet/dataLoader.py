import os, torch, numpy, cv2, random, glob, python_speech_features, librosa
from scipy.io import wavfile
from torchvision.transforms import RandomCrop
import numpy as np
from random import choice

MAX_INT16 = np.iinfo(np.int16).max

def generate_audio_set(dataPath, batchList):
    audioSet = {}
    for line in batchList:
        data = line.split('\t')
        videoName = data[0][:11]
        dataName = data[0]
        _, audio = wavfile.read(os.path.join(dataPath, videoName, dataName + '.wav'))
        audioSet[dataName] = audio
    return audioSet

def overlap(dataName, audio, audioSet):   
    #noiseName =  random.sample(set(list(audioSet.keys())) - {dataName}, 1)[0]

    if len(set(list(audioSet.keys())) - {dataName}) > 0:
        noiseName =  random.sample(set(list(audioSet.keys())) - {dataName}, 1)[0]
    else:
        noiseName = random.sample(set(list(audioSet.keys())), 1)[0]
        
    noiseAudio = audioSet[noiseName]    
    snr = [random.uniform(-5, 5)]
    if len(noiseAudio) < len(audio):
        shortage = len(audio) - len(noiseAudio)
        noiseAudio = numpy.pad(noiseAudio, (0, shortage), 'wrap')
    else:
        noiseAudio = noiseAudio[:len(audio)]
    noiseDB = 10 * numpy.log10(numpy.mean(abs(noiseAudio ** 2)) + 1e-4)
    cleanDB = 10 * numpy.log10(numpy.mean(abs(audio ** 2)) + 1e-4)
    noiseAudio = numpy.sqrt(10 ** ((cleanDB - noiseDB - snr) / 10)) * noiseAudio
    audio = audio + noiseAudio    

    return audio.astype(numpy.int16)
    
def add_noise_for_waveform(s, n, db):
    """
    为语音文件叠加噪声
    ----
    para:
        输入的语音均都是经过标准话读入的  比如
        s：原语音的时域信号
        n：噪声的时域信号
        db：信噪比
    ----
    return:
        叠加噪声后的语音
    """
    alpha = np.sqrt(
        np.sum(s ** 2) / (np.sum(n ** 2) * 10 ** (db / 10))
    )
    mix = s + alpha * n
    return mix
  
def overlap_speech(label_audio, noise, db):   
    if label_audio.shape[0] < noise.shape[0]:
        noise = noise[0:label_audio.shape[0]]
    else:
        shortage =  label_audio.shape[0] - noise.shape[0]
        noise = numpy.pad(noise, (( shortage), (0)), 'wrap')

    return add_noise_for_waveform(label_audio, noise, db)
    

def load_audio(data, dataPath, numFrames, audioAug, audioSet = None, noise_path = None, db=None ):
    dataName = data[0]
    fps = float(data[2])    
    audio = audioSet[dataName]    # wavfile.read 数据都是大数 
    if audioAug == True:
        augType = random.randint(0,1)
        if augType == 1:
            audio = overlap(dataName, audio, audioSet)
        else:
            audio = audio
    
    label_audio = audio/(MAX_INT16+1)  # SE 任务的标签
    
    noise = librosa.load(noise_path, sr = 16000)

    mix_audio = overlap_speech(label_audio, noise[0], db)  #混合噪音音频，作为SE任务的输入

    audio = mix_audio * (MAX_INT16+1) # 混合了噪音后的audio
    '''
    ### wavfile.read与librosa.load 的关系
    _, audio = wavfile.read(path)
    noise = librosa.load(path, sr = 16000)
    audio_gap = audio - noise[0]*(MAX_INT16+1)
    np.max(audio_gap) = 0
    '''
    maxAudio = int(numFrames*640)   # 固定音频长度，使得一个batch_size中的所有数据长度相同
    if mix_audio.shape[0] < maxAudio:
        shortage    = maxAudio - mix_audio.shape[0]
        mix_audio     = numpy.pad(mix_audio, (( shortage), (0)), 'wrap')
        label_audio = numpy.pad(label_audio,  (( shortage), (0)), 'wrap')
    mix_audio = mix_audio[:int(round(maxAudio))]  
    label_audio = label_audio[:int(round(maxAudio))]  


    
    # fps is not always 25, in order to align the visual, we modify the window and step in MFCC extraction process based on fps
    audio = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025 * 25 / fps, winstep = 0.010 * 25 / fps)
    maxAudio = int(numFrames * 4)
    if audio.shape[0] < maxAudio:
        shortage    = maxAudio - audio.shape[0]
        audio     = numpy.pad(audio, ((0, shortage), (0,0)), 'wrap')
    audioFeature = audio[:int(round(numFrames * 4)),:]  
    return audioFeature , mix_audio,  label_audio



def load_visual(data, dataPath, numFrames, visualAug): 
    dataName = data[0]
    videoName = data[0][:11]
    faceFolderPath = os.path.join(dataPath, videoName, dataName)
    faceFiles = glob.glob("%s/*.jpg"%faceFolderPath)
    sortedFaceFiles = sorted(faceFiles, key=lambda data: (float(data.split('/')[-1][:-4])), reverse=False) 
    faces = []
    H = 112
    if visualAug == True:
        new = int(H*random.uniform(0.7, 1))
        x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
        M = cv2.getRotationMatrix2D((H/2,H/2), random.uniform(-15, 15), 1)
        augType = random.choice(['orig', 'flip', 'crop', 'rotate']) 
    else:
        augType = 'orig'
    for faceFile in sortedFaceFiles[:numFrames]:
        face = cv2.imread(faceFile)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (H,H))
        if augType == 'orig':
            faces.append(face)
        elif augType == 'flip':
            faces.append(cv2.flip(face, 1))
        elif augType == 'crop':
            faces.append(cv2.resize(face[y:y+new, x:x+new] , (H,H))) 
        elif augType == 'rotate':
            faces.append(cv2.warpAffine(face, M, (H,H)))
    faces = numpy.array(faces)
    return faces


def load_label(data, numFrames):
    res = []
    labels = data[3].replace('[', '').replace(']', '')
    labels = labels.split(',')
    for label in labels:
        res.append(int(label))
    res = numpy.array(res[:numFrames])
    return res

class train_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, batchSize, noise_db=None,musanPath=None,  **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = []      
        mixLst = open(trialFileName).read().splitlines()
        # sort the training set by the length of the videos, shuffle them to make more videos in the same batch belong to different movies
        sortedMixLst = sorted(mixLst, key=lambda data: (int(data.split('\t')[1]), int(data.split('\t')[-1])), reverse=True)         
        start = 0        
        while True:
            length = int(sortedMixLst[start].split('\t')[1])
            end = min(len(sortedMixLst), start + max(int(batchSize / length), 1))
            self.miniBatch.append(sortedMixLst[start:end])
            if end == len(sortedMixLst):
                break
            start = end     
        self.noise_list = open(musanPath).read().splitlines()

        self.noise_db = noise_db

    def __getitem__(self, index):
        batchList    = self.miniBatch[index]
        numFrames   = int(batchList[-1].split('\t')[1])
        audioFeatures, visualFeatures, labels = [], [], []
        label_audios, mix_audios = [], []
        audioSet = generate_audio_set(self.audioPath, batchList) # load the audios in this batch to do augmentation
        for line in batchList:
            data = line.split('\t')        
            noise_path = choice(self.noise_list)
            audioFeature , mix_audio,  label_audio = load_audio(data, self.audioPath, numFrames, audioAug = True, audioSet = audioSet, noise_path = noise_path, db = self.noise_db)
            audioFeatures.append(audioFeature)  
            label_audios.append(label_audio)
            mix_audios.append(mix_audio)

            visualFeatures.append(load_visual(data, self.visualPath,numFrames, visualAug = True))
            labels.append(load_label(data, numFrames))
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels)) ,\
                torch.FloatTensor(numpy.array(mix_audios)), \
                torch.FloatTensor(numpy.array(label_audios))  

    def __len__(self):
        return len(self.miniBatch)


class val_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, noise_db=None,musanPath=None,  **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = open(trialFileName).read().splitlines()
        self.noise_list = open(musanPath).read().splitlines()

        self.noise_db = noise_db

    def __getitem__(self, index):
        line       = [self.miniBatch[index]]
        numFrames  = int(line[0].split('\t')[1])
        audioSet   = generate_audio_set(self.audioPath, line)        
        data = line[0].split('\t')
        noise_path = choice(self.noise_list)
        audioFeature , mix_audio,  label_audio = load_audio(data, self.audioPath, numFrames, audioAug = False, audioSet = audioSet, noise_path = noise_path, db = self.noise_db)
        audioFeatures = [audioFeature]
        label_audios = [label_audio]
        mix_audios = [mix_audio]

        visualFeatures = [load_visual(data, self.visualPath,numFrames, visualAug = False)]
        labels = [load_label(data, numFrames)]         
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels)) ,\
                torch.FloatTensor(numpy.array(mix_audios)), \
                torch.FloatTensor(numpy.array(label_audios))  

    def __len__(self):
        return len(self.miniBatch)
