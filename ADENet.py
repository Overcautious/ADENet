import torch, math
import torch.nn as nn
import torch.nn.functional as F

import sys, time, numpy, os, subprocess, pandas, tqdm
from SI_SNR import SI_SNR, cal_SISNR
from loss import lossAV, lossA, lossV, loss_aud, L2Loss
from model.ADENetModel import ADENetModel
from pystoi import stoi
from pypesq import pesq
from sklearn.metrics import roc_auc_score

class ADENet(nn.Module):
    def __init__(self, lr = 0.0001, lrDecay = 0.95, isDown = True , isUp = True, **kwargs):
        super(ADENet, self).__init__()        
        self.model = ADENetModel(isDown, isUp)
        self.model = self.model.cuda()

        self.lossAV = lossAV().cuda()
        self.lossA = lossA().cuda()
        self.lossV = lossV().cuda()
        self.lossSE = loss_aud().cuda()
        self.lossSim = L2Loss().cuda()
        self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)

        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1024 / 1024))

    def train_network(self, loader, epoch, **kwargs):
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        nloss = 0
        lr = self.optim.param_groups[0]['lr']        
        for num, (audioFeature, visualFeature, labels, audio, label_audios) in enumerate(loader, start=1):
            self.zero_grad()
            outsAV, outs_speech, outsA, outsV = self.model.forward(audioFeature[0].cuda(), visualFeature[0].cuda() ,audio[0].cuda() )

            labels = labels[0].reshape((-1)).cuda()
            nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels)
            nlossA = self.lossA.forward(outsA, labels)
            nlossV = self.lossV.forward(outsV, labels)

            nse_loss = self.lossSE.forward(outs_speech[0], label_audios[0].cuda())  

            nloss = nse_loss + nlossAV + 0.4 * nlossA + 0.4 * nlossV 

            loss += nloss.detach().cpu().numpy()
            top1 += prec
 
            nloss.backward()
            self.optim.step()
            index += len(labels)
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
            " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), 100 * (top1/index)))
            sys.stderr.flush()

        sys.stdout.write("\n")
        del nloss, index, top1
        torch.cuda.empty_cache()
        return loss/num, lr

    def evaluate_network(self, loader, evalCsvSave, evalOrig, **kwargs):
        self.eval()
        predScores = []
        pred = []
        label = []
        SNR = []
   
        for audioFeature, visualFeature, labels, audio, label_audios in tqdm.tqdm(loader):
            with torch.no_grad():      
                outsAV, outs_speech, outsA, outsV = self.model.forward(audioFeature[0].cuda(), visualFeature[0].cuda() ,audio[0].cuda() )
                SNR.append(cal_SISNR(outs_speech[0][0], label_audios[0][0].cuda()).cpu())

                labels = labels[0].reshape((-1)).cuda()             
                _, predScore, _, _ = self.lossAV.forward(outsAV, labels)    
                predScore = predScore[:,1].detach().cpu().numpy()
                predScores.extend(predScore)
        
        si_snr = numpy.mean(numpy.array(SNR))
        evalLines = open(evalOrig).read().splitlines()[1:]
        labels = []
        labels = pandas.Series( ['SPEAKING_AUDIBLE' for line in evalLines])
        scores = pandas.Series(predScores)
        evalRes = pandas.read_csv(evalOrig)
        evalRes['score'] = scores
        evalRes['label'] = labels
        evalRes.drop(['label_id'], axis=1,inplace=True)
        evalRes.drop(['instance_id'], axis=1,inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        cmd = "python -O utils/get_ava_active_speaker_performance.py -g %s -p %s "%(evalOrig, evalCsvSave)
        mAP = float(str(subprocess.run(cmd, shell=True, capture_output =True).stdout).split(' ')[2][:5])
        del evalRes, scores, labels, evalLines, label, pred
        torch.cuda.empty_cache()
        return mAP, si_snr

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name;
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)

    def evaluate_network_all_eva(self, loader, evalCsvSave, evalOrig, **kwargs):
        self.eval()
        predScores = []
        pred = []
        label = []
        SNR = []
        STOI = []
        PESQ = []
        AUC = []

        ground_label = []
        pred_label = []
        n_cnt = 0
        for audioFeature, visualFeature, labels, audio, label_audios in tqdm.tqdm(loader):
            with torch.no_grad():      
                outsAV, outs_speech, outsA, outsV = self.model.forward(audioFeature[0].cuda(), visualFeature[0].cuda() ,audio[0].cuda() )

                SNR.append(cal_SISNR(outs_speech[0][0], label_audios[0][0].cuda()).cpu())
                STOI.append(stoi(outs_speech[0][0].cpu(), label_audios[0][0], 16000 ))
                m_pesq = pesq(outs_speech[0][0].cpu(), label_audios[0][0], 16000 )
                if not math.isnan(m_pesq):
                    PESQ.append(m_pesq)                    

                labels = labels[0].reshape((-1)).cuda()             
                
                _, predScore, predLabel, _ = self.lossAV.forward(outsAV, labels)    

                ground_label.extend(labels.cpu().numpy().tolist())
                pred_label.extend(predLabel.cpu().numpy().tolist())
                predScore = predScore[:,1].detach().cpu().numpy()
                predScores.extend(predScore)


        auc = roc_auc_score(ground_label, predScores)
        si_snr = numpy.mean(numpy.array(SNR))
        # print('SI_SNR:', si_snr)
        m_stoi = numpy.mean(numpy.array(STOI))
        # print('STOI:', m_stoi)
        m_pesq = numpy.mean(numpy.array(PESQ))
        # print('PESQ:', m_pesq)
        evalLines = open(evalOrig).read().splitlines()[1:]
        labels = []
        labels = pandas.Series( ['SPEAKING_AUDIBLE' for line in evalLines])
        scores = pandas.Series(predScores)
        evalRes = pandas.read_csv(evalOrig)
        evalRes['score'] = scores
        evalRes['label'] = labels
        evalRes.drop(['label_id'], axis=1,inplace=True)
        evalRes.drop(['instance_id'], axis=1,inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        cmd = "python -O utils/get_ava_active_speaker_performance.py -g %s -p %s "%(evalOrig, evalCsvSave)
        mAP = float(str(subprocess.run(cmd, shell=True, capture_output =True).stdout).split(' ')[2][:5])
        del evalRes, scores, labels, evalLines, label, pred
        torch.cuda.empty_cache()
        return mAP, auc, si_snr, m_stoi, m_pesq