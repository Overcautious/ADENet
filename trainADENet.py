import time, os, torch, argparse, warnings, glob

from dataLoader import train_loader, val_loader
from utils.tools import *
from ADENet import ADENet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description = "ADENet Training")
    # Training details
    parser.add_argument('--lr',           type=float, default=0.0001,help='Learning rate')
    parser.add_argument('--lrDecay',      type=float, default=0.95,  help='Learning rate decay rate')
    parser.add_argument('--maxEpoch',     type=int,   default=45,    help='Maximum number of epochs')
    parser.add_argument('--testInterval', type=int,   default=1,     help='Test and save every [testInterval] epochs')
    parser.add_argument('--testBegin', type=int,   default=15,     help='Begin Test and save every [testInterval] epochs')
    parser.add_argument('--batchSize',    type=int,   default=304,  help='Dynamic batch size, default is 2500 frames, other batchsize (such as 1500) will not affect the performance')
    parser.add_argument('--nDataLoaderThread', type=int, default=12,  help='Number of loader threads')
    parser.add_argument('--noiseMixDB', type=int, default=0,  help='noise  Mix  DB')
    # Data path
    parser.add_argument('--dataPathAVA',  type=str, default="/mnt/data/xjw/data/AVAData", help='Save path of AVA dataset')
    parser.add_argument('--dataPathMUSAN',  type=str, default="/mnt/data/xjw/zy_data/ADENet/noise_dataset/noise_speech_txt", help='Save path of AVA dataset')
    parser.add_argument('--savePath',     type=str, default="exps")
    # Data selection
    parser.add_argument('--evalDataType', type=str, default="val", help='Only for AVA, to choose the dataset for evaluation, val or test')
    # For download dataset only, for evaluation only
    parser.add_argument('--downloadAVA',     dest='downloadAVA', action='store_true', help='Only download AVA dataset and do related preprocess')
    
    parser.add_argument('--isDown',     type=bool, default=True , help='circulant down stream')
    parser.add_argument('--isUp',      type=bool, default=True , help='circulant up stream')
    

    args = parser.parse_args()
    args = init_args(args)
    
    if args.downloadAVA == True:
        preprocess_AVA(args)
        quit()

    # Data loader
    loader = train_loader(trialFileName = args.trainTrialAVA, \
                          audioPath      = os.path.join(args.audioPathAVA , 'train'), \
                          visualPath     = os.path.join(args.visualPathAVA, 'train'), \
                          noise_db = args.noiseMixDB,\
                          musanPath = os.path.join(args.dataPathMUSAN, 'train_list.txt'),
                          **vars(args))

    trainLoader = torch.utils.data.DataLoader(loader, batch_size = 1, num_workers = args.nDataLoaderThread)

    loader = val_loader(trialFileName = args.evalTrialAVA, \
                        audioPath     = os.path.join(args.audioPathAVA , args.evalDataType), \
                        visualPath    = os.path.join(args.visualPathAVA, args.evalDataType), \
                        noise_db = args.noiseMixDB,\
                        musanPath = os.path.join(args.dataPathMUSAN, 'train_list.txt'),
                        **vars(args))
                    
    valLoader = torch.utils.data.DataLoader(loader, batch_size = 1, num_workers = args.nDataLoaderThread)



    modelfiles = glob.glob('%s/model_0*.model'%args.modelSavePath)
    modelfiles.sort()  
    if len(modelfiles) >= 1:
        print("Model %s loaded from previous state!"%modelfiles[-1])
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        s = ADENet(epoch = epoch, **vars(args))
        s.loadParameters(modelfiles[-1])
    else:
        epoch = 1
        s = ADENet(epoch = epoch, **vars(args))

    mAPs , SI_snrs = [], []
    STOI, PESQ = [], []
    AUC = []
    
    scoreFile = open(args.scoreSavePath, "a+")

    paramFile = open(args.paramSavePath, "a+")
    print('--------args----------', file=paramFile)
    localtime = time.asctime( time.localtime(time.time()) )
    print(localtime, file=paramFile)
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]), file=paramFile)
    print('--------args----------\n', file=paramFile)
    paramFile.close()
    while(1):        
        loss, lr = s.train_network(epoch = epoch, loader = trainLoader, **vars(args))
        s.saveParameters(args.modelSavePath + "/model_%04d.model"%epoch)
        if epoch % args.testInterval == 0 and epoch > args.testBegin:        
            s.saveParameters(args.modelSavePath + "/model_%04d.model"%epoch)

            mAPs.append(mAP)
            SI_snrs.append(si_snr)

            mAP, auc, si_snr, m_stoi, m_pesq = s.evaluate_network_all_eva(epoch = epoch, loader = valLoader, **vars(args))
        
            mAPs.append(mAP)
            SI_snrs.append(si_snr)
            STOI.append(m_stoi)
            PESQ.append(m_pesq)
            AUC.append(auc)
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, mAP %2.2f%%, bestmAP %2.2f%%, AUC %2.2f%%, bestAUC %2.2f%%  , si_snr %2.2f%% , bestSI_snr %2.2f%% , \
                STOI %2.2f%% , bestSTOI %2.2f%% , PESQ %2.2f%% , bestPESQ %2.2f%%"%(epoch, mAPs[-1], max(mAPs), AUC[-1], max(AUC), SI_snrs[-1], max(SI_snrs), STOI[-1], max(STOI), PESQ[-1], max(PESQ)))
            scoreFile.write("%d epoch, LR %f, LOSS %f, mAP %2.2f%%, bestmAP %2.2f%%, AUC %2.2f%%, bestAUC %2.2f%%  , si_snr %2.2f%% , bestSI_snr %2.2f%% , \
                STOI %2.2f%% , bestSTOI %2.2f%% , PESQ %2.2f%% , bestPESQ %2.2f%% \n"%(epoch,lr, loss, mAPs[-1], max(mAPs), AUC[-1], max(AUC), SI_snrs[-1], max(SI_snrs), STOI[-1], max(STOI), PESQ[-1], max(PESQ)))
            scoreFile.flush()
           
        if epoch >= args.maxEpoch:
            quit()

        epoch += 1

if __name__ == '__main__':
    main()
