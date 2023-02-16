import torch
import numpy as np
import math
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from torchvision.transforms import transforms
import torch.nn as nn
from tqdm import tqdm
from math import sqrt
import GPUtil
import pickle
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show
from timeit import default_timer as timer

from Models.UnifiedClassificaionAndRegressionAgeModel import UnifiedClassificaionAndRegressionAgeModel
from face_classes import FaceClassification, EvaluateNet, sigmoid
from Schedulers.GradualWarmupScheduler import GradualWarmupScheduler
from Optimizers.RangerLars import RangerLars

from Losses import AngleLinear, AngleLoss, Heatmap1Dloss, WeightedBinaryCrossEntropyLoss, MetricLearningLoss,\
    WeightedCrossEntropyLoss, MeanVarianceLoss, CenterLoss

# from utils.utils import LoadModel, PrepareLoders, SearchAutoAugmentations
# from utils.read_matlab_imdb import read_matlab_imdb

torch.autograd.set_detect_anomaly(True)
import warnings

warnings.filterwarnings("ignore", message="UserWarning: albumentations.augmentations.transforms.RandomResizedCrop")
from multiprocessing import Process, freeze_support

torch.backends.cudnn.deterministic = True  # needed
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    freeze_support()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NumGpus = 1  # torch.cuda.device_count()
    torch.cuda.empty_cache()
    GPUtil.showUtilization()

    print(device)
    # name = torch.cuda.get_device_name(0)

    ModelsDirName = './models1/'
    LogsDirName = './logs1/'
    Description = 'DeepAge'
    BestFileName = 'face_best'
    FileName = 'visnir_sym_triplet'
    TestDir = 'F:\\multisensor\\test\\'
    TrainFile = 'f:\\deepage\\face_dataset.hdf5'

    writer = SummaryWriter(LogsDirName)
    LowestError = 1e10

    # ----------------------------     configuration   ---------------------------
    MseLoss = nn.MSELoss().to(device)
    CentLoss = CenterLoss(reduction='mean').to(device)
    MeanVareLoss = MeanVarianceLoss(LamdaMean=0.2, LamdaSTD=0.05, device=device).to(device)
    MetricLearnLoss = MetricLearningLoss(LabelsDistT=3, reduction='mean').to(device)
    CeLoss = nn.CrossEntropyLoss().to(device)
    AngularleLoss = AngleLoss().to(device)
    BCEloss = nn.BCEWithLogitsLoss().to(device)
    CELogProb = nn.NLLLoss().to(device)
    KLDiv = nn.KLDivLoss(reduction='batchmean').to(device)

    ShowFigure = False

    # CnnMode = 'Shaked'
    # CnnMode = 'Cascade'
    CnnMode = 'Classify'

    # SamplingMode = 'UniformAge'
    SamplingMode = 'Random'

    # criterion=None
    criterion = nn.CrossEntropyLoss()

    InitializeOptimizer = True
    OuterBatchSize = 3*4
    InnerBatchSize = 16
    TrainAugmentNo = 1
    TestAugmentNo = 1

    RandAugmenParams = {"N": 3, "M": 4}

    LearningRate = 1e-3
    MinAge = 15
    MaxAge = 80
    AgeIntareval = 1  # 5
    NumLabels = (MaxAge - MinAge) / AgeIntareval + 1

    CascadeSupport = 15
    CascadeSkip = 5
    FcK = 256
    UseCascadedLoss = {}
    UseCascadedLoss['Class'] = False
    UseCascadedLoss['MetricLearning'] = False
    UseCascadedLoss['Center'] = False
    UseCascadedLoss['Ordinal'] = False
    UseCascadedLoss['Sphere'] = False
    UseCascadedLoss['Regress'] = False
    UseCascadedLoss['MeanVar'] = False

    UseGradientNorm = True
    DropoutP = 0.5
    weight_decay = 1e-5

    StartBestModel = False
    UseBestScore = False

    # FaceNet
    FreezeBaseCnn = True
    FreezeFaceNetFC = True

    # age classifcation
    FreezeClassEmbeddFC = True
    FreezeAgeClassificationFC = True

    # emebedding layer
    FreezeEmbedding = False

    # main ordinal
    FreezeOrdinalLayers  = False
    FreezeOrdinalFC      = False

    # heatmap
    FreezeHeatmapLayers = True

    # age and gender
    FreezeEthnicityLayers = True
    FreezeGenderLayers = True

    FreezeOrdinalEncoder = False


    # ordinal losses
    UseOridinalLoss = True
    UseExtendedOridinalLoss = False


    # losses
    UseAgeClassLoss = False
    UseClassCenterLoss = False
    UseOrdinalCenterLoss = False
    UseMetricLearLoss = False
    UseClassMeanValLoss = False
    UseAngularLoss = False
    UseRegressionLoss = False

    UseHeatmapLoss = False



    # gender & ethnicity & faces
    UseClassifyFacesLoss = False
    UseGenderLoss = False
    UseEthnicityLoss = False
    UseAgeGenderRace = False

    TrainingRatio = 0.8

    Heatmap10 = Heatmap1Dloss(device, NumLabes=NumLabels, sigma=10.0)
    WeightedBceLoss = WeightedBinaryCrossEntropyLoss(NumLabels, device, reduction='mean', binary_loss_type='BCE')

    ContinueMode = True

    # np.random.seed(0)
    SaveTrainState = False

    # ----------------------------- read data----------------------------------------------

    num_workers = 8
    Test_DataLoader, Train_DataLoader, Data = \
        PrepareLoders(TrainFile, TrainingRatio, SaveTrainState, MinAge, AgeIntareval, InnerBatchSize, OuterBatchSize,
                      TrainAugmentNo, TrainAugmentNo,
                      SamplingMode, num_workers, TrainTransformMode="TrainA", TestTransformMode="Test",
                      RandAugmenParams=RandAugmenParams)

    # -------------------------    loading previous results   ------------------------
    net = FaceClassification(NumClasses=int(NumLabels), CascadeSupport=CascadeSupport, CascadeSkip=CascadeSkip,
                             NumFaceClasses=Data['LinearIds'].max() + 1, K=FcK, DropoutP=DropoutP,
                             AugmentNo=TrainAugmentNo)

    # net = UnifiedClassificaionAndRegressionAgeModel(int(NumLabels), AgeIntareval, MinAge, MaxAge)

    StartEpoch = 0
    if ContinueMode:
        net, optimizer, LowestError, StartEpoch, scheduler = LoadModel(net, StartBestModel, ModelsDirName, BestFileName,
                                                                       UseBestScore, device)

    net.to(device)
    if InitializeOptimizer:

        if True:
            FaceNetNames = ['FaceNet.' + x[0] for x in
                            net.base_net.named_parameters()]  ## 'FaceNet.conv2d_1a.conv.weight'
            BaseParams = []
            for name, param in net.named_parameters():
                if name not in FaceNetNames:
                    BaseParams.append(param)

            optimizer = torch.optim.Adam([
                {'params': net.base_net.parameters(), 'lr': LearningRate, 'weight_decay': weight_decay},
                {'params': BaseParams, 'lr': LearningRate, 'weight_decay': weight_decay}
            ], lr=LearningRate)

            # optimizer = torch.optim.SGD([
            #    {'params': net.FaceNet.parameters(), 'lr': LearningRate / 10, 'weight_decay': weight_decay},
            #    {'params': BaseParams, 'lr': LearningRate, 'weight_decay': weight_decay}
            # ], lr=LearningRate)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad == True, net.parameters()), lr=LearningRate,weight_decay=weight_decay)

        #optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad == True, net.parameters()), lr=LearningRate,weight_decay=weight_decay)
        # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad == True, net.parameters()), lr=LearningRate,weight_decay=weight_decay,momentum=0.9)
        # optimizer = RangerLars(net.parameters(), lr=LearningRate)


        # scheduler  = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=0, verbose=True)
        scheduler = StepLR(optimizer, step_size=10, gamma=math.sqrt(0.1))
        # scheduler = GradualWarmupScheduler(optimizer,multiplier=2,total_epoch=5,after_scheduler=scheduler)
        # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0.)

    # ------------------------------------------------------------------------------------------

    # -------------------------------------  freeze layers --------------------------------------
    # classification

    net.FreezeBaseCnn(FreezeBaseCnn)
    net.FreezeFaceNetFC(FreezeFaceNetFC)

    net.FreezeAgeClassificationFC(FreezeAgeClassificationFC)
    net.FreezeClassEmbeddFC(FreezeClassEmbeddFC)

    net.FreezeEmbedding(FreezeEmbedding)

    # ordinal
    net.FreezeOrdinalLayers(FreezeOrdinalLayers)
    net.FreezeOrdinalFC(FreezeOrdinalFC)

    # heatmap
    net.FreezeHeatmapLayers(FreezeHeatmapLayers)

    # Gender
    net.FreezeGenderLayers(FreezeGenderLayers)

    # Ethnicirt
    net.FreezeEthnicityLayers(FreezeEthnicityLayers)

    net.FreezeOrdinalEncoder(FreezeOrdinalEncoder)

    if NumGpus > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
        #net = nn.parallel.DistributedDataParallel(net)

    ########################################################################
    # Train the network

    TempotalError = np.array([])

    # writer.add_graph(net, images)
    start_generator = 0
    end_generator = 0
    start_cnn = 0
    end_cnn = 0
    for epoch in range(StartEpoch, 200):  # loop over the dataset multiple times

        print('\n\nStarting epoch: ' + repr(epoch))


        #choose augmentation per epoch
        if Train_DataLoader.dataset.TransformMode == 'AugmentTrainA':
            Train_DataLoader.dataset.ChooseRandomAugmentation()
            print(Train_DataLoader.dataset.transformA)


        # if epoch == 0:# Warmpup
        #   scheduler = StepLR(optimizer, step_size=1, gamma=10)
        # if epoch==10:
        #   scheduler = StepLR(optimizer, step_size=5, gamma=sqrt(0.1))

        if NumGpus == 1:
            if epoch < 55:  net.FreezeBaseCnn(True)  # net.freeze_base_cnn(True)
            if epoch > 55:  net.FreezeBaseCnn(False)
            print('net.fc_class.weight.requires_grad: ' + repr(net.fc_class.weight.requires_grad))
            print('net.fc_class1.weight.requires_grad: ' + repr(net.fc_class1.weight.requires_grad))
            print('net.ordinal_fc.weight.requires_grad: ' + repr(net.ordinal_fc.weight.requires_grad))
            print('net.ordinal_fc1.weight.requires_grad: ' + repr(net.ordinal_fc1.weight.requires_grad))
            print('net.FaceNet.last_linear.weight.requires_grad: ' + repr(net.base_net.last_linear.weight.requires_grad))
        else:
            if epoch < 55:  net.module.FreezeBaseCnn(True)
            if epoch > 55:  net.module.FreezeBaseCnn(False)
            print('net.module.fc_class.weight.requires_grad: ' + repr(net.module.fc_class.weight.requires_grad))
            print('net.module.fc_class1.weight.requires_grad: ' + repr(net.module.fc_class1.weight.requires_grad))
            print('net.module.ordinal_fc.weight.requires_grad: ' + repr(net.module.ordinal_fc.weight.requires_grad))
            print('net.module.ordinal_fc1.weight.requires_grad: ' + repr(net.module.ordinal_fc1.weight.requires_grad))
            print('net.FaceNet.last_linear.weight.requires_grad: ' + repr(
                net.module.FaceNet.last_linear.weight.requires_grad))

        if epoch > StartEpoch:
            # scheduler.step(running_loss)
            scheduler.step()
            str = 'running_loss=' + repr(running_loss)[0:6]
        else:
            str = ''

        # Print Learning Rates

        str += ' LR: '
        for param_group in optimizer.param_groups:
            str += repr(param_group['lr']) + ' '
        print(str + '\n')

        running_loss = 0
        running_regression = 0
        optimizer.zero_grad()
        bar = tqdm(Train_DataLoader, 0, leave=False)
        for i, TrainData in enumerate(bar):

            net = net.train()

            # get the inputs
            Labels = TrainData['Labels']
            CurrentImages = TrainData['Images']
            Age = TrainData['Ages']
            # FaceIds = TrainData['Ids']
            # Ethnicity1 = TrainData['Race']
            # Gender1 = TrainData['Gender']

            Age = np.reshape(Age.transpose(0, 1), (Age.shape[0] * Age.shape[1]), order='F')
            # Ethnicity1 = np.reshape(Ethnicity1, (Ethnicity1.shape[0] * Ethnicity1.shape[1]), order='F')
            # Gender1 = np.reshape(Gender1, (Gender1.shape[0] * Gender1.shape[1]), order='F')
            # FaceIds = np.reshape(FaceIds, (FaceIds.shape[0] * FaceIds.shape[1]), order='F')
            Labels = np.reshape(Labels.transpose(0, 1), (Labels.shape[0] * Labels.shape[1]), order='F')
            CurrentImages = np.reshape(CurrentImages.transpose(0, 1),
                                       (CurrentImages.shape[0] * CurrentImages.shape[1], CurrentImages.shape[2],
                                        CurrentImages.shape[3],
                                        CurrentImages.shape[4]),
                                       order='F')

            Labels, CurrentImages, Age = Labels.to(device), CurrentImages.to(device), Age.to(device)
            # Ethnicity1, Gender1, FaceIds  = Ethnicity1.to(device), Gender1.to(device), FaceIds.to(device)

            if start_generator > 0:
                end_generator += timer() - start_generator

            start_cnn = timer()
            # Embed = net(CurrentImages, CnnMode, Labels=Age,DropoutP = 0.5)
            Embed = net(CurrentImages, Mode=CnnMode, Labels=Labels)
            end_cnn += timer() - start_cnn

            loss = 0

            # if (CnnMode == 'Classify') | (CnnMode == 'Shaked'):

                # if UseClassifyFacesLoss:
                #     # IdClassLoss = CeLoss(Embed['IdEmbed'], FaceIds )
                #     IdClassLoss = AngularleLoss(Embed['IdEmbed'], Labels.round().long())
                #     loss += IdClassLoss

                # if UseAgeGenderRace:
                #     AgeGenderRaceClassLoss = criterion(Embed['ClassAgeGenderRace'], torch.round(Labels).long(), )
                #     loss += AgeGenderRaceClassLoss

                # if UseMetricLearLoss:
                #     MlLoss = MetricLearnLoss(Embed['Base'], Labels.round().long(), Mode='Hard')
                #     # MlLoss = MetricLearnLoss(Embed['Base'], torch.round(Labels).long(), Mode='Random')
                #     loss += MlLoss

                # if UseAngularLoss:
                #     AngLoss = AngularleLoss(Embed['Sphere'], Labels.round().long())
                #     loss += AngLoss

                # if UseGenderLoss:
                #     GenderLoss = BCEloss(Embed['Gender'].squeeze(), Gender1.float())
                #     loss += GenderLoss

                # if UseEthnicityLoss:
                #     EthnicityLoss = CeLoss(Embed['Ethnicity'].squeeze(), Ethnicity1)
                #     loss += EthnicityLoss

                # if UseRegressionLoss:
                #     RegressionLoss = ((Embed['Regression'] - Age) ** 2).mean()
                #     loss += RegressionLoss
                #
                #     running_regression += (Embed['Regression'] - Age).abs().mean().item()

            # if (CnnMode == 'Cascade') | (CnnMode == 'Regress'):
            #
            #     if NumGpus == 1:
            #         loss1, AllLosses = criterion(CnnMode, Embed, Labels.round().long(), Age.long(),
            #                                      UseLoss=UseCascadedLoss,
            #                                      EmbeddingCenters=net.CascadeEmbedding,
            #                                      ApplyAngularLoss=net.CascadeAngularLoss)
            #     else:
            #         loss1, AllLosses = criterion(CnnMode, Embed, Labels.round().long(), Age.long(),
            #                                      UseLoss=UseCascadedLoss,
            #                                      EmbeddingCenters=net.module.CascadeEmbedding,
            #                                      ApplyAngularLoss=net.module.CascadeAngularLoss)
            #     # loss += loss1

            # if UseCascadedLoss['Regress']:
            #     RegressionLoss = ((Embed['ProbRegress'] - Age) ** 2).mean()
            #     loss += RegressionLoss
            #
            #     running_regression += (Embed['ProbRegress'] - Age).abs().mean().item()

            if UseOridinalLoss:
                OrdinalLoss = WeightedBceLoss(Embed['OrdinalClass'], Labels.round().long(), ComputeWeights=True)
                loss += OrdinalLoss

            # if UseExtendedOridinalLoss:
            #     CurrentLabels = torch.clamp(Labels, 0, NumLabels - 1)
            #     ExtendedOrdinalLoss = CELogProb(Embed['ExtendedOrdinalClass'].log(), torch.round(CurrentLabels).long())
            #     loss += ExtendedOrdinalLoss
            #
            #     if False:
            #         ClassProbLog = F.log_softmax(Embed['Class'], 1)
            #         KLDivLoss = KLDiv(ClassProbLog[:, :-1], Embed['ExtendedOrdinalClass']) + KLDiv(
            #             Embed['ExtendedOrdinalClass'].log(), F.softmax(Embed['Class'][:, :-1], 1))
            #         loss += KLDivLoss
            #
            #     # H = -((Embed['ExtendedOrdinalClass']*Embed['ExtendedOrdinalClass'].log()).sum(1)).mean()
            #     # loss += H
            #
            #     MVloss = MeanVareLoss(Embed['ExtendedOrdinalClass'], CurrentLabels.round().long(), IsProbability=True)
            #     loss += MVloss

            # if UseAgeClassLoss:
            #     AgeClassLoss = CeLoss(Embed['Class'], torch.round(Labels).long(), )
            #     loss += AgeClassLoss

            # if UseClassMeanValLoss:
            #     MVloss = MeanVareLoss(Embed['Class'], Labels.round().long())
            #     loss += MVloss

            # if UseHeatmapLoss:
            #     HeatmapLoss = Heatmap3(Embed['HeatmapClass'], Labels.round().long())
            #     loss += HeatmapLoss
            #
            #     HeatmapCascadeClassLoss = Heatmap1(Embed['HeatmapCascadeClass'], Labels.round().long())
            #     loss += HeatmapCascadeClassLoss

            # if UseClassCenterLoss:
            #
            #     if NumGpus == 1:
            #         CenterLoss = CentLoss(Embed['ClassEmbed'], net.Embedding, Labels.round().long(), Mode='MSE')
            #     else:
            #         CenterLoss = CentLoss(Embed['ClassEmbed'], net.module.Embedding, Labels.round().long(), Mode='MSE')
            #     loss += CenterLoss

            # if UseOrdinalCenterLoss:
            #
            #     if NumGpus == 1:
            #         OrdinalCenterLoss = CentLoss(Embed['OrdinalEmbed'], net.Embedding, Labels.round().long(),
            #                                      Mode='MSE')
            #     else:
            #         OrdinalCenterLoss = CentLoss(Embed['OrdinalEmbed'], net.module.Embedding, Labels.round().long(),
            #                                      Mode='MSE')
            #     loss += OrdinalCenterLoss

            # backward + optimize
            loss.backward()

            clipping_value = 1
            if UseGradientNorm:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_value)

            optimizer.step()  # Now we can do an optimizer step

            # zero the parameter gradients
            optimizer.zero_grad()

            running_loss += loss.item()

            start_generator = timer()

            SchedularUpadteInterval = 100
            if (i % SchedularUpadteInterval == 0) & (i > 0):

                # str = 'Loss: ' + repr(loss.item())[0:4]
                str = 'running_loss: ' + repr(running_loss / i)[0:6]

                # TRAINING LOSS
                # if CnnMode == 'Cascade':
                #     # CascadeOrdinal = (Embed['CascadeOrdinal'] - Labels).abs().mean()
                #     # CascadeProbMean = (Embed['CascadeProb'] - Labels).abs().mean()
                #     # str += ' CascadeOrdinal: ' + repr(CascadeOrdinal.item())[0:4] #+ ' CascadeProb: ' + repr(CascadeProbMean.item())[0:4]
                #
                #     if UseCascadedLoss['MetricLearning']: str += ' Cas MetricLearning: ' + repr(
                #         AllLosses['MlLoss'].item())[0:4]
                #     if UseCascadedLoss['Center']:         str += ' Cas CenterLoss: ' + repr(AllLosses['Center'].item())[
                #                                                                        0:4]
                #     if UseCascadedLoss['Sphere']:         str += ' Cas SphereLoss: ' + repr(AllLosses['Sphere'].item())[
                #                                                                        0:4]
                #     if UseCascadedLoss['Ordinal']:        str += ' Cas OrdinalLoss: ' + repr(
                #         AllLosses['Ordinal'].item())[0:4]
                #     if UseCascadedLoss['Class']:          str += ' Cas ClassLoss: ' + repr(
                #         AllLosses['CascadeClass'].item())[0:4]
                #     if UseCascadedLoss['Regress']:        str += ' Cas Regression Error: ' + repr(
                #         running_regression / i)[0:4]

                # if UseAgeClassLoss:      str += ' AgeClassLoss: ' + repr(AgeClassLoss.item())[0:4]
                # if UseClassCenterLoss:   str += ' CenterLoss: ' + repr(CenterLoss.item())[0:4]
                # if UseOrdinalCenterLoss: str += ' CenterLoss: ' + repr(OrdinalCenterLoss.item())[0:4]
                # if UseClassMeanValLoss:  str += ' MeanValLoss: ' + repr(MVloss)[0:4]
                # if UseMetricLearLoss:    str += ' MetricLearLoss: ' + repr(MlLoss.item())[0:4]
                # if UseClassifyFacesLoss: str += ' IdClassLoss: ' + repr(IdClassLoss.item())[0:4]
                # if UseAngularLoss:       str += ' AngularLoss: ' + repr(AngLoss.item())[0:4]
                # if UseRegressionLoss:      str += ' Regression Error: ' + repr(running_regression / i)[0:4]
                # if UseGenderLoss:
                #     Error = ((Embed['Gender'] > 0).squeeze().int() != Gender1).float().mean()
                #     str += ' GenderLoss: ' + repr(GenderLoss.item())[0:4] + ' G. Error: ' + repr(Error.item())[0:4]
                # if UseEthnicityLoss:
                #     Error = (Embed['Ethnicity'].argmax(1) != Ethnicity1).float().mean()
                #     str += ' GenderLoss: ' + repr(EthnicityLoss.item())[0:4] + ' Eth. Error: ' + repr(Error.item())[0:4]
                if UseOridinalLoss:
                    Error = ((Embed['OrdinalClass'] > 0).sum(1) - Labels).abs().mean()
                    str += ' OridinalLoss: ' + repr(OrdinalLoss.item())[0:4] + ' Ordinal Error: ' + repr(Error.item())[
                                                                                                    0:4]
                # if UseExtendedOridinalLoss:   str += ' Extended Ordinal Error: ' + repr(ExtendedOrdinalLoss.item())[0:4]
                # if UseHeatmapLoss:
                #     Error = (Embed['HeatmapClass'].argmax(1) - Labels).abs().mean()
                #     str += ' Heatmap Error: ' + repr(Error.item())[0:4]

                print(str)

            PrintStep = 1000
            if ((i % PrintStep == 0) or (i * InnerBatchSize >= len(Train_DataLoader) - 1)) and (i > 0):

                if i > 0:
                    print('generator time: ' + repr(end_generator / i)[0:5] + ' CNN time: ' + repr(end_cnn / i)[0:5])
                end_generator = 0
                end_cnn = 0

                if (i % PrintStep == 0):
                    running_loss /= PrintStep
                else:
                    running_loss /= i

                net = net.eval()

                str = '\n [%d, %5d] loss: %.3f' % (epoch, i, 100 * running_loss)

                if ShowFigure:
                    fig, ax = plt.subplots()

                with torch.set_grad_enabled(False):

                    # val accuracy

                    if CnnMode == 'Cascade':
                        TestMode = 'Cascade_test'
                    else:
                        TestMode = CnnMode

                    if (torch.cuda.device_count() > 1) and (NumGpus == 1):
                        net = nn.DataParallel(net)

                    #NoTests = 1000
                    #SearchAutoAugmentations(NoTests, net, Test_DataLoader, device, Mode=TestMode)

                    print('Evaluating results')

                    AugmentEmb = []
                    for k in range(TestAugmentNo):
                        AugmentEmb.append(EvaluateNet(net, Test_DataLoader, device, Mode=TestMode))

                    if (torch.cuda.device_count() > 1) and (NumGpus == 1):
                        net = net.module

                    Emb = dict()

                    if 'Class' in AugmentEmb[0].keys():
                        Emb['Class'] = AugmentEmb[0]['Class']
                    if 'ProbRegress' in AugmentEmb[0].keys():
                        Emb['ProbRegress'] = AugmentEmb[0]['ProbRegress']
                    if 'OrdinalClass' in AugmentEmb[0].keys():
                        Emb['OrdinalClass'] = AugmentEmb[0]['OrdinalClass']
                    for i in range(1, TestAugmentNo):
                        if 'Class' in AugmentEmb[0].keys():
                            Emb['Class'] += AugmentEmb[i]['Class']
                        if 'ProbRegress' in AugmentEmb[0].keys():
                            Emb['ProbRegress'] += AugmentEmb[i]['ProbRegress']
                        if 'OrdinalClass' in AugmentEmb[0].keys():
                            Emb['OrdinalClass'] += AugmentEmb[i]['OrdinalClass']

                    if 'ProbRegress' in AugmentEmb[0].keys():
                        Emb['ProbRegress'] /= TestAugmentNo
                    if 'OrdinalClass' in AugmentEmb[0].keys():
                        Emb['OrdinalClass'] /= TestAugmentNo
                    Emb['Labels'] = AugmentEmb[0]['Labels']
                    Emb['Ages'] = AugmentEmb[0]['Ages']

                    ClassResult = Emb['Class'].argmax(1)
                    ClassError = np.abs(ClassResult - Emb['Labels'].squeeze()) * AgeIntareval

                    TempotalError = np.append(TempotalError, ClassError.mean())

                    class_count = np.bincount(ClassError.astype(np.long))

                    if ShowFigure:
                        ax.plot(class_count, 'b-*', label='Class');

                    str += ' Class Average: ' + repr(ClassError.mean())[0:4]

                    # if UseAgeGenderRace:
                    #     ClassResult = Emb['ClassAgeGenderRace'].argmax(1)
                    #     Error = np.abs(ClassResult - np.round(TestLabels.numpy()))
                    #
                    #     str += ' ClassAgeGenderRace Average: ' + repr(Error.mean())[0:4]

                    if UseOridinalLoss:
                        OrdinalError = ((Emb['OrdinalClass'] > 0).sum(1) - np.round(
                            Emb['Labels'].squeeze())) * AgeIntareval
                        print('Ordinal error mean: ' + repr(OrdinalError.mean())[0:4])

                        OrdinalError = np.abs(OrdinalError)

                        # plot(Probs[idx[0],:]);show()

                        class_count = np.bincount(OrdinalError.astype(np.long))

                        if ShowFigure:
                            ax.plot(class_count, 'g-s', label='Ordinal')

                        str += ' Ordinal AverageLoss: ' + repr(OrdinalError.mean())[0:6]

                    # if UseExtendedOridinalLoss:
                    #     OrdinalProbs = sigmoid(Emb['OrdinalClass'])
                    #     BaseOrdinalClassificationProbs = -(OrdinalProbs[:, 1:] - OrdinalProbs[:, 0:-1])
                    #     # plt.plot(BaseOrdinalClassificationProbs[0, :].detach().cpu());show()
                    #     ExtendOrdinalClassIdx = BaseOrdinalClassificationProbs.argmax(1)
                    #
                    #     TestLabels = np.clip(np.round(Emb['Labels'].squeeze()), a_min=0, a_max=NumLabels - 1)
                    #     ExtendedOrdinalError = np.abs(TestLabels - ExtendOrdinalClassIdx).mean()
                    #
                    #     str += ' ExtendedOridinal Average: ' + repr(ExtendedOrdinalError)[0:4]

                    # if UseRegressionLoss:
                    #     RegressionLossError = np.abs((Emb['Regression'] - Emb['Ages'].squeeze())).mean()
                    #     str += ' Regression Error: ' + repr(RegressionLossError)[0:4]
                    #     CurrentError = RegressionLossError

                    # if UseHeatmapLoss:
                    #     HeatmapResult = Emb['HeatmapClass'].argmax(1)
                    #     HeatmapError = (HeatmapResult - np.round(TestLabels.numpy())) * AgeIntareval
                    #     str += ' Heatmap L1: ' + repr(np.abs(HeatmapError).mean())[0:4] + ' Heatmap mean: ' + repr(
                    #         HeatmapError.mean())[0:4]
                    #
                    #     class_count = np.bincount(np.abs(HeatmapError).astype(np.long))
                    #
                    #     if ShowFigure:
                    #         ax.plot(class_count, 'r-+', label='Heatmap');
                    #
                    #     HeatmapResult = Emb['HeatmapCascadeClass'].argmax(1)
                    #     HeatmapError = (HeatmapResult - np.round(TestLabels.numpy())) * AgeIntareval
                    #     str += ' CascadesHeatmap L1: ' + repr(np.abs(HeatmapError).mean())[
                    #                                      0:4] + ' CascadesHeatmap mean: ' + repr(HeatmapError.mean())[
                    #                                                                         0:4]
                    #
                    #     class_count = np.bincount(np.abs(HeatmapError).astype(np.long))
                    #     if ShowFigure:
                    #         ax.plot(class_count, 'k-*', label='CasHeatmap');

                    # if UseGenderLoss:
                    #     GenderError = ((Emb['Gender'] > 0).squeeze().astype(int) != Gender[Data['TestSamples']]).astype(
                    #         float).mean()
                    #     print('GenderError: ' + repr(GenderError)[0:4])
                    #
                    # if UseEthnicityLoss:
                    #     EthnicityError = (Emb['Ethnicity'].argmax(1) != Ethnicity[Data['TestSamples']]).astype(
                    #         float).mean()
                    #     print('EthnicityError: ' + repr(EthnicityError)[0:4])

                    # compute and print errors

                    print(str)

                    # if CnnMode == 'Cascade':
                    #
                    #     str = ''
                    #
                    #     # greedy
                    #     if False:
                    #         CasslError = np.abs(np.squeeze(Emb['CascadeGreedyClass']) - TestLabels.numpy())
                    #         str = ' Casscade greedy: ' + repr(CasslError.mean())[0:4]
                    #
                    #         # ordinal
                    #         CassOdinalError = (np.squeeze(Emb['CascadeOrdinal']) - TestLabels.numpy())[idx]
                    #         CasslErrorMean = CassOdinalError.mean()
                    #         CasslError = np.abs(CassOdinalError)
                    #         str += ' Casscade ordinal mean: ' + repr(CasslErrorMean.mean())[0:4]
                    #         str += ' Casscade ordinal: ' + repr(CasslError.mean())[0:4]
                    #
                    #         class_count = np.bincount((CasslError).astype(np.long))
                    #
                    #         if ShowFigure:
                    #             ax.plot(class_count, 'm-^', label='Cas oridinal');
                    #
                    #     if UseCascadedLoss['Regress']:
                    #         RegressionError = np.abs(
                    #             np.round(Emb['ProbRegress']).squeeze() - Emb['Ages'].squeeze()).mean()
                    #
                    #         str += ' Cascaded Regress Error: ' + repr(RegressionError)[0:4]
                    #
                    #     print(str)

                    if ShowFigure:
                        ax.legend(frameon=False, fontsize='xx-large');
                        show()

                    # if UseAgeClassLoss:                           CurrentError = ClassError.mean()
                    if UseOridinalLoss | UseExtendedOridinalLoss: CurrentError = OrdinalError.mean()
                    # if UseCascadedLoss['Regress']:                CurrentError = RegressionError.mean()
                    # if UseCascadedLoss['Ordinal']:                CurrentError = RegressionError.mean()

                    if (i * InnerBatchSize * OuterBatchSize >= (len(Train_DataLoader) - PrintStep)) or (
                            CurrentError < LowestError):  # | i>500:

                        state = {'epoch': epoch,
                                 'state_dict': net.state_dict() if (NumGpus == 1) else net.module.state_dict(),
                                 'optimizer_name': type(optimizer).__name__,
                                 'optimizer': optimizer.state_dict(),
                                 'scheduler_name': type(scheduler).__name__,
                                 'scheduler': scheduler.state_dict(),
                                 'Description': Description,
                                 'LowestError': LowestError,
                                 'OuterBatchSize': OuterBatchSize,
                                 'InnerBatchSize': InnerBatchSize,
                                 'DropoutP': DropoutP,
                                 'weight_decay': weight_decay,
                                 'CnnMode ': CnnMode}

                        if CurrentError < LowestError:
                            LowestError = CurrentError
                            print('Best error found and saved: ' + repr(LowestError)[0:5])
                            filepath = ModelsDirName + BestFileName + '.pth'
                            torch.save(state, filepath)

                        if (i >= (len(Train_DataLoader) - PrintStep)) | True:
                            # end of epoch
                            filepath = ModelsDirName + FileName + repr(epoch) + '.pth'
                            torch.save(state, filepath)
                            print('Saved checkpoint ' + filepath)

                    x = (epoch * len(Train_DataLoader) + i) / PrintStep
                    #writer.add_text('Text', str)
                    #writer.close()

                    net = net.train()

                    if (i * InnerBatchSize) > (len(Train_DataLoader) - 1):
                        bar.clear()
                        bar.close()
                        break

        print('\n\n')

    print('Finished Training')