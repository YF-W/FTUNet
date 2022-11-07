import torch
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from numpy import *

# from model_unet import UNET
# from WyfUnetTest import MyUNet
# from WyfUnetTest_v3_33_with7times7and9times9_withoutAttention import MyUNET
# from model_unet import UNET
# from BAS_two import BAS
# from BASTD import BAS
# from baseline_v2 import RAS
# from RAS_B_v2 import RAS
# from RTRAT_v2 import DTUM
# from TransUnet_baseline import TransUnet_baseline
from model import FTUNet
# from RAT import RAT
# from FAT_Net import FAT_Net
# from RTRAS_v2 import DTUM
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    DiceLoss
)
from torch.utils.data import DataLoader
from dataset import LungDataset
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, ConcatDataset
import warnings
warnings.filterwarnings("ignore")
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# Hyperparameters etc.
#
LEARNING_RATE = 1e-4
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 224  # 512 originally
IMAGE_WIDTH = 224  # 512 originally
PIN_MEMORY = False
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/data_cell/train_images"
TRAIN_MASK_DIR = "data/data_cell/train_masks"
VAL_IMG_DIR = "data/data_cell/val_images"
VAL_MASK_DIR = "data/data_cell/val_masks"

def train_fn(loader, model, optimizer, loss_fn, scaler, epoch):
    print(f"---Epoch:{epoch}---")
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    ########

    kfold = KFold(n_splits = 5, shuffle = False)

    # Start print
    print('--------------------------------')

    train_dataset = LungDataset(
        image_dir=TRAIN_IMG_DIR,
        mask_dir=TRAIN_MASK_DIR,
        transform=train_transform,
    )

    val_dataset = LungDataset(
        image_dir=VAL_IMG_DIR,
        mask_dir=VAL_MASK_DIR,
        transform=train_transform,
    )
    all_dataset = ConcatDataset([train_dataset, val_dataset])

    # K-fold Cross Validation model evaluation
    #5折交叉验证的总平均值
    total_acc_list = []
    total_dice_list = []
    total_jaccord_list = []
    total_precision_list = []
    total_recall_list = []
    total_f_score_list = []
    total_specificity_list = []
    total_auc_list = []
    for fold, (train_ids, test_ids) in enumerate(kfold.split(all_dataset)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SequentialSampler(train_ids)
        test_subsampler = torch.utils.data.SequentialSampler(test_ids)

        # Define data loaders for training and testing data in this fold

        trainloader = DataLoader(
            all_dataset,
            batch_size = BATCH_SIZE,
            num_workers = NUM_WORKERS,
            pin_memory = PIN_MEMORY, sampler = train_subsampler,drop_last=True
        )
        testloader = DataLoader(
            all_dataset,
            batch_size = BATCH_SIZE,
            num_workers = NUM_WORKERS,
            pin_memory = PIN_MEMORY,
            sampler = test_subsampler,drop_last=True
        )

        # model = MyUNet(in_channels=3, out_channels=1, w=512, h=512).to(DEVICE)
        # model = MyUNET(in_channels=3, out_channels=1).to(DEVICE)
        # model = UNET(in_channels=3, out_channels=1).to(DEVICE)
        # model = BAS(n_channels=3).to(DEVICE)
        # model = RAS(n_channels=3).to(DEVICE)
        # model = RAT(n_channels=3).to(DEVICE)
        model = FTUNet(n_channels=3).to(DEVICE)
        # model = FAT_Net(n_channels=3).to(DEVICE)
        # model = TransUnet_baseline(n_channels=3).to(DEVICE)
        loss_fn = nn.BCEWithLogitsLoss()
        # loss_fn = DiceLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)



        # if LOAD_MODEL:
        #     load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

        check_accuracy(testloader, model, device=DEVICE)
        scaler = torch.cuda.amp.GradScaler()
        epoch_list = []
        acc_list = []
        dice_list = []
        jaccord_list = []
        precision_list=[]
        recall_list=[]
        f_score_list=[]
        specificity_list=[]
        auc_list=[]
        tp_list=[]
        tn_list=[]
        fp_list=[]
        fn_list=[]

        for epoch in range(NUM_EPOCHS):

            epoch_list.append(epoch)
            train_fn(trainloader, model, optimizer, loss_fn, scaler, epoch)

            # # save model
            # checkpoint = {
            #     "state_dict": model.state_dict(),
            #     "optimizer": optimizer.state_dict(),
            # }
            # save_checkpoint(checkpoint)

            # check accuracy
            # check_accuracy(val_loader, model, device=DEVICE)
            acc, dice, jaccord, precision, recall,f_score,specificity,auc,tp,tn,fp,fn = check_accuracy(testloader, model, device = DEVICE)
            acc_list.append(acc)
            dice_list.append(dice)
            jaccord_list.append(jaccord)
            precision_list.append(precision)
            recall_list.append(recall)
            f_score_list.append(f_score)
            specificity_list.append(specificity)
            auc_list.append(auc)
            tp_list.append(tp)
            tn_list.append(tn)
            fp_list.append(fp)
            fn_list.append(fn)
            # plt.plot(epoch_list, acc_list)

            # print some examples to a folder
            save_predictions_as_imgs(
                testloader, model, folder="saved_images_lung{}".format(str(fold)), device=DEVICE
            )
        plt.plot(epoch_list, jaccord_list,precision_list,recall_list)
        # plt.plot(epoch_list, dice_list)
        plt.show()
        #
        print("------------Mean-----------")
        print(f"Mean Accuracy:{mean(acc_list)}")
        print(f"Mean Jaccord:{mean(jaccord_list)}")
        print(f"Mean Dice:{mean(dice_list)}")
        print(f"Mean Precision:{mean(precision_list)}")
        print(f"Mean Recall:{mean(recall_list)}")
        print(f"Mean F1-score:{mean(f_score_list)}")
        print(f"Mean Specificity:{mean(specificity_list)}")
        print(f"Mean AUC:{mean(auc_list)}")
        total_acc_list.append(mean(acc_list))
        total_dice_list.append(mean(dice_list)  )
        total_jaccord_list.append(   mean(jaccord_list) )
        total_precision_list.append( mean(precision_list) )
        total_recall_list.append(    mean(recall_list))
        total_f_score_list.append(   mean(f_score_list) )
        total_specificity_list.append( mean(specificity_list) )
        total_auc_list.append( mean(auc_list) )
        print("------------MAX-----------")
        print(f"Max Accuracy:{max(acc_list)}")
        print(f"Max Jaccord:{max(jaccord_list)}")
        print(f"Max Dice:{max(dice_list)}")
        print(f"Max Precision:{max(precision_list)}")
        print(f"Max Recall:{max(recall_list)}")
        print(f"Max F1-score:{max(f_score_list)}")
        print(f"Max Specificity:{max(specificity_list)}")
        print(f"Max AUC:{max(auc_list)}")



    print("----------TOTAL------------")
    print( f"Tatal_Mean Accuracy:{mean(  total_acc_list )}" )
    print( f"Tatal_Mean Jaccord:{mean(   total_jaccord_list )}" )
    print( f"Tatal_Mean Dice:{mean(      total_dice_list )}" )
    print( f"Tatal_Mean Precision:{mean( total_precision_list )}" )
    print( f"Tatal_Mean Recall:{mean(    total_recall_list )}" )
    print( f"Tatal_Mean F1-score:{mean(  total_f_score_list )}" )
    print( f"Tatal_Mean Specificity:{mean( total_specificity_list )}" )
    print( f"Tatal_Mean AUC:{mean( total_auc_list )}" )
    print("LEARNING_RATE=",LEARNING_RATE)
    if '/' in str(TRAIN_IMG_DIR):
        data = TRAIN_IMG_DIR.split("/", 2)[1]
    print("{}".format(str(data)))
    if '(' in str(model):
        models = str(model).split("(", 1)[0]
    print("{}".format(str(models)))
if __name__ == "__main__":
    main()


