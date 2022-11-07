import torch
import torchvision
import torch.nn as nn
from dataset import LungDataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import auc
import  numpy as np

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
):
    train_ds = LungDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        shuffle=True,
    )

    val_ds = LungDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        shuffle=False,
    )

    return train_loader, val_loader


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    TP=0
    FP=0
    FN=0
    TN=0
    tp= 0
    fp = 0
    fn = 0
    tn = 0
    auc = 0
    f_score = 0
    specificity = 0
    dice_score = 0
    jaccord_score = 0
    recall = 0
    precision = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            x = model(x)
            x = torch.tensor(x)
            preds = torch.sigmoid(x)
            preds = (preds > 0.5).float()
            #auc = roc_auc_score(y[:, 0], preds[:, 0])
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            TP = (preds * y).sum()
            TN = num_correct - (preds * y).sum()
            FP = (preds - preds * y).sum()
            FN = (y - preds * y).sum()
            tp += (preds * y).sum()
            tn += num_correct - tp
            fp += (preds - preds * y).sum()
            fn += (y - preds * y).sum()
            # auc = metrics.auc(fpr, tpr)
            a=y.cpu().numpy() # 标签tensor转为list
            b=preds.cpu().numpy() # 预测tensor转为list
            aa = list(np.array(a).flatten()) # 高维转为1维度
            bb = list(np.array(b).flatten()) # 高维转为1维度
            auc = metrics.roc_auc_score(aa, bb,multi_class='ovo')
            precision += TP / ((TP + FP) + 1e-8)
            recall += TP / ((TP + FN) + 1e-8)
            f_score += (2 * TP) / ((FP + 2 * TP + FN) + 1e-8)
            specificity += TN / ((TN + FP) + 1e-8)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            jaccord_score += (preds * y).sum() / ((preds + y).sum() + 1e-8 - (preds * y).sum())

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.4f}")
    print(f"Dice score: {dice_score / len(loader)}")
    print(f"Jaccord score: {jaccord_score / len(loader)}")
    print(f"Precision:{precision / len(loader)}")
    print(f"Recall:{recall / len(loader)}")
    print(f"F1-score:{f_score / len(loader)}")
    print(f"Specificity:{specificity / len(loader)}")
    print(f"AUC:{auc}")
    print(f"TP total :{tp}")
    print(f"TN total:{tn}")
    print(f"FP total:{fp}")
    print(f"FN total:{fn}")


    # print(type(num_correct/num_pixels*100))
    model.train()

    return (num_correct / num_pixels * 100).cpu().numpy(), (dice_score / len(loader)).cpu().numpy(), (
                jaccord_score / len(loader)).cpu().numpy(), (precision / len(loader)).cpu().numpy(), (
                       recall / len(loader)).cpu().numpy(), (f_score / len(loader)).cpu().numpy(), (
                       specificity / len(loader)).cpu().numpy(), auc, tp.cpu().numpy(),tn.cpu().numpy(), fp.cpu().numpy(), fn.cpu().numpy()




def save_predictions_as_imgs(
        loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/lable_{idx}.png")
        torchvision.utils.save_image(x, f"{folder}/image_{idx}.png")

    model.train()