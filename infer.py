import torch
import glob
import torchvision.transforms as transforms
from PIL import Image
from model import ChestXRayCNN  # Import your CNN model class
import os
# from omegaconf import DictConfig
# import hydra
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import config


def confusion(ans,preds,acc,dic):
    cm = confusion_matrix(y_true=np.asarray(ans), y_pred=np.asarray(preds))
    cmn = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    sns.heatmap(cmn*100, cmap='Blues', annot=True, fmt='.2f', xticklabels=dic.values(), yticklabels=dic.values())
    ax.xaxis.set_label_position("bottom")
    plt.setp(ax.get_yticklabels())
    plt.setp(ax.get_xticklabels())
    # plt.tight_layout()
    # plt.title("HTS-AT " +config.mode_type+ " Multi Model - Confusion Matrix "+config.dataset_type+ " Tested on RAVDESS ACE - Accuracy: " + str(round(acc,4))+"%")
    plt.title( "Chest-XRay Accuracy: " + str(round(acc*100,1))+"%")
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig("/home/dsi/ohadico97/chest-XRay/cm.png",dpi=300)
    
    

# @hydra.main(config_name='train_cfg')
# def test(cfg: DictConfig) -> None:
def test():
    # Load trained model
    model = ChestXRayCNN.load_from_checkpoint('/home/dsi/ohadico97/chest-XRay/checkpoints/Xray.ckpt',  config = config)
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    print('✅ Model loaded')

    # Define image transformations (same as training)
    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    # Define class labels
    classes = ['NORMAL', 'PNEUMONIA']


    img_paths = []
    labels = []

    for label, class_name in enumerate(classes):
        class_dir = os.path.join("/home/dsi/ohadico97/chest-XRay/data_copy/test", class_name)
        if os.path.exists(class_dir):
            for img_file in glob.glob(os.path.join(class_dir, "*")):
                img_paths.append(img_file)
                labels.append(label)

    # Zip images with labels
    image_label_pairs = list(zip(img_paths, labels))

    pred_list = []
    label_list = []

    for img, lbl in image_label_pairs:
        # Load image
        image = Image.open(img).convert('RGB')
        image = transform(image).unsqueeze(0).to(model.device)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            y_hat = model(image)
            pred = int(torch.argmax(y_hat, dim=-1))
        pred_list.append(pred)
        label_list.append(lbl)
        print(f'📌 Image: {img} | Prediction: {classes[pred]}')

    acc = accuracy_score(label_list, pred_list)
    dic = {0:"NORMAL",1:"Pneumonia"}
    
    print(classification_report(label_list, pred_list, target_names=list(dic.values())))
    print(acc)
    confusion(label_list,pred_list,acc,dic)
if __name__ == "__main__":
    test()