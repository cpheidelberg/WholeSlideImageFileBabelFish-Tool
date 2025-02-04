#%% import section
import numpy as np
from tools_MetadataExtraction.utils_image import sharpen_image
import torch
import cv2
from tools_MetadataExtraction.label_repair.models.FastSCNN import FastSCNN # used in LargeHealingPortion
import os
from skimage import restoration, util

def image2tensor(img):

    if img.dtype == np.uint8 and np.max(img) > 1:
        img = img/255
    img = torch.from_numpy(img)

    if len(img.shape) == 3:
        img = img.permute(2, 0, 1)

    if img.dtype == torch.int8:
        img = img.long()
    else:
        img = img.float()
    return img

def tensor2image(tensor):
    tensor = tensor.squeeze()

    if len(tensor.shape) ==3:
        tensor = tensor.permute(1,2,0)

    image = tensor.detach().cpu().numpy()

    return image

#%% augmentation class
def augment_by_combination(label, mask):

    mask = 1-mask

    if len(mask.shape) == 2:
        mask = np.stack((mask, mask, mask))
        mask = np.transpose(mask, (1,2,0))

    if np.max(label) != np.max(mask):
        mask = mask * 255 #np.max(label) asume normal image

    mask = mask.astype(np.uint8)
    if not (label.shape == mask.shape):
        mask = cv2.resize(mask, (label.shape[1],label.shape[0]))

    mask = mask.astype(np.float64)
    label = label.astype(np.float64)
    label_combined = (label + mask) / 2
    label_combined = label_combined.astype(np.uint8)

    return label_combined

#%% Label class
class Label():
    def __init__(self, label_image):

        if isinstance(label_image, str):
            img = cv2.imread(label_image)
        else:
            img = label_image
        self.label_image = img #cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def get_label(self):
        return self.label_image

#%% define healing portion class
class SmallHealingPortion():
    def __init__(self,method = "erosion"):

        self.method = method

    def heal(self, label2repair):

        img = sharpen_image(label2repair)

        if self.method == "ball":
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray = util.invert(img_gray)
            background = restoration.rolling_ball(img_gray)
            img_gray_restored = img_gray - background
            img_gray_restored= util.invert(img_gray_restored)
            label_repaired = cv2.cvtColor(img_gray_restored, cv2.COLOR_GRAY2RGB)

        elif self.method == "erosion":

            mask = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 50).astype(np.uint8)
            kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 4))
            kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 4))
            mask = cv2.medianBlur(mask, 3)
            mask = cv2.erode(mask, kernel1)
            mask = cv2.erode(mask, kernel2)
            label_repaired = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2RGB)
            #label_repaired = augment_by_combination(img, mask)

        return label_repaired

#%% define healing portion class
class LargeHealingPortion:
    def __init__(self, model="default", resize_factor = 5, device="cpu"):

        if isinstance(model, str):

            if os.path.exists(model):
                model_path = model
            else:
                model_path = f"./tools_MetadataExtraction/label_repair/models/FastSCNN.pt"

            model = FastSCNN(in_channels=3, num_classes=2)

            state_dict = torch.load(model_path,
                                    map_location=torch.device('cpu'))
            state_dict = state_dict['model_state_dict']
            model.load_state_dict(state_dict)
            model.eval()

        self.model = model
        if device == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)

        self.resize_factor = resize_factor

    def heal(self, label2repair):

        self._Label = Label(label2repair)
        label2repair = self._Label.get_label()

        label2repair = cv2.resize(label2repair,
                                  (self.resize_factor * label2repair.shape[1], self.resize_factor * label2repair.shape[0]))

        output = self.model(image2tensor(label2repair).unsqueeze(0).to(self.device))
        #print(np.max(image2tensor(label2repair)))

        if len(output.shape) == 4:
            _, output = output.max(dim=1)
        elif len(output.shape) == 3:
            _, output = output.max(dim=0)
        label_repaired = np.uint8(tensor2image(output.cpu()))

        self.label2repair = label2repair
        self.label_repaired = label_repaired

        label_repaired = augment_by_combination(label2repair, label_repaired)

        label_repaired= cv2.resize(label_repaired,
                                  (label_repaired.shape[1]//self.resize_factor, label_repaired.shape[0]//self.resize_factor))

        return label_repaired

#%% test it
if __name__ == "__main__":

    #%% load the image
    import matplotlib.pyplot as plt
    test_image = "tools_MetadataExtraction/label_repair/Macro.png"
    test_label = cv2.imread(test_image)
   #test_label = cv2.resize(test_label, (4* test_label.shape[1], 4* test_label.shape[0]))


    #%% test small portion
    Portion = SmallHealingPortion(method = "erosion")
    repaired_image = Portion.heal(test_label)
    plt.subplot(131)
    plt.imshow(test_label)
    plt.title("original")
    plt.subplot(132)
    plt.imshow(test_label)
    plt.title("input")
    plt.subplot(133)
    plt.imshow(repaired_image)
    plt.title("repaired by small portion")
    plt.savefig("tools_MetadataExtraction/label_repair/smallHealingPortion.png")
    plt.show()

    # %% test large portion
    Portion = LargeHealingPortion()
    repaired_image = Portion.heal(test_label)
    plt.subplot(221)
    plt.imshow(test_label)
    plt.title("original")
    plt.subplot(222)
    plt.imshow(test_label)
    plt.title("input")
    plt.subplot(223)
    plt.imshow(Portion.label_repaired)
    plt.colorbar()
    plt.title("mask")
    plt.subplot(224)
    plt.imshow(repaired_image)
    plt.title("repaired by large portion")
    plt.savefig("tools_MetadataExtraction/label_repair/largeHealingPortion.png")
    plt.show()
