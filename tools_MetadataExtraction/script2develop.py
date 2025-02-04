
#%%
import sys
import matplotlib
if sys.platform == 'darwin': # on macos, this value is "darwin", on linux it is "linux", and on windows it is "win32"
    matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import easyocr
from tqdm import tqdm
from PIL import Image

#%% define label class

def repair(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
             cv2.THRESH_BINARY_INV,9,11)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    dilate = cv2.dilate(close, kernel, iterations=1)
    result = 255 - dilate
    return result+gray

def repair(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Create a CLAHE object (Arguments are optional)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Apply CLAHE to the image
    equalized_image = clahe.apply(gray)

    return equalized_image

class Label():
    def __init__(self, label_image):

        if isinstance(label_image, str):
            img = cv2.imread(label_image)
        if isinstance(label_image, Image.Image):
            img = np.array(label_image)

        img = cv2.resize(img, (2 * img.shape[1], 2 * img.shape[0]))
        #img = enhance_contrast(img)
        #img = sharpen_image(img)
        #img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = augment_by_combination(img, img < 100)
        self.label_image = img

    def get_label(self):
        return self.label_image[0:600, 0:-1]

#%% test it
img_list = glob.glob(os.path.join("tools_MetadataExtraction/test_label_images", '*.png'))
from tools_MetadataExtraction.label_repair.HealingPortion import SmallHealingPortion, LargeHealingPortion
SmallPortion = SmallHealingPortion(method = "ball")
LargePortion = LargeHealingPortion()
from tools_MetadataExtraction.HitchhikersGuide import HitchhickerGuide

for i, i_img in enumerate(tqdm(img_list, desc="test images")):
    macro_image = Label(i_img)
    label = macro_image.get_label()

    label = SmallPortion.heal(label)
    #label2 = LargePortion.heal(label)
    #label = augment_by_combination(label1, label2)
    reader = easyocr.Reader(['en', 'de'])
    extract_info = reader.readtext(label, paragraph=False)
    words_found = [i_finding[1] for i_finding in extract_info]

    words_found = " ".join(words_found)
    Test = HitchhickerGuide(label)  # known HE
    Test.get_slideID()
    id = Test.fileID
    title = f" file #{i} / {i_img} \n words found: {words_found} \n id mounted: {id}"

    #Test.SlideLabel.slide_label

    plt.figure()
    plt.suptitle(title)
    plt.subplot(121)
    plt.imshow(cv2.imread(i_img))
    plt.title("macro")
    plt.subplot(122)
    plt.imshow(Test.SlideLabel.slide_label)
    plt.title("label")
    plt.show()
    plt.tight_layout()
    plt.pause(0.1)

#%%
