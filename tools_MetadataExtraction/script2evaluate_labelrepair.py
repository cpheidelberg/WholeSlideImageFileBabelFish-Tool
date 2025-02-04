
#%% import section
import sys
import os
import matplotlib.pyplot as plt
import cv2
from tools_MetadataExtraction.HitchhikersGuide import HitchhickerGuide
from tqdm import tqdm
import glob
import pandas as pd

methods = ["SmallPortion", 'conventional']
dataset = "test"
img_path = "/path/to/datasets/" + dataset
img_list = glob.glob(os.path.join(img_path, '*.png'))

manifest = pd.read_excel(img_path + "/manifest_" + dataset + ".xlsx", engine='openpyxl',
                         sheet_name="ground truth")
manifest.case = manifest.case.astype(str)

#%% loop of them
prefix, case, suffix, block, stain, file = [],[],[],[],[],[]
e = 0
error_list, error_id, error_id_gt = [],[],[]

for i, i_img in enumerate(tqdm(img_list, desc= dataset + " images")):

    i_gt = manifest[manifest.file == i_img].astype(str)
    id_gt = i_gt.prefix + "/" + i_gt.suffix + "/" + i_gt.case + "/" + i_gt.block + "/" + i_gt.stain
    id_gt = id_gt.values[0]

    label_noisy = cv2.imread(i_img)

    plt.imshow(label_noisy)
    plt.title("noisy macro image")
    plt.show()

    try:
        Test = HitchhickerGuide(label_noisy,
                                methods=methods)  # known HE
        id = Test.fileID
        prefix.append(Test.prefix)
        suffix.append(Test.suffix)
        case.append(Test.case)
        block.append(Test.block)
        file.append(i_img)
        stain.append(Test.stain)

        if id == id_gt:
            print(f"\033[32mand id generated from it {id}\033[0m")
        else:
            e += 1
            print(f"\033[31mand id generated from it {id}\033[0m")
            print(f"\033[31mand id expected would have been {id_gt}\033[0m")
            print(f"occured for file {i_img}")
            print(f"now the error count is {e} for {i + 1} samples")

    except:
        id = "Error "
        prefix.append('None')
        suffix.append('None')
        case.append('None')
        block.append('None')
        file.append(i_img)
        stain.append('None')
        e += 1
        print(f"\033[31mand id generated from it {id}\033[0m")
        print(f"\033[31mand id expected would have been {id_gt}\033[0m")
        print(f"occured for file {i_img}")
        print(f"now the error count is {e} for {i + 1} samples")

    print("_" * 25)

print(f"n={e} errors for n={len(img_list)} examples")

#%% mount it back and save it
import numpy as np
df_pred = pd.DataFrame({"file": file,
                   "prefix_pred": prefix, "case_pred": case, "suffix_pred": suffix,
                   "block_pred": block, "stain_pred": stain,
                   })
df_pred = pd.merge(manifest, df_pred, on='file', how='right')
df_pred = df_pred[["file", "prefix_pred", "case_pred", "suffix_pred", "block_pred", "stain_pred"]]
df_pred.columns = ["file", "prefix", "case", "suffix", "block", "stain"]
df_pred = df_pred.sort_values(by = "file")
matrix_pred = df_pred.values
df_gt = manifest.drop(columns=['Spalte1', 'comment']).sort_values(by='file')
df_gt['suffix'] = [str(int(i)) for i in df_gt['suffix']]
matrix_gt = df_gt.values

accuracy = (np.sum(matrix_gt == matrix_pred) - matrix_gt.shape[0]) / (matrix_gt.shape[0] * (matrix_gt.shape[1] -1))
print(f"accuracy is {accuracy}")
df = pd.DataFrame(matrix_gt == matrix_pred, columns=df_gt.columns)
df['files'] = df_gt.file

if "SmallPortion" in methods:
    file_name = "ImageInpainting_SmallPortionErosion.xlsx"
if "LargePortion" in methods:
    file_name = "ImageInpainting_LargePortion.xlsx"

#%%
from tools_MetadataExtraction.utils import find_highest_number
excel_file = pd.ExcelFile(img_path + "/" + file_name)

sheet_names = excel_file.sheet_names

if not "results#1" in sheet_names:
    page = 1
else:
    page = int(find_highest_number(sheet_names) + 1)

page = str(page)

#%%
with pd.ExcelWriter(img_path + "/" + file_name, 'openpyxl', mode='a', if_sheet_exists="replace") as writer:
    # fix line
    #writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
    df.to_excel(writer, "comparison#" + page)

with pd.ExcelWriter(img_path + "/" + file_name, 'openpyxl', mode='a', if_sheet_exists="replace") as writer:
    # fix line
    #writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
    df_pred.to_excel(writer, "results#" + page)

#%% compare the results to the ground truth



