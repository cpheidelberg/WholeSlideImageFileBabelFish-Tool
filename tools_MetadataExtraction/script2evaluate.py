
#%% import section
import sys
import pickle
import os
import matplotlib.pyplot as plt
import cv2
from tools_MetadataExtraction.HitchhikersGuide import HitchhickerGuide
from tqdm import tqdm
import glob
import pandas as pd
from tools_MetadataExtraction.utils import clean_up

verbose = False
dataset = "test"
img_path = "/path/to/datasets/" + dataset
if verbose:
    error_df = pd.read_csv("tools_MetadataExtraction/error_list.csv")
    img_list = error_df.error_list.tolist()
else:
    img_list = glob.glob(os.path.join(img_path, '*.png'))

manifest = pd.read_excel(img_path + "/manifest_" + dataset + ".xlsx", engine='openpyxl',
                         sheet_name="ground truth")
manifest.case = manifest.case.astype(str)

#%% loop of them
prefix, case, suffix, block, stain, file = [],[],[],[],[],[]
e = 0
error_list, error_id, error_id_gt = [],[],[]

for i, i_img in enumerate(tqdm(img_list, desc=dataset + " images")):

    i_gt = manifest[manifest.file == i_img].astype(str)
    id_gt = i_gt.prefix + "/" + i_gt.suffix + "/" + i_gt.case + "/" + i_gt.block + "/" + i_gt.stain
    id_gt = id_gt.values[0]

    Test = HitchhickerGuide(i_img)  # known HE

    #Test.get_slideID()
    id = Test.fileID

    prefix.append(Test.prefix)
    suffix.append(Test.suffix)
    case.append(Test.case)
    block.append(Test.block)
    file.append(i_img)
    stain.append(Test.stain)

    fig, axs = plt.subplots(ncols=2, nrows=1)
    plt.suptitle(f"words found {Test.words_found} and \n id generated from it {id} \n id correct {id_gt}")
    axs[0].imshow(cv2.imread(i_img))
    axs[0].set_title("macro")
    Test.TextFound._plot(ax=axs[1])
    axs[1].set_title("label")

    clean_up(i_img)
    if id == id_gt:
        save_path = i_img[0:-4] + "_label.jpg"
    else:
        save_path = i_img[0:-4] + "_error.jpg"
        e += 1
        error_list.append(i_img)
        error_id.append(id)
        error_id_gt.append(id_gt)

    plt.savefig(save_path)
    if id != id_gt:
        plt.show()

    plt.close()

    print("_" * 25)
    print(f"words found {Test.TextFound.finding.tolist()}")

    if id == id_gt:
        print(f"\033[32mand id generated from it {id}\033[0m")
    else:
        print(f"\033[31mand id generated from it {id}\033[0m")
        print(f"\033[31mand id expected would have been {id_gt}\033[0m")
        print(f"occured for file {i_img}")
        print(f"now the error count is {e}")
    print("_" * 25)

print(f"n={e} errors for n={len(img_list)} examples")

if not verbose:
    with open("tools_MetadataExtraction/error_list.pkl", "wb") as f:
        pickle.dump(error_list, f)
    error_df = pd.DataFrame({"error_list":error_list, "id_pred": error_id, "id_gt": error_id_gt})
    error_df.to_csv("tools_MetadataExtraction/error_list.csv")

#%% mount it back and save it
df = pd.DataFrame({"file": file,
                   "prefix_pred": prefix, "case_pred": case, "suffix_pred": suffix,
                   "block_pred": block, "stain_pred": stain,
                   })
df = pd.merge(manifest, df, on='file', how='right')

with pd.ExcelWriter(img_path + "/manifest_" + dataset + ".xlsx", 'openpyxl', mode='a', if_sheet_exists="replace") as writer:
    # fix line
    #writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
    df.to_excel(writer, "results")

#%% compare the results to the ground truth
import numpy as np
df_pred = pd.DataFrame({"file": file,
                   "prefix": prefix, "case": case, "suffix": suffix,
                   "block": block, "stain": stain,
                   }).astype(str).sort_values(by='file')
matrix_pred = df_pred.values
df_gt = manifest.drop(columns=['Spalte1', 'comment']).sort_values(by='file')
df_gt['suffix'] = [str(int(i)) for i in df_gt['suffix']]
matrix_gt = df_gt.values

accuracy = (np.sum(matrix_gt == matrix_pred) - matrix_gt.shape[0]) / (matrix_gt.shape[0] * (matrix_gt.shape[1] -1))
print(f"accuracy is {accuracy}")
df = pd.DataFrame(matrix_gt == matrix_pred, columns=df_gt.columns)
df['files'] = df_gt.file

with pd.ExcelWriter(img_path + "/manifest_" + dataset + ".xlsx", 'openpyxl', mode='a', if_sheet_exists="replace") as writer:
    # fix line
    #writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
    df.to_excel(writer, "comparison")

