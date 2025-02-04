
#%% setting
import pandas as pd
import numpy as np

data = "Ball"

#%%
if data == "Large":
    file_name = "ImageInpainting_LargePortion.xlsx"
if data == "Erosion":
    file_name = "ImageInpainting_SmallPortionErosion.xlsx"
if data == "Ball":
    file_name = "ImageInpainting_SmallPortionBall.xlsx"

a = []
for i in range(1, 4):
    manifest = pd.read_excel("tools_MetadataExtraction/label_repair/" + file_name,
                             sheet_name="comparison#" +str(i))

    d = (manifest.prefix * manifest.case * manifest.suffix * manifest.block * manifest.stain).to_list()
    a.append(sum(d) / len(d))

print(f" mean accurarcy is  {np.mean(a)} +/- {np.std(a)}")

