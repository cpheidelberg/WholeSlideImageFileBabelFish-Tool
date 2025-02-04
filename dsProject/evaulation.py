
import os
import glob
import pandas as pd

#%%
import sys
import pandas as pd

base_bath = "./path/to/data/"

dataset = "train"
img_path = base_bath + dataset
manifest = pd.read_excel(img_path + "/manifest_" + dataset + ".xlsx", engine='openpyxl',
                         sheet_name="ground truth")

prediction = pd.read_excel(img_path + "/manifest_" + dataset + ".xlsx", engine='openpyxl',
                            sheet_name="results")

#%% plot the statitics
print(f"prefix types {manifest.prefix.value_counts()}")
print(f"suffix types {manifest.suffix.value_counts()}")
#print(f"suffix types {manifest.stain.value_counts()}")

#%%
import numpy as np

def count_numbers(string):
    count = 0
    for char in string:
        if char.isdigit():  # Check if the character is an alphabet character
            count += 1
    return count

def count_characters(string):
    count = 0
    for char in string:
        if char.isalpha():  # Check if the character is an alphabet character
            count += 1
    return count

n = np.sum([count_numbers(i) > 0 for i in manifest.block])
n = np.sum([i != "A 1" for i in manifest.block])

#%%
import matplotlib.pyplot as plt
stain = manifest.stain.value_counts()
stain = stain.reset_index()
stain.columns = ['stain', 'n']
stain =stain.sort_values('n', ascending = False)

df = stain[:4].copy()

new_row = pd.DataFrame(data = {
    'stain' : ['others'],
    'n' : [stain['n'][5:].sum()]
})

df = pd.concat([df, new_row])

plt.pie(df.n, labels = df.stain)
plt.axis('equal')
plt.title("testomg data set")
plt.savefig("pie_chart.png")
plt.show()
