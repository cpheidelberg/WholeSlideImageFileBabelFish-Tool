
#%%
import pandas as pd

dataset = "test"
manifest = pd.read_excel("tools_MetadataExtraction/manifest_" + dataset + ".xlsx",
                         sheet_name="comparison")

#%% count
n = sum(manifest.prefix)
print(f" prefix n = {n} for total {len(manifest)} cases")
print(f" accurarcy is  {n/len(manifest)}")

n = sum(manifest.case)
print(f" case ID n = {n} for total {len(manifest)} cases")
print(f" accurarcy is  {n/len(manifest)}")

n = sum(manifest.suffix)
print(f" suffix n = {n} for total {len(manifest)} cases")
print(f" accurarcy is  {n/len(manifest)}")

n = sum(manifest.block)
print(f" block ID n = {n} for total {len(manifest)} cases")
print(f" accurarcy is  {n/len(manifest)}")

n = sum(manifest.stain)
print(f" stain n = {n} for total {len(manifest)} cases")
print(f" accurarcy is  {n/len(manifest)}")

n = sum(manifest.prefix * manifest.case * manifest.suffix * manifest.block * manifest.stain)
print(f" entire set = {n} for total {len(manifest)} cases")
print(f" accurarcy is  {n/len(manifest)}")