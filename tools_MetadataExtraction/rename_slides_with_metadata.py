import os
import json
import pandas as pd

def make_hyperlink_and_preprocess(value, appeareance=None):
    value = value.replace(".import", ".png")
    if appeareance is None:
        appeareance = value
    return '=HYPERLINK("{}","{}")'.format(value, appeareance)

def rename_slide_files(files_dir: str, meta_dir: str, slide_postfix=".ndpi", meta_postfix=".ndpi.import",
                       meta_keys_for_renaming=["SlideId", "ExamId"], append_old_names_to_new_ones=True, armed=False):
    '''
    Renames all slide files in files_dir using the metadata files in meta_dir.
    :param files_dir:
    :param meta_dir:
    :param slide_postfix:
    :param meta_postfix:
    :param meta_keys_for_renaming: The first key in this list that is found in the metadata file will be used as the new name.
    :param append_old_names_to_new_ones: If True, the old name will be appended to the new name in brackets.
    :param armed: You can run this function in an unarmed mode (armed=False) to see what would happen without actually renaming the files. (see returned renaming-dict)
    :return: returns a dict with the old names, new names and metadata files of the renamed files.
    '''

    # Hole alle .ndpi und .json Dateien
    ndpi_files = {f for f in os.listdir(files_dir) if f.endswith(slide_postfix)}
    json_files = {f for f in os.listdir(meta_dir) if f.endswith(meta_postfix)}

    dict_renamed_files = {"old_name": [], "new_name": [], "metadata": []}

    # Gehe durch jedes .ndpi File und suche das passende .json File
    for ndpi_file in ndpi_files:
        json_file = ndpi_file.replace(slide_postfix, meta_postfix)
        if json_file in json_files:
            json_path = os.path.join(meta_dir, json_file)
            ndpi_path = os.path.join(files_dir, ndpi_file)

            # Lade das JSON File
            with open(json_path, "r", encoding="utf-8") as f:
                metadata_of_file = json.load(f)

            # Hole den neuen Dateinamen aus metadata_of_file
            new_name = None
            for k in meta_keys_for_renaming:
                new_name = metadata_of_file.get(k)
                if new_name:
                    break
            if new_name:
                if append_old_names_to_new_ones:
                    new_name = f"{new_name}({os.path.splitext(os.path.basename(ndpi_path))[0]})"
                new_ndpi_path = os.path.join(files_dir, f"{new_name}{slide_postfix}")
                if armed:
                   os.rename(ndpi_path, new_ndpi_path)
                #print(f"Renamed: {ndpi_path} -> {new_ndpi_path}")
                dict_renamed_files["old_name"].append(make_hyperlink_and_preprocess(ndpi_path))
                dict_renamed_files["new_name"].append(make_hyperlink_and_preprocess(new_ndpi_path))
                dict_renamed_files["metadata"].append(make_hyperlink_and_preprocess(json_path))
            else:
                print(f"Warning: No {meta_keys_for_renaming} found in {json_file}")
                dict_renamed_files["old_name"].append(make_hyperlink_and_preprocess(ndpi_path))
                dict_renamed_files["new_name"].append(None)
                dict_renamed_files["metadata"].append(make_hyperlink_and_preprocess(json_path))

    return dict_renamed_files

def main():
    files_to_rename_dir = "D:\Research\Slides\LuFi"
    meta_data_dir = "D:\Research\Slides\LuFi\meta-extraction-results"
    armed = False

    renamed_files = rename_slide_files(files_to_rename_dir, meta_data_dir, armed=armed)

    df = pd.DataFrame(renamed_files)
    df.to_excel(files_to_rename_dir + "/renamed_files.xlsx", index=False)

    print("Done.")
    print(f"Renamed files saved to {files_to_rename_dir}/renamed_files.csv")
    print(df)


if __name__ == "__main__":
    main()