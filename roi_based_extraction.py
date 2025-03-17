import os, sys
from tqdm import tqdm
import yaml
import json
from tools_MetadataExtraction.RoiBasedMetaDataExtractor import RoiBasedMetaDataExtractor, regex_check
import filecmp
import datetime

# script params:
save_print_to_log_file = True

# argument parsing:
import argparse
parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--openslide_dll', type=str, required=False, default=None, help='Path to the OpenSlide DLL directory')
parser.add_argument('--dmxt_dll', type=str, required=False, default=None, help='Path to the libdmxt_dll DLL directory')
parser.add_argument('--config', type=str, required=True, help='the .yaml config file (use config/roi_config_example.yaml as template for your configuration!)')

# main function:
def main():

    args = parser.parse_args()

    conf_data_path = args.config

    # load config from yaml file:
    with open(conf_data_path, 'r') as file:
        config = yaml.safe_load(file)

    log_folder = ".logs"
    if not os.path.exists(config["folder_to_watch"] + f"/{log_folder}"):
        os.makedirs(config["folder_to_watch"] + f"/{log_folder}")

    # redirect prints into a log file:
    if save_print_to_log_file:
        lof_file = config["folder_to_watch"] + f"/{log_folder}/meta-extraction-log.log"
        print(f"!!! Redirecting print output to {lof_file}. So please check this file for the output !!!")
        log = open(lof_file, "a")
        sys.stdout = log

    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"=== {time_stamp} ===")

    folder_to_watch = config['folder_to_watch']
    debug_mode = config['debug_mode']

    if config['rename_wsi_files']:
        if not config['renaming_pattern']:
            raise ValueError("If you want to rename the WSI files, you have to provide a renaming pattern.")

    print(f"Using config file: {conf_data_path}, with configuration:")
    for k_config in config:
        print(f"  {k_config}: {config[k_config]}")



    # precompute the renaming pattern as a list:
    renaming_pattern_list = []  # should be a list of shape [meta_key, seperation_symbol, meta_key, seperation_symbol, ..., meta_key]
    if config['rename_wsi_files']:
        assert '{' in config['renaming_pattern'] and '}' in config['renaming_pattern'], 'renaming_pattern must contain at least one pair of curly brackets!'
        for start_word in config['renaming_pattern'].split('{'):
            if start_word == '':
                continue

            if len(start_word.split('}')) == 1:
                meta_key = start_word.split('}')[0]
                assert meta_key in [roi_name for roi_name in config['extraction_rules']], \
                    f"Meta key '{meta_key}', which appears in the renaming_pattern, not found in extraction rules!"
                if meta_key != '':
                    renaming_pattern_list += [meta_key]
            elif len(start_word.split('}')) == 2:
                meta_key = start_word.split('}')[0]
                assert meta_key in [roi_name for roi_name in config['extraction_rules']], \
                    f"Meta key '{meta_key}', which appears in the renaming_pattern, not found in extraction rules!"
                seperation_symbol = start_word.split('}')[1]
                renaming_pattern_list += [meta_key, seperation_symbol] if seperation_symbol else [meta_key]
            else:
                raise ValueError(f"Renaming pattern '{config['renaming_pattern']}' is not supported!")

        print(f"Renaming pattern list: {renaming_pattern_list}")

    roi_extractor = RoiBasedMetaDataExtractor(config['ROI_set_file'], config=config['extraction_rules'],
                                              wsi_macro_img_tag=config['wsi_macro_img_tag'],
                                              wsi_macro_img_rotation=config['wsi_macro_img_rotation'],
                                              ROI_set_ref_res=config['ROI_set_ref_res'] if 'ROI_set_ref_res' in config else None,)

    plausibility_regex_checks = {roi_name: config['extraction_rules'][roi_name]['plausibility_regex_check']
                                 for roi_name in config['extraction_rules']}

    wsi_files = []
    for file_type in config['wsi_types']:
        wsi_files += [os.path.join(folder_to_watch, f) for f in os.listdir(folder_to_watch) if f.endswith(file_type)]
    for wsi_file in tqdm(wsi_files):

        print(f"== {time_stamp} ==")

        # skip if wsi_file is already processed (if renaming_pattern can be found in filename and if values are plausible)
        if config['rename_wsi_files'] and not config['force_meta_extraction']:
            wsi_name = os.path.basename(wsi_file)
            fyle_type = os.path.basename(wsi_file).split('.')[-1]
            wsi_name = wsi_name.replace(fyle_type, '')
            if '{' in wsi_name and '}' in wsi_name:
                if len(wsi_name.split('{')) == len(wsi_name.split('}')) and len(wsi_name.split('{')) == len(config['renaming_pattern'].split('{')):
                    print(f"Skipping '{wsi_name}' as it is already processed (renaming pattern '{config['renaming_pattern']}' matches with '{wsi_name}').")
                    continue

        both_results = {}
        merged_result = {}
        for ocr_engine in ["pytesseract", "easyocr"]:
            both_results[ocr_engine] = roi_extractor.extract_metadata_with_roiset(wsi_file, debug_mode=debug_mode,
                                                                                  ocr_engine=ocr_engine)

        used_ocr_engine = {}
        errors = {"pytesseract": {}, "easyocr": {}}
        for roi_name in both_results["pytesseract"]:

            errors["pytesseract"][roi_name] = regex_check(both_results["pytesseract"][roi_name],
                                                          plausibility_regex_checks[roi_name])
            errors["easyocr"][roi_name] = regex_check(both_results["easyocr"][roi_name],
                                                      plausibility_regex_checks[roi_name])

            if errors["pytesseract"][roi_name] and errors["easyocr"][roi_name]:
                merged_result[roi_name] = None
            elif errors["pytesseract"][roi_name] and not errors["easyocr"][roi_name]:
                merged_result[roi_name] = both_results["easyocr"][roi_name]
                used_ocr_engine[roi_name] = "easyocr"
            elif errors["easyocr"][roi_name] and not errors["pytesseract"][roi_name]:
                merged_result[roi_name] = both_results["pytesseract"][roi_name]
                used_ocr_engine[roi_name] = "pytesseract"
            else:
                if both_results["pytesseract"][roi_name] == both_results["easyocr"][roi_name]:
                    merged_result[roi_name] = both_results["pytesseract"][roi_name]
                    used_ocr_engine[roi_name] = "both"
                else:
                    print(f"Warning: OCR engines did not return the same result for {roi_name} in {wsi_file}:")
                    print(f"  pytesseract: {both_results['pytesseract'][roi_name]}")
                    print(f"  easyocr: {both_results['easyocr'][roi_name]}")
                    print("However, both results are plausible. Using pytesseract result.")
                    merged_result[roi_name] = both_results["pytesseract"][roi_name]
                    used_ocr_engine[roi_name] = "pytesseract"

        if any(merged_result[k] is None for k in merged_result):
            print(f"Plausibility check failed for {wsi_file}:")
            print(errors)
            # store the error in a json file:

            with open(f"{wsi_file}.ERROR.json", 'w') as f:
                json.dump(errors, f, indent=4)
        else:
            # rename wsi:
            if config['rename_wsi_files']:
                wsi_name = os.path.basename(wsi_file)
                fyle_type = '.' + wsi_name.split('.')[-1]
                new_wsi_name = '' #f"{config['renaming_pattern'].format(**merged_result)}{fyle_type}"
                for i, key in enumerate(renaming_pattern_list):
                    if key in merged_result:
                        new_wsi_name += ('{' + merged_result[key] + '}')
                    else:
                        new_wsi_name += key
                new_wsi_name += fyle_type

                if os.path.exists(os.path.join(folder_to_watch, new_wsi_name)):
                    # is the file with same name is a different file than the current one?
                    if not filecmp.cmp(wsi_file, os.path.join(folder_to_watch, new_wsi_name)):
                        print(f"WARNING: File '{new_wsi_name}' already exists as different file! "
                              f"Skipping and storing error in '{wsi_file}.ERROR.json'...")
                        with open(f"{wsi_file}.ERROR.json", 'w') as f:
                            errors["renaming-error"] = f"File  already exists!"
                            json.dump(errors, f, indent=4)
                        continue
                else:
                    print(f"Renaming '{wsi_name}' to '{new_wsi_name}'")
                    os.rename(wsi_file, os.path.join(folder_to_watch, new_wsi_name))

            # todo: implement all other metadata export methods (json, csv, etc.)...

    exit()


# test section
if __name__ == "__main__":
    main()