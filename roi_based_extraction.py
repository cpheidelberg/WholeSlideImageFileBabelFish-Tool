import os, sys
from tqdm import tqdm
import yaml
import json
from tools_MetadataExtraction.RoiBasedMetaDataExtractor import RoiBasedMetaDataExtractor, regex_check, get_macro_image_from_wsi
import filecmp
import datetime, time
from tools_ROIconfig.train_resnet_label_classifier import SlideLabelResnetClassifier

# script params:
save_print_to_log_file = False

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

    if not config['rename_wsi_files']:
        raise NotImplementedError("Currently, only renaming of WSI files is supported. Please set 'rename_wsi_files' to True.")

    print(f"Using config file: {conf_data_path}, with configuration:")
    for k_config in config:
        print(f"  {k_config}: {config[k_config]}")
    print()

    # load slide label classifier model:
    slide_label_classifier = None
    if 'model_path' in config['slide_label_classifier']:
        if config['slide_label_classifier']['model_path']:
            assert os.path.exists(config['slide_label_classifier']['model_path']), f"Model path '{config['slide_label_classifier']['model_path']}' does not exist!"
            assert 'model_type' in config['slide_label_classifier'], f"Model type must be specified in the config file!"

            try:
                slide_label_classifier = SlideLabelResnetClassifier(config['slide_label_classifier']['model_path'], config['slide_label_classifier']['model_type'])
            except Exception as e:
                raise Exception(f"Error loading slide label classifier model {config['slide_label_classifier']['model_path']}: {e}"
                      f"\nPlease check the model path and type in the config file or use null "
                      f"to replace slide-label-classification with brute-force search.")


    # for each label configuration, create a renaming pattern list and roi_extractor:
    roi_extractors = {}
    for label_config_name in config['label_configs'].keys():
        label_config = config['label_configs'][label_config_name]
        renaming_pattern_list = []  # should be a list of shape [meta_key, seperation_symbol, meta_key, seperation_symbol, ..., meta_key]
        if not label_config['renaming_pattern']:
            raise ValueError("If you want to rename the WSI files, you have to provide a renaming pattern.")
        assert '{' in label_config['renaming_pattern'] and '}' in label_config['renaming_pattern'], 'renaming_pattern must contain at least one pair of curly brackets!'
        for start_word in label_config['renaming_pattern'].split('{'):
            if start_word == '':
                continue
            if len(start_word.split('}')) == 1:
                meta_key = start_word.split('}')[0]
                assert meta_key in [roi_name for roi_name in label_config['extraction_rules']], \
                    f"Meta key '{meta_key}', which appears in the renaming_pattern, not found in extraction rules!"
                if meta_key != '':
                    renaming_pattern_list += [meta_key]
            elif len(start_word.split('}')) == 2:
                meta_key = start_word.split('}')[0]
                assert meta_key in [roi_name for roi_name in label_config['extraction_rules']], \
                    f"Meta key '{meta_key}', which appears in the renaming_pattern, not found in extraction rules!"
                seperation_symbol = start_word.split('}')[1]
                renaming_pattern_list += [meta_key, seperation_symbol] if seperation_symbol else [meta_key]
            else:
                raise ValueError(f"Renaming pattern '{config['renaming_pattern']}' is not supported!")
        config['label_configs'][label_config_name]['renaming_pattern_list'] = renaming_pattern_list
        #print(f"Renaming pattern list: {renaming_pattern_list}")



        roi_extractors[label_config_name] = RoiBasedMetaDataExtractor(label_config['ROI_set_file'], config=label_config['extraction_rules'],
                                                  wsi_macro_img_tag=config['wsi_macro_img_tag'],
                                                  wsi_macro_img_rotation=label_config['wsi_macro_img_rotation'],
                                                  ROI_set_ref_res=label_config['ROI_set_ref_res'] if 'ROI_set_ref_res' in label_config else None,)

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
                skip = False
                for renaming_pattern in [config['label_configs'][config_name]['renaming_pattern'] for config_name in config['label_configs'].keys()]:
                    if renaming_pattern:
                        if (len(wsi_name.split('{')) == len(wsi_name.split('}')) and
                                len(wsi_name.split('{')) == len(renaming_pattern.split('{'))):
                            skip = True
                            break
                if skip:
                    print(
                        f"Skipping '{wsi_name}' as it is already processed "
                        f"(renaming pattern '{renaming_pattern}' matches with '{wsi_name}').")
                    continue

        # apply pretrained slide label classifier to get the most probable label configuration:
        try:
            # todo: low-priority: potential performance improvement: Avoid loading the macro-image two times
            #  (currently we load it once for slide-label-classification and once for the roi extractor)
            if config['debug_mode']:
                start_time = time.time()
                sorted_label_type_names, sorted_probabilities = slide_label_classifier.predict(
                    get_macro_image_from_wsi(wsi_file, config['wsi_macro_img_tag']))
                end_time = time.time()
                print(f"Predicted label type names: {sorted_label_type_names}")
                print(f"Predicted probabilities: {[round(p,4) for p in sorted_probabilities]}")
                print(
                    f"Slide label classifier took {(end_time - start_time) * 1000:.2f} ms to classify the slide '{wsi_file}'")
            else:
                sorted_label_type_names, sorted_probabilities = slide_label_classifier.predict(
                    get_macro_image_from_wsi(wsi_file, config['wsi_macro_img_tag']))
        except Exception as e:
            sorted_label_type_names = list(config['label_configs'].keys())
            if len(sorted_label_type_names) > 1:
                print(f"Warning: Slide label classifier failed to classify the slide '{wsi_file}' ({e}). "
                      f"Will randomly try all label configurations in the following order: {sorted_label_type_names}")

        # for each label_type configuration, try to extract metadata (break if plausible result is found)
        for label_config_name in sorted_label_type_names:

            label_config = config['label_configs'][label_config_name]
            roi_extractor = roi_extractors[label_config_name]

            plausibility_regex_checks = {
                roi_name: label_config['extraction_rules'][roi_name]['plausibility_regex_check']
                for roi_name in label_config['extraction_rules']}

            # extract metadata with pytesseract and easyocr and store in dict both_results:
            both_results = {}
            for ocr_engine in ["pytesseract", "easyocr"]:
                both_results[ocr_engine] = roi_extractor.extract_metadata_with_roiset(wsi_file, debug_mode=debug_mode,
                                                                                      ocr_engine=ocr_engine)

            # merge results of both ocr engines in a smart way, so that only plausible results are stored in merged_result:
            merged_result = {}
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

            # store error if any errors, otherwise export the extracted metadata somehow:
            # (currently only renaming of WSI files is supported)
            if any(merged_result[k] is None for k in merged_result):
                print(f"Plausibility check failed for {wsi_file}, when using label_config {label_config_name}:")
                print(errors)
                # store the error in a json file:
                with open(f"{wsi_file}.ERROR.json", 'w') as f:
                    json.dump(errors, f, indent=4)
            else:
                print(f"Plausibility check passed for {wsi_file}, when using label_config {label_config_name}:")
                # rename wsi:
                if config['rename_wsi_files']:
                    wsi_name = os.path.basename(wsi_file)
                    fyle_type = '.' + wsi_name.split('.')[-1]
                    new_wsi_name = '' #f"{label_config['renaming_pattern'].format(**merged_result)}{fyle_type}"
                    for i, key in enumerate(config['label_configs'][label_config_name]['renaming_pattern_list']):
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
                    else:
                        print(f"Renaming '{wsi_name}' to '{new_wsi_name}'")
                        os.rename(wsi_file, os.path.join(folder_to_watch, new_wsi_name))

                # todo: implement all other metadata export methods (json, csv, etc.)...

                break # as soon as we have a plausible result, we dont need to try other label_configs anymore

    exit()


# test section
if __name__ == "__main__":
    main()