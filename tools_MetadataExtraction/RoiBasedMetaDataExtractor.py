import numpy as np
import cv2
from tools_MetadataExtraction.utils_image import enhance_contrast, sharpen_image
import easyocr
import re
from roifile import roiread
import os, sys
import pytesseract

import argparse
parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--openslide_dll', type=str, required=False, default=None, help='Path to the OpenSlide DLL directory')
parser.add_argument('--dmxt_dll', type=str, required=False, default=None, help='Path to the libdmxt_dll DLL directory')
parser.add_argument('--config', type=str, required=True, help='the .yaml config file (use config/roi_config_example.yaml as template for your configuration!)')
args = parser.parse_args()



if hasattr(os, 'add_dll_directory'):  # Windows
    try:
        OPENSLIDE_PATH = args.openslide_dll
        PYLIBDMTX_PATH = args.dmxt_dll
        with os.add_dll_directory(OPENSLIDE_PATH):
            import openslide
        with os.add_dll_directory(PYLIBDMTX_PATH):
            from pylibdmtx.pylibdmtx import decode
    except:
        print(f"WARNING: Failed to add necessary dll directories for openbslide and pylibmtx!\n"
              f"On windows, you have to pass openslide dll directory as argument --openslide_dll "
              f"(e.g. .../OpenSlide/openslide-bin-4.0.0.2-windows-x64/bin) \n"
              f"and the PYLIBDMTX dll directory as argument --dmxt_dll "
              f"(e.g. .../Lib/site-packages/pylibdmtx/libdmtx_64bit) "
              f"in order to be able to import openbslide and pylibmtx!")
        import openslide
        from pylibdmtx.pylibdmtx import decode
else:
    import openslide
    from pylibdmtx.pylibdmtx import decode

def get_macro_image_from_wsi(wsi_file_path, macro_img_tag):
    """
    Get the macro image from a WSI file.
    :param wsi_file_path: Path to the WSI file.
    :param macro_img_tag: Tag for the macro image.
    :return: Macro image.
    """
    wsi = openslide.OpenSlide(wsi_file_path)
    macro_img = wsi.associated_images[macro_img_tag]
    return macro_img

class RoiBasedMetaDataExtractor():

    def __init__(self, roi_set_path, config=None, wsi_macro_img_tag='macro', wsi_macro_img_rotation=90,
                 ROI_set_ref_res=None):
        '''
        Initialize the ROI-based metadata extractor.
        :param roi_set_path: The path to the ROI set file (.zip, exported roi-set from ImageJ).
        :param config: A dictionary with the ROI names as keys and the configuration as values.
        '''

        # load roi:
        self.rois = roiread(roi_set_path)
        self.wsi_macro_img_tag = wsi_macro_img_tag
        self.wsi_macro_img_rotation = wsi_macro_img_rotation
        self.roiset_ref_res = ROI_set_ref_res
        if self.roiset_ref_res:
            assert len(self.roiset_ref_res) == 2 and type(self.roiset_ref_res[0]) == int, \
                "Parameter 'ROI_set_ref_res' (in config file) must be a tuple with two values (width, height)."

        default_config = {'is_datamatrix': False,
                          'replacement_patterns': [(r'\s+$', '')],
                          'plausibility_regex_check': None,
                          'allowed_chars': None,
                          }
        if not config:
            self.config = {roi.name: default_config for roi in self.rois}
        else:
            for roi in self.rois:
                if roi.name not in config:
                    config[roi.name] = default_config
                else:
                    for key in default_config.keys():
                        if key not in config[roi.name]:
                            config[roi.name][key] = default_config[key]
            self.config = config

            for roi_name in self.config:
                if roi_name not in [roi.name for roi in self.rois]:
                    raise ValueError(f"ROI name '{roi_name}' appears in passed configuration but it appears not in {roi_set_path}!"
                                     f"\nPlease make sure to configure only existing ROIs in the config file."
                                     f"\nAvailable ROI names are {[roi.name for roi in self.rois]}.")

    def extract_metadata_with_roiset(self, wsi_file_path, debug_mode=False,
                                     pxl_offset=0, ocr_engine="easyocr"
                                     ):
        '''
        Extract metadata from the WSI file using the ROI set.
        :param wsi_file_path: The path to the WSI file to process.
        :param debug_mode: If True, debug images are stored in basepath(wsi_file_path)/metadata_debug.
        :param pxl_offset: The offset in pixels to add to the ROI coordinates.
        :param ocr_engine: The OCR engine to use. Supported engines are 'easyocr' and 'pytesseract'.
        :return: A dictionary with the ROI names as keys and the extracted metadata as values.
        '''

        if not ocr_engine.lower() in ["easyocr", "pytesseract"]:
            raise ValueError(f"OCR engine {ocr_engine} not supported! Supported engines are 'easyocr' and 'pytesseract'.")

        # load wsi object with openslide:
        macro_img = get_macro_image_from_wsi(wsi_file_path, self.wsi_macro_img_tag)
        meta_data = {roi.name: None for roi in self.rois}

        # todo: maybe use resize loop if ocr fails?: e.g.:
        # img = cv2.resize(img, (size_factor * img.shape[1], size_factor * img.shape[0]))

        if self.roiset_ref_res:
            macro_img = cv2.resize(np.array(macro_img), (self.roiset_ref_res[0], self.roiset_ref_res[1]))

        if debug_mode:
            debug_folder = os.path.dirname(wsi_file_path) + "/metadata_debug"
            wsi_file_name = os.path.basename(wsi_file_path)
            if not os.path.exists(debug_folder):
                os.makedirs(debug_folder)
            cv2.imwrite(f"{debug_folder}/{wsi_file_name}.png", np.array(macro_img))

        for roi in self.rois:
            left = roi.left + pxl_offset
            right = roi.right + pxl_offset
            top = roi.top + pxl_offset
            bottom = roi.bottom + pxl_offset

            # postprocess, crop according to roi and rotate:
            macro_img_array = np.array(macro_img)
            macro_img_array = enhance_contrast(macro_img_array)
            macro_img_array = sharpen_image(macro_img_array)

            macro_img_array = macro_img_array[top:bottom, :]
            macro_img_array = macro_img_array[:, left:right]

            macro_img_array = cv2.cvtColor(macro_img_array, cv2.COLOR_BGR2GRAY)
            if self.wsi_macro_img_rotation == 0 or not self.wsi_macro_img_rotation:
                pass
            elif self.wsi_macro_img_rotation == 90:
                macro_img_array = cv2.rotate(macro_img_array, cv2.ROTATE_90_CLOCKWISE)
            elif self.wsi_macro_img_rotation in [180, -180]:
                macro_img_array = cv2.rotate(macro_img_array, cv2.ROTATE_180)
            elif self.wsi_macro_img_rotation in [270, -90]:
                macro_img_array = cv2.rotate(macro_img_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                raise ValueError(f"Rotation angle {self.wsi_macro_img_rotation} not supported. Supported angles are 0, 90, 180 and 270.")

            if self.config[roi.name]['is_datamatrix']:
                h, w = macro_img_array.shape[:2]
                decoded_matrix = decode((macro_img_array.tobytes(), w, h))
                if decoded_matrix:
                    meta_data[roi.name] = decoded_matrix[0].data.decode('UTF-8')
            else:
                # apply ocr:
                extracted_text = None
                if ocr_engine.lower() == "easyocr":
                    reader = easyocr.Reader(['en'])
                    data = reader.readtext(macro_img_array, text_threshold=0.7, ycenter_ths=0.5, slope_ths=0.1, paragraph=False)
                    if data:
                        text = ''
                        for i, i_data in enumerate(data):
                            if i_data[2] > 0.25:
                                text += i_data[1] + ' '
                        extracted_text = text
                elif ocr_engine.lower() == "pytesseract":
                    if self.config[roi.name]['allowed_chars'] and type(self.config[roi.name]['allowed_chars']) == str:
                        tesser_config = '-c tessedit_char_whitelist= ' + self.config[roi.name]['allowed_chars']
                    else:
                        tesser_config = ''
                    data = pytesseract.image_to_data(macro_img_array, config=tesser_config, output_type='dict')
                    text = ''
                    for i, i_text in enumerate(data['text']):
                        if (not i_text == '' and not all(char.isspace() for char in i_text)) and data['conf'][i] > -1:
                            text += i_text + ' '
                    extracted_text = text
                else:
                    raise ValueError(f"OCR engine {ocr_engine} not supported! Supported engines are 'easyocr' and 'pytesseract'.")

                if extracted_text:
                    meta_data[roi.name] = extracted_text
                else:
                    meta_data[roi.name] = None

            if self.config[roi.name]['replacement_patterns']:
                meta_data[roi.name] = self._text_post_processing(meta_data[roi.name], self.config[roi.name]['replacement_patterns'])

            if debug_mode:
                cv2.imwrite(f"{debug_folder}/{wsi_file_name}.{roi.name}.{ocr_engine}.{meta_data[roi.name]}.png", macro_img_array)

        return meta_data

    def _text_post_processing(self, text, match_replacement_tuples=[(r'\s+$', '')]):
        if text:
            for pattern, replacement in match_replacement_tuples:
                text = re.sub(pattern, replacement, text)
        return text

def regex_check(text, regex_checks: list):
    '''
    Check if the text matches the regex checks.
    :param text: The text to check.
    :param regex_checks: A list of regex patterns to check against.
    :return: A dictionary with the regex pattern as key and the error message as value. If the text matches the pattern an empty dictionary is returned.
    '''

    error_list = {}

    if not regex_checks:
        return "No regex checks provided."

    for check in regex_checks:
        if not text:
            error_list[check] = "No text found"
            continue
        if not re.search(check, text):
            error_list[check] = f"Does not match '{text}'."

    return error_list

def main():
    from tqdm import tqdm
    import yaml

    #### testing with ROI-config file supported metadata-extraction:

    conf_data_path = args.config
    # load config from yaml file:
    with open(conf_data_path, 'r') as file:
        config = yaml.safe_load(file)

    test_folder = config['folder_to_watch']
    debug_mode = config['debug_mode']

    if config['rename_wsi_files']:
        if not config['renaming_pattern']:
            raise ValueError("If you want to rename the WSI files, you have to provide a renaming pattern.")

    print(f"Using config file: {conf_data_path}, with configuration:")
    for k_config in config:
        print(f"  {k_config}: {config[k_config]}")

    roi_extractor = RoiBasedMetaDataExtractor(config['ROI_set_file'], config=config['extraction_rules'],
                                              wsi_macro_img_tag=config['wsi_macro_img_tag'],
                                              wsi_macro_img_rotation=config['wsi_macro_img_rotation'],
                                              ROI_set_ref_res=config['ROI_set_ref_res'])

    plausibility_regex_checks = {roi_name: config['extraction_rules'][roi_name]['plausibility_regex_check']
                                 for roi_name in config['extraction_rules']}

    wsi_files = []
    for file_type in config['wsi_types']:
        wsi_files += [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.endswith(file_type)]
    for wsi_file in tqdm(wsi_files):

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

        print(f"=== Results for {wsi_file} ===")
        for k in merged_result:
            print(f"{k}: {merged_result[k]}")
            if errors["pytesseract"][k]:
                print(f"  Errors for pytesseract: {errors['pytesseract'][k]}")
            if errors["easyocr"][k]:
                print(f"  Errors for easyocr: {errors['easyocr'][k]}")


    exit()


# test section
if __name__ == "__main__":
    main()
