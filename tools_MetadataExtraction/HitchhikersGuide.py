# import
import sys
from thefuzz import fuzz
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tools_MetadataExtraction.utils_image import enhance_contrast, sharpen_image
import easyocr
import pandas as pd
import math
import re
from tools_MetadataExtraction.utils_label import replace_numbers_with_words, clean_findings
from tools_MetadataExtraction.TextFindings import TextFindings
from tools_MetadataExtraction.label_repair.HealingPortion import SmallHealingPortion, LargeHealingPortion

SmallPortion = SmallHealingPortion(method="ball")
LargePortion = LargeHealingPortion()

from tools_MetadataExtraction.utils_block import additional_OCR, find_block_id, find_match
from tools_MetadataExtraction.utils_string import ends_with_any, find_string_with_pattern, find_first_number, \
    count_numbers
from tools_MetadataExtraction.utils_string import count_characters, check_pattern, special_characters
from tools_MetadataExtraction.TextFindings import coordinates_to_left_top_width_height
from tools_MetadataExtraction.utils_ocr import adapt_position

from roifile import roiread

import os

WSI_MACRO_IMG_KEY = 'macro'

if hasattr(os, 'add_dll_directory'):  # Windows
    try:
        OPENSLIDE_PATH = sys.argv[1]
        PYLIBDMTX_PATH = sys.argv[2]
        with os.add_dll_directory(OPENSLIDE_PATH):
            import openslide
        with os.add_dll_directory(PYLIBDMTX_PATH):
            from pylibdmtx.pylibdmtx import decode
    except:
        print(f"WARNING: Failed to add necessary dll directories for openbslide and pylibmtx!\n"
              f"On windows, you have to pass openslide dll directory as first argv "
              f"(e.g. .../OpenSlide/openslide-bin-4.0.0.2-windows-x64/bin) \n"
              f"and the PYLIBDMTX dll directory as second argv "
              f"(e.g. .../Lib/site-packages/pylibdmtx/libdmtx_64bit) "
              f"in order to be able to import openbslide and pylibmtx!")
        import openslide
        from pylibdmtx.pylibdmtx import decode
else:
    import openslide
    from pylibdmtx.pylibdmtx import decode


# define class to read the label
class LabelReader():
    def __init__(self, file, methods=None):

        if methods is None:
            methods = ["SmallPortion", 'conventional']
        self.methods = methods

        if isinstance(file, np.ndarray):
            self.wsi = file
        else:
            if ends_with_any(file):
                self.wsi = file
            else:
                self.wsi = openslide.OpenSlide(file)

        self.slide_label = self.__get_label()
        self.slide_datamatrix = self.__get_data_matrix()

        if isinstance(file, np.ndarray):
            self.macro = file

        else:

            if ends_with_any(file):
                self.macro = cv2.imread(file)
            else:
                self.macro = np.array(self.wsi.associated_images[WSI_MACRO_IMG_KEY])

    def __get_label(self):

        if isinstance(self.wsi, np.ndarray):
            img = self.wsi

        else:
            if isinstance(self.wsi, str):
                img = cv2.imread(self.wsi)
            else:
                img = np.array(self.wsi.associated_images[WSI_MACRO_IMG_KEY])

        size_factor = 1
        img = cv2.resize(img, (size_factor * img.shape[1], size_factor * img.shape[0]))
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = img[0:int(size_factor * 300), 0:-1]

        methods = self.methods
        # methods = ["LargePortion",'conventional']

        if "SmallPortion" in methods:
            img = SmallPortion.heal(img)

        if "LargePortion" in methods:
            img = LargePortion.heal(img)

        if "conventional" in methods:
            img = enhance_contrast(img)
            img = sharpen_image(img)

        return img

    def __get_data_matrix(self):

        if isinstance(self.wsi, np.ndarray):
            label = self.wsi
            return label

        else:

            if isinstance(self.wsi, str):
                label = cv2.imread(self.wsi)
            else:
                label = np.array(self.wsi.associated_images[WSI_MACRO_IMG_KEY])

            label = enhance_contrast(label)
            label = sharpen_image(label)
            label = cv2.rotate(label, cv2.ROTATE_90_CLOCKWISE)
            label = label[0: 400, :]
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
            h, w = label.shape[:2]
            return label


#
# define the class for getting information from the found word soup
class BableFish():
    def __init__(self, bable_list, label_image, TextFound=None, staining_list="StoneOfRosette_Stainings.xlsx"):

        self.soup = self.__adapt_list(bable_list)
        self.bable = self.__get_bable(self.soup)
        self._bable_words = self.__get_bable_words(bable_list)
        self.staining_list = self.__get_staining_list(staining_list)
        self.staining_names = self.__get_staining_names(staining_list)
        self._staining_list_words = self.__get_staining_list_words(self.staining_list)
        self.TextFound = TextFound
        self.label_image = label_image

        # set the vaules to default in case of error
        self.number_circle = "None"
        self.year = "None"
        self.case_id = "None"
        self.stain = "None"
        self.fraction = "None"

    def __adapt_list(self, word_list):

        words_adapted = []
        for i_word in word_list:
            if re.search('0', i_word) is not None:
                words_adapted.append(i_word.replace('O', '0'))
            if re.search('1', i_word) is not None:
                # pattern = r'([a-zA-Z])1'
                # words_adapted.append(re.sub(pattern, r'\1/', i_word))
                words_adapted.append(i_word.replace('1', '/'))

        remove_list = []
        for i, i_word in enumerate(words_adapted):
            if i_word == "/":
                remove_list.append(i)
            if i_word.find(" /") > -1:
                remove_list.append(i)

        for i, i_word in enumerate(words_adapted):
            if not i in remove_list:
                word_list.append(i_word)

        for i, i_string in enumerate(word_list):
            word_list[i] = i_string.replace(" /", "/")

        return word_list

    def __get_bable(self, word_list):
        bable = ' # '.join(word for word in word_list)
        return bable

    def __get_bable_words(self, words_found):
        bable = ' # '.join(word for word in words_found)
        bable = bable.replace('O', '0')
        bable = replace_numbers_with_words(bable)
        return bable

    def __get_staining_list(self, staining_list):
        df = pd.read_excel(staining_list)
        df = df.sort_values(["ID"], ignore_index=True)
        staining_list = df.Stain.tolist()
        staining_list = [x for x in staining_list if not isinstance(x, float) or not math.isnan(x)]
        return staining_list

    def __get_staining_names(self, staining_list):
        df = pd.read_excel(staining_list)
        df = df.sort_values(["ID"], ignore_index=True)
        staining_names = df.Name.tolist()
        staining_names = [x for x in staining_names if not isinstance(x, float) or not math.isnan(x)]
        return staining_names

    def __get_staining_list_words(self, staining_list):
        staining_list_words = [replace_numbers_with_words(i_stain) for i_stain in staining_list]
        return staining_list_words

    def __get_staining(self):

        # get the staining based on the soup
        stain = []
        for i, i_stain in enumerate(self._staining_list_words):
            if not re.search(i_stain + " ", self._bable_words) is None:
                stain.append(self.staining_list[i])

        # do fuzzy matching
        if not len(stain) == 1:  # or "CD" in stain: # still no match found

            match_list = np.zeros((len(self.staining_list), len(self.soup)))
            for x, i_word in enumerate(self.soup):
                for y, i_stain in enumerate(self.staining_list):
                    if check_pattern(i_word) or len(i_word) < 3 or len(i_word) < len(i_stain):
                        match_list[y, x] = 0
                    else:
                        match_list[y, x] = fuzz.partial_ratio(i_stain, i_word)
            # print(match_list)
            i_stain, i_word = np.unravel_index(np.argmax(match_list), match_list.shape)
            if np.max(match_list) > 80:
                stain = self.staining_list[i_stain]  # + "The Fuzz"
                if np.max(match_list) < 100:
                    print(f"fuzzy logic used to find stain {stain} with machting {np.max(match_list)}")

        # react to no finding at all
        if len(stain) == 0:
            stain = "HE"
        else:
            # last step to map alternative staining names to their official counter part
            if isinstance(stain, list):
                stain_found = stain[0]
            elif isinstance(stain, str):
                stain_found = stain
            stain = self.staining_names[self.staining_list.index(stain_found)]

        return stain

    def _get_label_parts(self):

        stain = self.__get_staining()
        self.stain = stain

        # % get the case ID (year, caseID per year, number system) based on the soup
        caseID = 'No ID'
        # print(self.soup)
        caseID_found = find_string_with_pattern(self.soup)
        # print(caseID_found)

        if caseID_found is not None:
            caseID = caseID_found.replace(" ", "")

            pattern = re.compile(r'[a-zA-Z]')
            if re.search(pattern, caseID) is not None:  # case IDs with a letter
                year = caseID[caseID.find('/') + 1:caseID.find('/') + 3]
                number_circle = caseID[0]
                case_id = caseID[1:caseID.find('/')]
            else:  # case ID without leading letters, for example IHC
                year = caseID[caseID.find('/') + 1:caseID.find('/') + 3]
                number_circle = "E"
                case_id = caseID[0:caseID.find('/')]

            if not number_circle in ["E", "R", "T"]:
                number_circle = "E"  # is the default

            case_id = find_first_number(case_id)

        else:
            case_id = "None"
            number_circle = "None"
            year = "None"

        if number_circle is not None:
            self.number_circle = number_circle
        if case_id is not None:
            self.case_id = case_id
        if year is not None:
            self.year = year

    def _get_fraction(self, caseID, method="by position"):

        # find the word found containing the case ID
        finding_similarity = []
        TextFound = self.TextFound

        boxes = []
        for i, i_points in enumerate(TextFound.position.tolist()):
            boxes.append(coordinates_to_left_top_width_height(i_points))
        TextFound['box'] = boxes

        finding = []

        if len(TextFound.finding) < 300:
            TextFound = additional_OCR(self.label_image, TextFound)

        if method == "by pattern":

            finding = [string for string in TextFound.finding if len(string) < 5 and count_characters(string) < 2]
            if len(finding) == 0:
                fraction = "A 1"
                return fraction

            if len(finding) == 1:
                finding = finding[0]

        if method == "by position" or len(finding) > 1:

            TextFound = find_block_id(caseID, TextFound)

            if len(TextFound.finding) < 5:
                n = len(TextFound.finding)
                # print(f"additional OCR done sine only {len(TextFound.finding)} text elements found")
                TextFound = additional_OCR(self.label_image, TextFound)
                # print(f"{len(TextFound.finding) - n} elements are added")

            TextFound = find_block_id(caseID, TextFound)
            self.TextFound = TextFound

            idx_fraction = find_match(TextFound)

            # print(f" fractions idx {idx_fraction}")
            # print(TextFound.finding)
            if all(element == 0 for element in idx_fraction):
                fraction = "A 1"
                return fraction

            finding = TextFound.loc[idx_fraction, 'finding'].tolist()[0]

            if special_characters(finding):
                finding = finding[0]

            if len(finding) == 0:
                fraction = "A 1"
                return fraction

        # there is no fraction at the expected position
        if fuzz.ratio(caseID, finding) > 80:
            fraction = "A 1"  # the default
            return fraction

        # crap found
        if special_characters(finding):
            fraction = "A 1"
            return fraction

        # check for missing empty spaces
        if len(finding) > 1 and finding.count(' ') == 0 and finding[0].isalpha():
            finding = finding[0] + " " + finding[1:-1] + finding[-1]

        # perfect fit
        if len(finding) <= 4 and count_characters(finding) == 1 \
                and count_numbers(finding) <= 2 and count_numbers(finding) > 0:
            fraction = finding
            return fraction

        # case with single character
        if count_characters(finding) == 1 and count_numbers(finding) == 0 \
                and len(finding) == 1:
            fraction = finding + " 1"
            return fraction

        # case with single number
        if count_characters(finding) == 0 and count_numbers(finding) <= 2 \
                and len(finding) <= 2:
            fraction = "A " + finding
            return fraction

        # case with number and character without space
        if len(finding) <= 3 and count_characters(finding) == 1 \
                and count_numbers(finding) <= 2:
            fraction = finding[0] + " " + finding[1:len(finding)]
            return fraction

        return "A 1"  # complete default value

    def get_label(self):
        self.case_id = "None"
        self.year = "None"
        self.stain = "None"
        self.number_circle = "None"

        self._get_label_parts()
        fileID = self.number_circle + "/" + self.year + "/" + self.case_id
        self.fraction = self._get_fraction(self.number_circle + " " + self.case_id)
        self.fileID = fileID + "/" + self.fraction + "/" + self.stain

        fileID_SectraStyle = self.number_circle + "_" + self.year + '_' + self.case_id + "#"  # + str(fraction) + "+" + stain
        self.fileID_SectraStyle = fileID_SectraStyle


#
# define the combined class
class HitchhickerGuide():
    def __init__(self, file_path, methods=["SmallPortion", 'conventional']):

        self.file_path = file_path
        self.SlideLabel = LabelReader(file_path, methods=methods)
        self.slideLabel = self.SlideLabel.slide_label
        self.dataMatrix = self.SlideLabel.slide_datamatrix

        # init class:
        self.get_slideID()
        self.read_datamatrix()

    def read_label(self):
        reader = easyocr.Reader(['en'])

        try:
            nonsense_list = pd.read_excel("IgnoreList.xlsx").Term
            nonsense_list = [str(i) for i in nonsense_list]
        except:
            nonsense_list = []
            print("WARNING: No IgnoreList.xlsx found")

        resizing_factor = [1, 2, 3]  # vorher 3
        words_found, word_position, p = [], [], []
        for resize in resizing_factor:
            label = self.slideLabel
            if not resize == 1:
                new_width = int(label.shape[1] * resize)
                new_height = int(label.shape[0] * resize)

                # Resize the image
                label = cv2.resize(label, (new_width, new_height))

            extract_info = reader.readtext(label,
                                           text_threshold=0.7,
                                           ycenter_ths=0.5,
                                           slope_ths=0.1,
                                           paragraph=False)

            for el in extract_info:
                if not el[1] in nonsense_list and el[2] > 0.25:
                    words_found.append(el[1])
                    word_position.append(adapt_position(el[0], resize))
                    p.append(el[2])

        self.TextFound = TextFindings({"position": word_position, "finding": words_found,
                                       "p": p})
        # words_found = clean_findings(words_found, word_position)
        self.LabelText = BableFish(words_found, self.slideLabel, self.TextFound)

    def read_datamatrix(self):
        label = self.dataMatrix
        h, w = label.shape[:2]

        data_matrix = decode((label.tobytes(), w, h))
        if not data_matrix:
            self.text_datamatrix = None
        else:
            self.text_datamatrix = data_matrix

    def get_slideID(self):
        self.read_label()
        self.LabelText.get_label()
        self.fileID = self.LabelText.fileID
        if self.LabelText.number_circle is None or self.LabelText.year is None or self.LabelText.case_id is None:
            self.caseID = "None"
        else:
            self.caseID = self.LabelText.number_circle + "_" + self.LabelText.year + '_' + self.LabelText.case_id
        self.prefix = self.LabelText.number_circle
        self.case = self.LabelText.case_id
        self.suffix = self.LabelText.year
        self.study = self.caseID  # histology number = caseID in DICOM; no patient ID here
        self.block = self.LabelText.fraction
        self.stain = self.LabelText.stain
        self.series = self.block + "_" + self.stain  # series is in DICOM the combination of block and staining
        self.TextFound = self.LabelText.TextFound
        self.TextFound.attrs['label'] = self.SlideLabel.slide_label
        self.words_found = self.TextFound.finding.tolist()


class RoiBasedMetaDataExtractor():

    def __init__(self, roi_set_path, methods=None):

        if methods is None:
            methods = ["SmallPortion", 'conventional']

        # load roi:
        self.rois = self.load_roi_set(roi_set_path)

    def load_roi_set(self, roi_path):
        rois = roiread(roi_path)
        return rois

    def extract_metadata_with_roiset(self, wsi_file_path, these_rois_are_datamatrices=[], debug_mode=False, pxl_offset = 0):

        # load wsi object with openslide:
        wsi = openslide.OpenSlide(wsi_file_path)

        macro_img = wsi.associated_images[WSI_MACRO_IMG_KEY]

        meta_data = {roi.name: None for roi in self.rois}

        for roi in self.rois:
            debug_file_name = f"{wsi_file_path}.ROIresult.{roi.name}.png"
            left = roi.left + pxl_offset
            right = roi.right + pxl_offset
            top = roi.top + pxl_offset
            bottom = roi.bottom + pxl_offset
            if roi.name in these_rois_are_datamatrices:
                macro_img_array = np.array(macro_img)
                macro_img_array = enhance_contrast(macro_img_array)
                macro_img_array = sharpen_image(macro_img_array)

                if debug_mode:
                    cv2.imwrite(debug_file_name.replace('.png', '.in.png'), macro_img_array)

                macro_img_array = macro_img_array[top:bottom, :]
                macro_img_array = macro_img_array[:, left:right]
                macro_img_array = cv2.cvtColor(macro_img_array, cv2.COLOR_BGR2GRAY)

                # slide label is usually left, so lets turn it to the right to have the label at top:
                macro_img_array = cv2.rotate(macro_img_array, cv2.ROTATE_90_CLOCKWISE)

                if debug_mode:
                    cv2.imwrite(debug_file_name.replace('.png', '.out.png'), macro_img_array)

                h, w = macro_img_array.shape[:2]
                meta_data[roi.name] = decode((macro_img_array.tobytes(), w, h))
            else:
                pass # todo: implement metadata extraction from ROIs



# test section
if __name__ == "__main__":
    #### new tests with ROI-config file supported metadata-extraction:
    roi_extractor = RoiBasedMetaDataExtractor("./tools_ROIconfig/RoiSetE.zip")
    roi_extractor.extract_metadata_with_roiset("D:\\Research\\Slides\\lufi-test\\to_whatch\\LuFi001_I_HE_PAS.ndpi",
                                               these_rois_are_datamatrices=["slide-id"], debug_mode=True)

    exit()

    #### old tests with hard-coded meta-extraction:
    # test label readinng class
    Test = LabelReader("./test_data/test#3.ndpi")
    print(f"slide label read has size {Test.slide_label.shape}")
    print(f"slide data matrx read has size {Test.slide_datamatrix.shape}")

    #
    # test the Letter Soup Class
    test_soup = ['22 CD2O', 'Kt', 'Uni HD', '08,02.2024', '6094124', '40499']
    Test = BableFish(test_soup)
    Test.get_label()
    print(f"label derived is {Test.fileID}")

    #
    # tes the Hitchhikers Guide class
    Test = HitchhickerGuide("./test_data/test#2.ndpi")  # known HE
    Test.get_slideID()
    plt.imshow(Test.SlideLabel.macro)
    plt.title("Macro image")
    plt.savefig("test_data/test#2_macro.png")
    print(f"words found are {Test.LabelText.soup}")
    print(f"label read as {Test.fileID} for known HE")

    Test = HitchhickerGuide("./test_data/test#3.ndpi")  # known CD20
    Test.get_slideID()
    plt.imshow(Test.SlideLabel.macro)
    plt.savefig("test_data/test#3_macro.png")
    print(f"words found are {Test.LabelText.soup}")
    print(f"label read as {Test.fileID} for known CD20")

    Test = HitchhickerGuide("./test_data/test#4.ndpi")  # known CD20
    Test.get_slideID()

    #
    plt.imshow(Test.SlideLabel.macro)
    plt.savefig("test_data/test#5_macro.png")
    print(f"words found are {Test.LabelText.soup}")
    print(f"label read as {Test.fileID} for known HE with different fraction")
