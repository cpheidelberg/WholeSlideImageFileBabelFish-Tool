
#%% import section
import pytesseract
from tools_MetadataExtraction.TextFindings import left_top_width_height_to_coordinates, calculate_iou, coordinates_to_left_top_width_height
from tools_MetadataExtraction.utils_label import jaccard_similarity, WordBox, distance_between_points
from tools_MetadataExtraction.utils_string import count_characters, count_numbers
import pandas as pd
import cv2
from tools_MetadataExtraction.TextFindings import TextFindings
from icecream import ic
ic.disable()
import numpy as np

#%% define helper functions
def pattern_distance(string):

    d = 1
    if len(string) > 3:
        d += 1

    if count_characters(string) > 1:
        d += 1

    if count_numbers(string) > 2:
        d += 1

    if len(string)>1:
        if string[0].isdigit() and not string[1].isdigit():
            d+=1

    return d

#%% add text found by terasarct
def additional_OCR(label, TextFound):

    print(f"Using additional OCR (tesseract)")

    #%% adapt the label
    resizing_factor = [0.5, 2, 4]
    finding, position, boxes, p = [], [], [], []

    for i_resizing_factor in resizing_factor:
        if not resizing_factor == 1:
            new_width = int(label.shape[1] * i_resizing_factor)
            new_height = int(label.shape[0] * i_resizing_factor)
            # Resize the image
            i_label = cv2.resize(label, (new_width, new_height))
        else:
            i_label = label

        # %% perform OCR
        allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ|#*+[]{}'
        #config = '-c tessedit_char_whitelist=' + allowlist + " --psm 11"
        config = '-c tessedit_char_whitelist=' + allowlist
        data = pytesseract.image_to_data(i_label,
                                         config=config,
                                         output_type='dict')

        # %% add it to the existing data
        for i, i_text in enumerate(data['text']):
            if (not i_text == '' and not all(char.isspace() for char in i_text)) and data['conf'][i] > -1 :
                if i_text == "|":
                    i_text = "I"
                #print(i_text)
                p.append(data['conf'][i]/100) # crude transformation from certaintiy to probability...
                finding.append(i_text)
                position.append(left_top_width_height_to_coordinates(int(data['left'][i]/i_resizing_factor),
                                                                     int(data['top'][i]/i_resizing_factor),
                                                                     int(data['width'][i]/i_resizing_factor),
                                                                     int(data['height'][i]/i_resizing_factor)))
                boxes.append((int(data['left'][i]/i_resizing_factor),
                                int(data['top'][i]/i_resizing_factor),
                                int(data['width'][i]/i_resizing_factor),
                                int(data['height'][i]/i_resizing_factor)))

    df = TextFindings({'finding': finding,
                       'position': position,
                       'box': boxes,
                       'p':p})
    df['method'] = "pytesseract"

    #%% mount it back
    boxes = []
    for i, i_points in enumerate(TextFound.position.tolist()):
        boxes.append(coordinates_to_left_top_width_height(i_points))
    TextFound['box'] = boxes
    TextFound['method'] = "easyocr"

    TextFound = pd.concat([TextFound, df], ignore_index=True)
    TextFound = TextFindings({'finding':TextFound.finding.tolist(),
                              'position': TextFound.position.tolist(),
                              'box': TextFound.box.tolist(),
                              'method': TextFound.method.tolist(),
                              "p": TextFound.p
                              })
    #print(TextFound)

    return TextFound

#%% find caseID and in relation block id
def find_block_id(caseID, TextFound):

    #%% find the caseID in the soup
    finding_similarity = []
    for i in TextFound.finding:
        finding_similarity.append(jaccard_similarity(caseID, i))

    if "method" in TextFound.columns:
        finding_similarity = [a * b for a, b in zip(finding_similarity, TextFound.method == "easyocr")]

    idx_caseID = finding_similarity.index(max(finding_similarity))
    TextFound.attrs['caseID'] = TextFound.finding[idx_caseID]
    TextFound.attrs['caseID_idx'] = idx_caseID
    box = TextFound.box[idx_caseID]
    TextFound.attrs['region_case'] = (box[0], box[1], box[2], 30) # default value

    #%% intersection over union to the case ID
    IoU = []
    box = TextFound.attrs['region_case']
    #box = (0, box[1], 1000, box[3]-5)
    #TextFound.attrs['region_case'] = box # to show later the no go area
    for i in TextFound.box:
        IoU.append(calculate_iou(box, i))
    TextFound['IoU'] = IoU

    # %% intersection over union to the pre-defined region below the case ID
    IoU = []
    box = TextFound.attrs['region_case']
    region = (box[0], box[1] + box[3], box[2], 50) # another default value
    TextFound.attrs['region_block'] = region

    for i in TextFound.box:
        IoU.append(calculate_iou(region, i))

    TextFound['region'] = IoU
    ic(TextFound.finding.tolist())
    ic(IoU)

    #%% calculate distance
    # calculate the distance to the other found words, to identify the fraction description
    # based on its known position
    TextFound['WordBox'] = 'WordBox'
    for i in range(0, len(TextFound)):
        TextFound.loc[i, 'WordBox'] = WordBox(TextFound.position[i],
                                              TextFound.finding[i])

    # distance in real space on the label
    TextFound['distance'] = 1e6
    for i in range(0, len(TextFound)):
        TextFound.loc[i, 'distance'] = distance_between_points(TextFound.WordBox[idx_caseID].pointE,
                                                               TextFound.WordBox[i].pointF)

    #%% distance in so-called pattern space
    TextFound['distance_pattern'] = 1e6
    for i in range(0, len(TextFound)):
        TextFound.loc[i, 'distance_pattern'] = pattern_distance(TextFound.loc[i, 'finding'])

    #%% decision if above or below the case number
    TextFound['distance_box'] = 1e6
    for i in range(0, len(TextFound)):
        TextFound.loc[i, 'distance_box'] = TextFound.box[idx_caseID][1] - TextFound.box[i][1]

    return TextFound

#%% decide
def find_match(TextFound):

    # %% answer the single questions
    idx_distance = TextFound.distance < 90 # distance of the potential block ID from the case ID
    idx_pattern = TextFound.distance_pattern < 2 # check, if the pattern of the potential block ID is correct
    idx_location = TextFound.distance_box < 0 # check if the potential block ID is in the area below the case ID
    idx_IoU = TextFound.IoU <= 0.45 # check that the potential block ID does not overlap with the case ID or is a part of it
    idx_region = TextFound.region > 0.1 # check if the potential block ID is in the expected area / region

    # %% formulate the decision
    idx = idx_pattern * idx_location * idx_distance * idx_IoU * idx_region

    # %% react to more than one finding / find the one
    method = ["decision"]

    # just prefer the method
    if ("heuristic" in method or "easyocr" in method or "pytesseract" in method) \
                and sum(idx) > 1:
        idx_method = TextFound.method == method[0]
        idx = idx_pattern * idx_location * idx_distance * idx_IoU * idx_method * idx_region

    # just prefer the highest probability
    if "prob" in method and sum(idx) > 1:
        idx_max = TextFound.p.tolist().index(max(TextFound.p * idx))
        idx = [idx[i] if i == idx_max else False for i in range(len(idx))]

    # do a complex trade-off (does also include heuristics)
    if "decision" in method and sum(idx) > 1:
        idx_method = np.float64(TextFound.method == "easyocr")
        max_easyocr = TextFound.p.tolist().index(max(TextFound.p * idx * idx_method))
        idx_max_easyocr = [idx[i] if i == max_easyocr else False for i in range(len(idx))]

        idx_method = np.float64(TextFound.method == "pytesseract")
        max_pytesseract = TextFound.p.tolist().index(max(TextFound.p * idx * idx_method))
        idx_max_pytesseract = [idx[i] if i == max_pytesseract else False for i in range(len(idx))]

        if sum(idx_max_pytesseract) > 0 and sum(idx_max_easyocr) > 0:
            if TextFound.loc[idx_max_easyocr, 'p'].iloc[0] > \
                    TextFound.loc[idx_max_pytesseract, 'p'].iloc[0]:# - 0.25:
                idx = idx_max_easyocr
            else:
                idx = idx_max_pytesseract
        else:
            if sum(idx_max_pytesseract) == 0 and sum(idx_max_easyocr) > 0:
                idx = idx_max_easyocr
            elif sum(idx_max_pytesseract) > 0 and sum(idx_max_easyocr) == 0:
                idx = idx_max_pytesseract

    if sum(idx) > 1:
        idx = [(idx == 1)[0]]

    return idx
