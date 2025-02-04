import re
import pandas as pd
import math
def replace_numbers_with_words(text):
    # Define a dictionary to map numbers to their word equivalents
    number_words = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }

    # Define a regular expression pattern to find numbers in the text
    pattern = re.compile(r'\d')

    # Use a lambda function as the replacement function to convert the matched number to its word equivalent
    replaced_text = pattern.sub(lambda match: number_words.get(match.group(0), match.group(0)), text)

    return replaced_text

def is_word_in_string(word_list, input_string):
    words_in_string = input_string.split()
    for word in word_list:
        if word in words_in_string:
            return True
    return False


def clean_findings(list2clean, nonsense_list="./IgnoreList.xlsx"):

    if isinstance(nonsense_list, str):
        nonsense_list = pd.read_excel(nonsense_list).Term

    #nonsense_list = set(nonsense_list)
    filtered_list = [elem for elem in list2clean if elem not in nonsense_list]

    return filtered_list

def jaccard_similarity(str1, str2):
    """
    Compute the Jaccard similarity between two strings.
    """
    if str1 is None:
        str1 = "None"
    set1 = set(str1)
    if str2 is None:
        str2 = "None"
    set2 = set(str2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

class WordBox():
    def __init__(self, bbox, word):
        self.pointA = bbox[1]
        self.pointB = bbox[2]
        self.pointC = bbox[3]
        self.pointD = bbox[0]
        self.word = word

        self.pointE = (
            (self.pointC[0] + 0.3* abs(self.pointB[1] - self.pointC[1])),
            (self.pointC[1] + self.pointB[1])/2
                        )

        self.pointF = (
            (self.pointD[0] + 0.5 * abs(self.pointA[0] - self.pointD[0])),
            (self.pointA[1] + self.pointD[1]) / 2,
                       )

def distance_between_points(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance
