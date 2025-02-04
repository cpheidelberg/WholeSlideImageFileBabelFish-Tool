
import re
import difflib
ENDINGS = ['.png', 'tif', 'jpeg']

#%% define all the helper functions ... quite many
def ends_with_any(file_name):
    for ending in ENDINGS:
        if file_name.endswith(ending):
            return True
    return False

def change_char_in_string(s, index, new_char):
    if index < 0 or index >= len(s):
        raise IndexError("Index out of range")
    # Convert the string to a list of characters
    s_list = list(s)
    # Change the character at the specified index
    s_list[index] = new_char
    # Join the list back into a string
    return ''.join(s_list)

def remove_nearly_similar_elements(input_list, similarity_threshold=0.8):
    # Function to determine if two strings are similar
    def is_similar(str1, str2):
        return difflib.SequenceMatcher(None, str1, str2).ratio() > similarity_threshold

    result_list = []
    for i, item in enumerate(input_list):
        idx = [is_similar(item, i) for i in input_list]
        if sum(idx) < 2:
            result_list.append(item)

    return result_list


def find_first_number(string):
    match = re.search(r'\d+', string)
    if match:
        return match.group()
    else:
        return None

def find_number_series(input_string):
    # Define a regex pattern to match a series of numbers separated by a '/'
    pattern = r'\b(\d+/\d+(?:/\d+)*)\b'

    # Use re.findall to find all occurrences of the pattern in the input string
    matches = re.findall(pattern, input_string)

    return matches

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

def check_pattern(input_string):
    pattern = r'^\s*[a-zA-Z0-9]{1}\s*[0-9]{1}\s*$'
    return bool(re.match(pattern, input_string))

def special_characters(input_string):
    # Define a set of characters you consider as special
    special_characters = set("!@#$%^&*()_+{}[]:\";'<>?,./\\|")

    # Check if any character in the input string is in the set of special characters
    for char in input_string:
        if char in special_characters:
            return True
    return False

def count_special_characters(input_string):
    # Define a set of characters you consider as special
    special_characters = set("!@#$%^&*()_+{}[]:\";'<>?,./\\|")

    # Check if any character in the input string is in the set of special characters
    n = 0
    for char in input_string:
        if char in special_characters:
            n+=1

    return n

def contains_only_alphanumeric(input_string):
    # Define a regular expression pattern to match alphanumeric characters, blank spaces, and special signs
    pattern = r'^[a-zA-Z0-9\s!@#$%^&*()-_=+{}\[\]:;"\'|\\,<.>/?`~]*$'

    # Use the re.match() function to check if the input string matches the pattern
    if re.match(pattern, input_string):
        return True
    else:
        return False

def remove_special(text):
    signs = ["'", "|", "]", "[", "{", "}", ":", ".", '"']
    for sign in signs:
        text = text.replace(sign, "")
    return text

def remove_duplicates(lst):
    lst = flatten_list(lst)
    return list(set(lst))

def flatten_list(nested_list):
    flat_list = []

    def flatten(sublist):
        for item in sublist:
            if isinstance(item, list):
                flatten(item)
            else:
                flat_list.append(item)

    flatten(nested_list)
    return flat_list


def measure_pattern(string):

    p = 0

    if len(string) < 4:
        p = 0
        return p

    if string[0].isalpha():
        p+=1

    if string[1] == " ":
        p+=1

    if string[-3] == "/":
        p+=1

    if string[0] == "/":
        p-=1

    return p

#%% finally the function
def find_string_with_pattern(strings):

    if not isinstance(strings, list):
        strings = [strings]
    #print(strings)
    strings = remove_duplicates(strings)
    string_found = []

    #%%
    n = 0
    while len(string_found)==0  and n < 1:

        for string in strings:
            if string.count('/') == 1:  # Check if there's exactly one '/'
                parts = string.split('/')
                if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 2:
                    if string[-3] == "/":
                        if string[0].isalpha() and count_special_characters(string[1]) > 0:
                            string = string[0] + string[2:-1] + string[-1]
                        string_found.append(string)

        if len(string_found) == 1:
            string_found = remove_special(string_found[0])

        elif isinstance(string_found, list) and len(string_found) > 1:

            strings = string_found
            string_found = []
            for string in strings:
                if count_characters(string) <= 1:
                    string_found.append(string)

            if len(string_found) == 0:
                string_found = []
                for string in strings:
                    string_found.append(find_number_series(string)[0])

            if len(string_found) == 1:
                string_found = string_found[0]
            else:
                strings = string_found
                string_found = []
                for string in strings:
                    if string.count('/') == 1 and count_characters(string)<=1 and \
                            count_special_characters(string) == 1:
                        string_found.append(string)
                    elif string.count('/') == 1 and count_characters(string)<=1 and \
                            count_special_characters(string) > 1:
                        string_found.append(find_number_series(string)) #

        if len(string_found) ==0:

            fuzzy_string = []
            for i_string in strings:
                if len(i_string)>3:
                    fuzzy_string.append(i_string)

            for i, i_string in enumerate(fuzzy_string):
                i_string = change_char_in_string(i_string, len(i_string)-3, "/")
                fuzzy_string[i] = i_string

            strings = remove_nearly_similar_elements(fuzzy_string)
            print("fuzzy string matching combined with patterns for case number used")

        n+=1

    #%%
    if isinstance(string_found, list) and len(string_found) > 1:
        string_found = remove_duplicates(string_found)

    if isinstance(string_found, list) and len(string_found)==0:
        string_found = "None"

    elif isinstance(string_found, list) and len(string_found) == 1:
        string_found = string_found[0]

    elif isinstance(string_found, list) and len(string_found) > 1:
        p = []
        for string in string_found:
            if isinstance(string, list):
                string = string[0]
            p.append(measure_pattern(string))
        string_found = string_found[p.index(max(p))]
        if isinstance(string_found, list):
            string_found = string_found[0]

    if string_found[0].isdigit() == False and string_found[0] not in ['T', 'R', 'E'] \
            and isinstance(string_found, list):
        string_found = find_number_series(string_found)[0]
    elif string_found[0].isdigit():
        string_found = "E " + string_found
    #%%

    return string_found
