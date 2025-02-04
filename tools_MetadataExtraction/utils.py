
import os

def clean_up(i_img):

    label_file = i_img[0:-4] + "_label.jpg"

    if os.path.exists(label_file):
        os.remove(label_file)

    error_file = i_img[0:-4] + "_error.jpg"

    if os.path.exists(error_file):
        os.remove(error_file)


def find_highest_number(strings):
    highest_number = None
    for i_string in strings:
        for s in i_string:
            try:
                # Try to convert the string to a float
                num = float(s)
                # Update the highest number if necessary
                if highest_number is None or num > highest_number:
                    highest_number = num
            except ValueError:
                # Ignore strings that cannot be converted to a number
                continue

    return highest_number