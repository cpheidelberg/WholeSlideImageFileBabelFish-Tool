import os
from glob import glob

default_wait_for_transfer_period = 1  # in seconds

def get_new_imported_slides(conf: dict):

    if not conf['debug_mode']:
        print("Config-ERROR: This tool only works on debug mode currently!")
        raise NotImplementedError("Config-ERROR: This tool only works on debug mode currently!")

    folder_name_for_imported_slides = "meta-extraction-results"
    folder_to_watch = conf['folder_to_watch']
    patterns = conf['patterns']
    all_slides = []
    for datatype in patterns:
        all_slides += [y.replace(folder_to_watch, "") for x in os.walk(folder_to_watch) for y in
                           glob(os.path.join(x[0], datatype))]

    all_slides = [p for p in all_slides if not folder_name_for_imported_slides in p]

    already_processed_slides = [y.replace(folder_to_watch + f"/{folder_name_for_imported_slides}", "").replace('.png', '').replace('error_', '')
                          for x in os.walk(folder_to_watch + f"/{folder_name_for_imported_slides}") for y in
                       glob(os.path.join(x[0], "*.ndpi*"))]

    slides_to_process = [folder_to_watch + file for file in all_slides if not file in already_processed_slides]

    return slides_to_process
