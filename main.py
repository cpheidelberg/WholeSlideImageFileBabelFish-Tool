import shutil
from tools_MetadataExtraction.generate_slide_meta_info import get_slide_meta_data, is_valid_slide_id, is_valid_case_id
from tools_MetadataExtraction.HitchhikersGuide import HitchhickerGuide
from tools_FileObserving.SlidesWatchdog import get_new_imported_slides
import time, os, sys
import json
import datetime
import openslide
import pandas as pd
from glob import glob
import traceback

### script params: ###
conf_data_path= './tools_FileObserving/slides_meta_data_extraction_test.json'
schedule_time = 300 # seconds. can be very slow frequence. The event handler will allways be triggered, even during sleep
config = json.load(open(conf_data_path))
out_base_path = config["target_folder"]
log_folder = ".logs"
tables_folder = "tables"
save_print_to_log_file = True

if not os.path.exists(config["folder_to_watch"] + f"/{log_folder}"):
    os.makedirs(config["folder_to_watch"] + f"/{log_folder}")

if not os.path.exists(config["folder_to_watch"] + f"/{tables_folder}"):
    os.makedirs(config["folder_to_watch"] + f"/{tables_folder}")

# redirect prints into a log file:
if save_print_to_log_file:
    lof_file = config["folder_to_watch"] + f"/{log_folder}/meta-extraction-log.log"
    log = open(lof_file, "a")
    sys.stdout = log

only_extract_meta_data = False
if "only_extract_meta_data" in config:
    only_extract_meta_data = config["only_extract_meta_data"]

    ##### functions: #####
def make_hyperlink(value, appeareance=None):
    if appeareance is None:
        appeareance = value
    return '=HYPERLINK("{}","{}")'.format(value, appeareance)

def on_new_slide_created(path_to_slide=None, save_meta_in_dicom_header=False, move_slide=True):
    if path_to_slide:
        path_to_slide = path_to_slide.replace('\\', '/')
        print(f"Processing '{path_to_slide}'")

        meta_data_file_path = (out_base_path + "/" +
                               path_to_slide.replace('\\', '/').replace(config["folder_to_watch"].replace('\\', '/') + '/', "")
                               + ".import")

        if not os.path.exists(os.path.dirname(meta_data_file_path)):
            os.mkdir(os.path.dirname(meta_data_file_path))

        if os.path.isfile(meta_data_file_path):
            print(f"skipping this file, because there is already a meta-data-file for this ({meta_data_file_path})")
            return

        # get the correct LIS-db-related slides-meta-info (workaround via OCR, since roche has unusable slide-ids)
        slide_loaded = False
        for t in range(10):
            try:
                slide_reader = HitchhickerGuide(path_to_slide)
                meta_data = get_slide_meta_data(path_to_slide, slide_reader)
                slide_loaded = True
                break
            except openslide.lowlevel.OpenSlideUnsupportedFormatError as e:
                print(f"waiting for file-transfer...({e})")
                print(traceback.format_exc())
                time.sleep(2)
            except Exception as e:
                print(f"ERROR: Failed to get meta-data: {str(type(e)).split('.')[-1]}: {e}")
                sys.stdout.flush()  # update the log file
                return

        if not slide_loaded:
            print(f"ERROR: Giving up to try to load the slide...")
            sys.stdout.flush()  # update the log file
            return

        # plausibility_check:
        passed_plausibility_check = True
        if config["plausibility_check"]:
            passed_plausibility_check = False
            if "SlideId" in list(meta_data.keys()):
                if len(meta_data["SlideId"]) > 11:
                    passed_plausibility_check = is_valid_slide_id(meta_data["SlideId"])
            elif "ExamId" in list(meta_data.keys()):
                passed_plausibility_check = is_valid_case_id(meta_data["ExamId"])

            if not passed_plausibility_check:
                print("Plausibility check failed!")

        if config["debug_mode"]:
            # save macro png image:
            import matplotlib.pyplot as plt
            plt.imshow(slide_reader.SlideLabel.macro)
            plt.title(
                f"{path_to_slide.replace('\\', '/').replace(config['folder_to_watch'].replace('\\', '/'), '')}:\n"
                f"{str(meta_data).replace(', ', '\n')}" if meta_data else "no meta extracted.")
            png_results_folder = f"{config['folder_to_watch'].replace('\\', '/')}/meta-extraction-results"
            png_results_folder += os.path.dirname(path_to_slide).replace('\\', '/').replace(
                config['folder_to_watch'].replace('\\', '/'), '')
            if not os.path.exists(png_results_folder):
                os.makedirs(png_results_folder)

            if config["plausibility_check"]:
                out_file_name = f"{'' if passed_plausibility_check else 'error_'}{os.path.basename(path_to_slide)}.png"
                if not passed_plausibility_check:
                    table_name = "implausible-results"
                    error_table_path = f"{config['folder_to_watch']}/{tables_folder}/{table_name}.xlsx"
                    error_json_path = f"{config['folder_to_watch']}/{log_folder}/{table_name}.json"
                    slide_file_link = make_hyperlink(path_to_slide.replace('C:', '//CPHGPU1').replace('/', '\\'), str(os.path.basename(path_to_slide)))
                    slide_preview_link = make_hyperlink(f"{png_results_folder}/{out_file_name}".replace('C:', '//CPHGPU1').replace('/', '\\'), 'preview')

                    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

                    if not os.path.isfile(error_json_path):
                        failed_plausible_slides = {'SlideFile': [slide_file_link],
                                           'Slide-Preview': [slide_preview_link],
                                                   'date-time': [time_stamp],
                                           'ExamId': [''],
                                           'RequestId': [''],
                                           'Staining': [''],
                                           'Block': [''],
                                           'SpecimenBox': ['']
                                            }
                        print(f"creating {error_json_path}")
                    else:
                        failed_plausible_slides = json.load(open(error_json_path))
                        if not slide_file_link in failed_plausible_slides['SlideFile']:
                            failed_plausible_slides['SlideFile'].append(slide_file_link)
                            failed_plausible_slides['Slide-Preview'].append(slide_preview_link)
                            failed_plausible_slides['date-time'].append(time_stamp)
                            failed_plausible_slides['ExamId'].append('')
                            failed_plausible_slides['RequestId'].append('')
                            failed_plausible_slides['Staining'].append('')
                            failed_plausible_slides['Block'].append('')
                            failed_plausible_slides['SpecimenBox'].append('')
                            print(f"updating {error_json_path}")
                        #df.index = df.index + 1  # shifting index
                        #df = df.sort_index()  # sorting by index

                    with open(error_json_path, "w") as outfile:
                        json.dump(failed_plausible_slides, outfile)

                    try:
                        pd.DataFrame(failed_plausible_slides).to_excel(error_table_path, index=False)
                    except Exception as e:

                        pd.DataFrame(failed_plausible_slides).to_excel(
                            error_table_path.replace('.xlsx', f'_{time_stamp}.xlsx'), index=False)

            else:
                out_file_name = f"{os.path.basename(path_to_slide)}.png"

            plt.savefig(f"{png_results_folder}/{out_file_name}")

        ####### save meta data:
        if passed_plausibility_check:
            if save_meta_in_dicom_header:
                raise NotImplementedError("This feature is not yet implemented yet")
            else: # save as json
                # send the new files + meta-data to target folder:
                print(f"generated meta-data:\n{meta_data}")
                with open(meta_data_file_path, "w") as outfile:
                    json.dump(meta_data, outfile)
                print(f"Stored meta-data in {meta_data_file_path}.")


                ### move the slide to target location (not easy on windows share...):

                if move_slide:
                    new_slide_location = meta_data_file_path.replace(".import", "")

                    # possible workaround:
                    '''shutil.copy(file_name, new_slide_location)
                    file = None
                    while file is None:
                        time.sleep(1)
                        try:
                            file = open(new_slide_location)
                        except OSError:
                            print(f"transfering...")
                            continue
                    time.sleep(1)
                    os.remove(file_name)'''

                    slide_moved = False
                    for t in range(5):
                        try:
                            #os.replace(path_to_slide, meta_data_file_path.replace(".import", "")) # move the .ndpi file
                            #shutil.move(path_to_slide, meta_data_file_path.replace(".import", ""))
                            shutil.copy(path_to_slide, new_slide_location)

                            slide_moved = True
                            break
                        except Exception as e:
                            if os.path.exists(new_slide_location):
                                slide_moved = True
                                break
                            print(f"ERROR: failed to move {path_to_slide} to {meta_data_file_path.replace('.import', '')}: {e}")
                            sys.stdout.flush()
                            time.sleep(2)
                            continue

                    if not slide_moved:
                        print(f"ERROR: Giving up to try to move the file to target location...")
                    else:
                        print(f"copied {path_to_slide} to {meta_data_file_path.replace('.import', '')}.")
        else:
            print(f"Skipping import, because plausibility_check failed")

    else:
        print(f"New slide has been scanned, But invalid file path!? (got {path_to_slide})")

    sys.stdout.flush()

def generate_overview_table_of_imported_slides():
    # update excel table with all relevant infos:
    data = {'SlideFile': [], 'MetaDataFile': [], 'ResultFile': [], 'PlausibleMetaData': [],
            'SlideId': [], 'ExamId': [], 'RequestId': [], 'Staining': [], 'Block': [], 'SpecimenBox': []}

    passed_to_PACS_path = f"{config["folder_to_watch"]}/copied"

    files_passed_to_PACS = [y.replace("\\", "/") for x in os.walk(passed_to_PACS_path) for y in
                    glob(os.path.join(x[0], "*.import"))]

    for meta_file in files_passed_to_PACS:

        imported_in_pacs = True
        meta_extracted = True

        result_file = (f"{config["folder_to_watch"]}/meta-extraction-results"
                       + f"/{meta_file.split('copied/')[-1].replace('.import', '.png')}")

        if os.path.exists(result_file):
            if not "error" in result_file:
                data['PlausibleMetaData'].append(1)
            else:
                data['PlausibleMetaData'].append(0)
            data['ResultFile'].append(make_hyperlink(result_file.replace('/', '\\'), os.path.basename(result_file)))
        else:
            data['ResultFile'].append("")
            data['PlausibleMetaData'].append("???")
            meta_extracted = False

        slide_file = meta_file.replace(".import", '')
        if os.path.isfile(slide_file):
            data['SlideFile'].append(make_hyperlink(slide_file.replace('/', '\\'), os.path.basename(slide_file)))
        else:
            data['SlideFile'].append("")
            imported_in_pacs = False

        try:
            slide_meta = json.load(open(meta_file))
            data['MetaDataFile'].append(make_hyperlink(meta_file.replace('/', '\\'), os.path.basename(meta_file)))
            data['SlideId'].append(slide_meta['SlideId'] if 'SlideId' in slide_meta else "")
            data['ExamId'].append(slide_meta['ExamId'] if 'ExamId' in slide_meta else "")
            data['RequestId'].append(slide_meta['RequestId'] if 'RequestId' in slide_meta else "")
            data['Staining'].append(slide_meta['Staining']['Name'] if 'Staining' in slide_meta else "")
            data['Block'].append(slide_meta['Block']['Name'] if 'Block' in slide_meta else "")
            data['SpecimenBox'].append(slide_meta['SpecimenBox']['Name'] if 'SpecimenBox' in slide_meta else "")
        except:
            imported_in_pacs = False
            data['MetaDataFile'].append("")
            data['SlideId'].append("")
            data['ExamId'].append("")
            data['RequestId'].append("")
            data['Staining'].append("")
            data['Block'].append("")
            data['SpecimenBox'].append("")

    df = pd.DataFrame(data)
    # df.to_csv(config["folder_to_watch"] + "/meta-extraction-list.csv", sep=";", index=False)
    df.to_excel(config["folder_to_watch"] + f"/{tables_folder}/imparted_to_pacs.xlsx", index=False)


if __name__ == '__main__':

    try:
        new_slides = get_new_imported_slides(config)

        if new_slides:
            print(f"\n==== {datetime.datetime.now()} ==== ")
            print(f"Found {len(new_slides)} new slides.")
            for i, slide in enumerate(new_slides):
                print(f"\n{i+1}/{len(new_slides)})")
                try:
                    on_new_slide_created(slide, move_slide=not only_extract_meta_data)
                except Exception as e:
                    print("ERROR during main-loop!!!")
                    print(traceback.format_exc())
                print(str(datetime.datetime.now().strftime('%Y.%m.%d-%H:%M')))
                sys.stdout.flush()

            generate_overview_table_of_imported_slides()

    except Exception as e:
        print("ERROR in main.py!!!")
        print(traceback.format_exc())
        sys.stdout.flush()

