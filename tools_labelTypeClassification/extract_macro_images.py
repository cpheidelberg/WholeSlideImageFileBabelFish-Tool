import os, sys
import argparse

'''
What this script does:
- Extracts the macro images from all WSI files in a given folder (argument --in_folder) and saves them as .png files in a subfolder.
- The extracted macro images can then be used to: 
    - Create RoiSet.zip Files to configure WSI-BabelFish to process your slide-labels (see README.md chapter "How to configure roi_based_extraction.py" for detailed instructions).
    - Use our provided scripts to fine-tune a cv model to detect the label-type of a given slide (see README.md for detailed instructions).

Examples:
Given folder structure:
    - /path/to/WSIs
        - /label_type1
            - slide11.ndpi
            - slide12.ndpi
        - /label_type2
            - slide21.ndpi
            - slides2.ndpi
            
Command:
    python extract_macro_images.py --in_folder /path/to/WSIs --file_type ".ndpi"

Output:
    - /path/to/WSIs_MACROs
        - /label_type1
            - slide11.png
            - slide12.png
        - /label_type2
            - slide21.png
            - slide22.png

'''




def main():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--openslide_dll', type=str, required=False, default=None,
                        help='Path to the OpenSlide DLL directory')
    parser.add_argument('--in_folder', type=str, required=True, help='Input folder containing WSI files')
    parser.add_argument('--file_type', type=str, required=True, help='File type postfix to filter WSI files')
    parser.add_argument('--max_samples', type=str, required=False, default=None,
                        help='File type postfix to filter WSI files')
    parser.add_argument('--macro_img_tag', type=str, required=False, default='macro',
                        help='File type postfix to filter WSI files')

    args = parser.parse_args()

    if hasattr(os, 'add_dll_directory'):  # Windows
        try:
            with os.add_dll_directory(args.openslide_dll):
                import openslide
        except:
            print(f"WARNING: Failed to add necessary dll directory for openbslide!\n"
                  f"On windows, you have to pass openslide dll directory as argument --openslide_dll. "
                  f"This may cause a crash!")
            import openslide
    else:
        import openslide

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

    try:
        in_folder = args.in_folder
        file_name_postfixes = [args.file_type] if ',' not in args.file_type else args.file_type.split(',')
    except Exception as e:
        raise ValueError(f"Failed to parse arguments: {e}. \nPlease provide the input folder as first argument and the "
                         f"file name postfixes as the rest of the arguments.")

    out_folder = in_folder + '_MACROs'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # for each file in in_folder + subfolders:
    subfolders = [f.path for f in os.scandir(in_folder) if f.is_dir()]

    if not subfolders:
        subfolders = [in_folder]


    for in_folder in subfolders:
        if 'macro_' in in_folder.lower():
            continue
        num_samples_processed = 0
        print(f"Processing folder {in_folder}")
        # get the last part of the path of in_folder
        out_subfolder = os.path.basename(in_folder)
        out_sub_dir = os.path.join(out_folder, out_subfolder)
        if not os.path.exists(out_sub_dir):
            os.makedirs(out_sub_dir)
        for wsi_file_name in os.listdir(in_folder):

            if any([not wsi_file_name.endswith(file_name_postfix) for file_name_postfix in file_name_postfixes]):
                continue

            wsi_file_path = os.path.join(in_folder, wsi_file_name)
            file_type = '.' + wsi_file_name.split('.')[-1]

            macro_img = get_macro_image_from_wsi(wsi_file_path, args.macro_img_tag)
            out_path = str(out_sub_dir) + '/' + wsi_file_name.replace(file_type, '') + ".png"
            macro_img.save(out_path)
            print(f"{wsi_file_path}\t=>\t{out_path}")

            if args.max_samples is not None:
                num_samples_processed += 1
                if num_samples_processed >= int(args.max_samples):
                    break

if __name__ == "__main__":
    main()