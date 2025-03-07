import os, sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--openslide_dll', type=str, required=False, default=None, help='Path to the OpenSlide DLL directory')
    parser.add_argument('--in_folder', type=str, required=True, help='Input folder containing WSI files')
    parser.add_argument('--file_type', type=str, required=True, help='File type postfix to filter WSI files')
    parser.add_argument('--max_samples', type=str, required=False, default=None, help='File type postfix to filter WSI files')

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

    try:
        in_folder = args.in_folder
        file_name_postfixes = [args.file_type] if ',' not in args.file_type else args.file_type.split(',')
    except Exception as e:
        raise ValueError(f"Failed to parse arguments: {e}. \nPlease provide the input folder as first argument and the "
                         f"file name postfixes as the rest of the arguments.")

    num_samples_processed = 0
    for wsi_file_name in os.listdir(in_folder):

        if any([not wsi_file_name.endswith(file_name_postfix) for file_name_postfix in file_name_postfixes]):
            continue

        wsi_file_path = os.path.join(in_folder, wsi_file_name)

        file_type = '.' + wsi_file_name.split('.')[-1]
        try:
            wsi = openslide.OpenSlide(wsi_file_path)
        except Exception as e:
            print(f"Failed to load {wsi_file_path} due to loading error: {e}")
        macro_img = wsi.associated_images['macro']
        out_path = wsi_file_path.replace(file_type, '') + "_macro.png"
        macro_img.save(out_path)
        print(f"Extracted \t{out_path}")

        if args.max_samples is not None:
            num_samples_processed += 1
            if num_samples_processed >= int(args.max_samples):
                break

if __name__ == "__main__":
    main()