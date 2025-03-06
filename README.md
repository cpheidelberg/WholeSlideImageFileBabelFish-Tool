

#  WholeSlideImageFileBabelFish-Tool

This python project was created as part of the article \
_"Presenting the framework of the whole slide image file Babel fish: An OCR-based file labeling tool"._ \
The paper can be found here: https://doi.org/10.1016/j.jpi.2024.100402 

WSI-BabelFish is a tool to extract meta information
like case number, year, slide number, block number etc. 
automatically from macro images of digital slides.

WSI-BabelFish is based on optical character recognition (OCR). For most information `easyOCR` is used. 
For the block number and cases with not enough results in the first OCR round, a second OCR with `pytesseract` is applied. 

## Intended use

This diagram shows the intended use of WSI-BabelFish:

![wsi-babelfish-schema](./wsi-babelfish-schema.png)

It is designed to operate on an infrastructure where digital slides are uploaded to a network share (here: `folder_to_watch`).

WSI-BabelFish will watch this folder and extract the meta information from the uploaded slides using OCR tools.

The extracted meta information is saved as json file in the `target_folder` and the slide is moved to `target_folder`.

The extracted meta information can be used to import the slides into a database or PACS or LIS or to rename the slides.

A visualization of the results will be saved in the `folder_to_watch/meta-extraction-results/` directory as png files.

BableFish does also perform a plausibility check on the extracted meta information and will not save the json file if the plausibility check fails.
Failed slides will be collected in a table at `folder_to_watch/tables/implausible-results.xlsx`. This table can be used to manually correct the extracted meta information.
And a corrected table can be used to re-import the slides and to fine-tune the OCR over time.

Successful extracted data will be collected in a table at `folder_to_watch/tables/imparted_to_pacs.xlsx`. 

Errors and warnings will be logged in `folder_to_watch/.logs/` (this is a hidden directory!).

## How to configure WSI-BabelFish

Can be configured by editing the file `tools_FileObserving/slides_meta_data_extraction.json`:

WSI-BabelFish will watch the folder `folder_to_watch` (can be set in `slides_meta_data_extraction.json`) and trigger an event if
new files which match pattern `patterns` have been uploaded to `folder_to_watch`.

WSI-BabelFish will then extract the meta-data of the new imported slide and save it as json at 
`target_folder/<slide-file>.ndpi.import`. 

After the metadata extraction is done, WSI-BabelFish moves the slide to `target_folder/slide-file.ndpi` if, `only_extract_meta_data` is set to `False` in the config file (`slides_meta_data_extraction.json`).

WSI-BabelFish needs to be executed frequently using a task scheduler.
For this, one need to set up a scheduled task so that `/path/to/this/repo/main.py` gets executed each x minute. For this, see the next chapter(s):

#### windows:
We had some trouble to make WSI-BabelFish work together with the windows task scheduler. This was our solution:

1. Open the Windows Task Scheduler
2. Create a new task
3. Set the action to run a program
4. In the action-settings, enter `/path/to/this/repo/execute.bat` for "Program/Script" and enter `/path/to/this/repo` for "Start in" so that this will be used as working directory.
5. Now modify the `execute.bat` accordingly to your environment. 

#### linux:
On linux one can use crontab to schedule the execution of `/path/to/this/repo/main.py`. 
Feel free to edit this chapter if you have some experience with it!

#### How to configure ROI-configurations:
WSI Babelfish needs to know which information (staining-letters, block-num, etc...) is located in which region on the label of the slide.

For this, WSI-BabelFish can load ROI-configuration files. (ROI = Region of Interest).

To create a new ROI-configuration file, one can use the scripts located in `./tools_ROIconfig` and some open source tools:

1. Download and install [ImageJ](https://imagej.net/ij/).
2. Get a macro-image (png or jpg) of one of your slide-files. To extract a macro-image from a WSI, you can use the `extract_macro_image.py` script in `./tools_ROIconfig`.
3. Open the macro-image in ImageJ.
4. Use the `Rectangle`-Tool to draw ROI's. Add each rectangle to a ROI-set using right-mouse click -> `add to ROI-Manager` and name each ROI accordingly to what information is located in this region.
5. In the ROI manager, select all ROIs and export them as a RoiSet zip file and copy it into the `./tools_ROIconfig` directory.
6. todo... implement ROI-configurations...

#### How to adapt the OCR tools:

Unfortunately, the OCR tools are not perfect and need to be adapted to the specific use case. 

For this, one can edit the `tools_MetadataExtraction/HitchhikersGuide.py` code.
Moreover, the function `get_slide_meta_data` in `tools_MetadataExtraction/extract_meta_data.py` can be adapted to change how the metadata dictionary should be built.

Implement/adjust the function `is_valid_slide_id` and or `is_valid_case_id` in `tools_MetadataExtraction/extract_meta_data.py` to adjust how the plausibility check should be done on your extracted metadata.



## Installation

We recommend to first create a conda environment to install WSI-BabelFish, e.g. using 

```
conda create --name wsi_babblefish
conda activate wsi_babblefish
```

Next, install the following packages:

### Basic python packages:

(allways use `python -m pip` to ensure that it will be installed in the activated conda environment)

```
python -m pip install matplotlib
python -m pip install openpyxl
python -m pip install pandas
python -m pip install rapidfuzz
python -m pip install thefuzz
python -m pip install tqdm
python -m pip install icecream

```

### Install openslide:
OpenSlide Python requires OpenSlide, which must be installed separately.

On Linux and macOS, the easiest way to get both components is to use conda:

`conda install -c conda-forge openslide openslide-python`

On windows, see: https://openslide.org/api/python/#installing, then use `pip install openslide-python`

### Install easy OCR 
there will be things like torchvision be installed

``python -m pip install easyocr``

### Install pylibdmtx:

on Linux/OSX use:
```
pip install pylibdmtx or conda install conda-forge::pylibdmtx

brew install libdmtx
```  
on windows, get the `libdmtx.dll` manually from the web. We've found it [here](https://github.com/NaturalHistoryMuseum/pylibdmtx/issues/64).

Then, in `tools_MetadataExtraction/HitchhikersGuide.py`, import it like this in our python scripts:

```
PYLIBDMTX_PATH = r'C:/Users/.../.conda/envs\.../Lib/site-packages/pylibdmtx/libdmtx-64bit.dll'
   with os.add_dll_directory(PYLIBDMTX_PATH):
       from pylibdmtx.pylibdmtx import decode
```

### Install pytesseract:

**On windows:**

Download and install tesseract 5.3:

https://tesseract-ocr.github.io/tessdoc/Installation.html

Add `\path\where\installed\Tesseract-OCR` to path environment variable

Check in terminal if command `tesseract` works.

Then install pytesseract into your environment using ``python -m pip install pytesseract``

**On Linux:**

Feel free to edit this chapter if you have some experience with it!

## Usage

### 

## Contribute
Contributions are very welcome! Here's how to get involved:

1. Clone or fork the repository.
2. Make your changes or improvements.
3. Create a pull request.
4. If you find any bugs or have suggestions, please log them here as well.

(Include additional details on development environment setup, coding style, testing, and issue reporting as needed.)

## How to cite

If you use WSI-BabelFish in your research, please cite the following paper:

```
@article{ENGLERT2024100402,
    title = {Presenting the framework of the whole slide image file Babel fish: An OCR-based file labeling tool},
    journal = {Journal of Pathology Informatics},
    volume = {15},
    pages = {100402},
    year = {2024},
    issn = {2153-3539},
    doi = {https://doi.org/10.1016/j.jpi.2024.100402},
    url = {https://www.sciencedirect.com/science/article/pii/S2153353924000415},
    author = {Nils Englert and Constantin Schwab and Maximilian Legnar and Cleo-Aron Weis},
    keywords = {DICOM, Digital pathology, Optical character recognition, Automatization}
}
```

---
**Authors of this Repo:** Cleo-Aron Weis and Maximilian Legnar <br>
**Contact:** [cleo-aron.weis@uni-heidelberg.de](mailto:cleo-aron.weis@uni-heidelberg.de)
