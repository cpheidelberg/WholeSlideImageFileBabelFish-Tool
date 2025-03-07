from roifile import roiread

def load_roi_set(roi_path):
    rois = roiread(roi_path)

    for roi in rois:
        print(f"ROI name: {roi.name}")
        print(f"Top: {roi.top}")
        print(f"Bottom: {roi.bottom}")
        print(f"Left: {roi.left}")
        print(f"Right: {roi.right}")
        print(f"Position: {roi.position}")
        print()

def main():
    load_roi_set('RoiSetE.zip') # test the roiset loader

if __name__ == "__main__":
    main()

