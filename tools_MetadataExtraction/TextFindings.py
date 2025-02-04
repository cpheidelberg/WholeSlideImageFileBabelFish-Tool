
#%% import section
import pandas as pd
import numpy as np
import cv2
from matplotlib import patches, text, patheffects
import matplotlib.pyplot as plt
import pickle

#%% helper functions
def left_top_width_height_to_coordinates(left, top, width, height):
    A = [left, top]
    B = [left + width, top]
    C = [left + width, top + height]
    D = [left, top + height]
    return [A, B, C, D]

def coordinates_to_left_top_width_height(points):
    A, B, C, D = points[0], points[1], points[2], points[3]
    left = A[0]
    top = A[1]
    width = B[0] - left
    height = C[1] -top
    return (left, top, width, height)

def calculate_iou(box1, box2):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes.

    Args:
        box1 (tuple): Coordinates of the first bounding box in the format (x1, y1, x2, y2).
        box2 (tuple): Coordinates of the second bounding box in the format (x1, y1, x2, y2).

    Returns:
        float: IoU value.
    """
    x1, y1, w1, h1 = box1[0], box1[1], box1[2], box1[3]
    x2, y2, w2, h2 = box2[0], box2[1], box2[2], box2[3]

    # Calculate coordinates of the intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    # If the boxes don't intersect, return 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate area of each bounding box
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Calculate union area
    union_area = box1_area + box2_area - intersection_area

    if box1_area > box2_area:
        area2normalize = box2_area
    else:
        area2normalize = box1_area

    # Calculate IoU (or overlap normalized to smaller box
    if area2normalize == 0:
        iou = 1
    else:
        iou = intersection_area / area2normalize

    return iou

#%% define class
class TextFindings(pd.DataFrame):
    def __init__(self, datadict, label= None, *args, **kwargs):
        super().__init__(datadict, *args, **kwargs)

        self.attrs['label'] = label
        self.attrs['region_case'] = None
        self.attrs['region_block'] = None

    def boxes(self):
        boxes = []
        for i, i_points in enumerate(self.positions.tolist()):
            boxes.append(coordinates_to_left_top_width_height(i_points))
        self['box'] = boxes

    def plot(self, label=None):

        if label is None:
            label = self.attrs['label']

        if isinstance(label, str):
            label = cv2.imread(label)

        boxes = []
        for i, i_points in enumerate(self.position.tolist()):
            boxes.append(coordinates_to_left_top_width_height(i_points))
        self['box'] = boxes

        plt.imshow(label)
        for i, box in enumerate(self.box):

            box_color = "b"
            if "method" in self.columns:
                if self.method[i] == "pytesseract":
                    box_color = "r"

            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=2,
                                     edgecolor=box_color, facecolor='none')
            plt.add_patch(rect)

        if "region_case" in self.attrs.keys():
            box = self.attrs['region_case']
            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=2,
                                     edgecolor="orange", facecolor='none')
            plt.add_patch(rect)

        #plt.savefig("test.png")
        plt.show()

    def dump_it(self, file_name = "TextFound.pkl"):

        columns_as_lists = {}
        for column in self.columns:
            columns_as_lists[column] = self[column].tolist()

        # Step 4: Save these lists to a pickle file
        with open(file_name, 'wb') as f:
            pickle.dump(columns_as_lists, f)

    def _plot(self, ax, label = None):

        if label is None:
            label = self.attrs['label']

        if isinstance(label, str):
            label = cv2.imread(label)

        boxes = []
        for i, i_points in enumerate(self.position.tolist()):
            boxes.append(coordinates_to_left_top_width_height(i_points))
        self['box'] = boxes

        ax.imshow(label)
        for i, box in enumerate(self.box):

            box_color = "b"
            if "method" in self.columns:
                if self.method[i] == "pytesseract":
                    box_color = "r"
                elif "region" in self.columns and self.method[i] == "pytesseract":
                    if self.region == 1:
                        box_color = "r"
                    else:
                        box_color = "y"

            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=2,
                                     edgecolor=box_color, facecolor='none')
            ax.add_patch(rect)

    def plot_finding(self, label = None):

        if label is None:
            label = self.attrs['label']

        if isinstance(label, str):
            label = cv2.imread(label)

        boxes = []
        for i, i_points in enumerate(self.position.tolist()):
            boxes.append(coordinates_to_left_top_width_height(i_points))
        self['box'] = boxes

        fig, axs = plt.subplots(ncols=2, nrows=1)
        axs[0].imshow(label)
        for i, box in enumerate(self.box):

            box_color = "b"
            if "method" in self.columns:
                if self.method[i] == "pytesseract":
                    box_color = "r"

            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=2,
                                     edgecolor=box_color, facecolor='none')
            axs[0].add_patch(rect)

        if "region_case" in self.attrs.keys():
            box = self.attrs['region_case']
            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=2,
                                     edgecolor="orange", facecolor='none')
            axs[0].add_patch(rect)

        if "region_block" in self.attrs.keys():
            box = self.attrs['region_block']
            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=2,
                                     edgecolor="y", facecolor='none')
            axs[0].add_patch(rect)
        # plt.savefig("test.png")

        axs[1].imshow(np.ones(label.shape))

        for i, text in enumerate(self.finding):

            text_color = "b"
            if "method" in self.columns:
                if self.method[i] == "pytesseract":
                    text_color = "r"

            axs[1].text(self.box[i][0], self.box[i][1] + 10,
                     text, fontsize=12, color = text_color)

        if "region_case" in self.attrs.keys():
            box = self.attrs['region_case']
            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=2,
                                     edgecolor="y", facecolor='none')
            axs[1].add_patch(rect)

        if "region_block" in self.attrs.keys():
            box = self.attrs['region_block']
            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=2,
                                     edgecolor="y", facecolor='none')
            axs[1].add_patch(rect)

        plt.show()

#%% testing
if __name__ == "__main__":
    label = cv2.imread("tools_MetadataExtraction/TextFound_example.png")

    with open('tools_MetadataExtraction/TextFound_example.pkl', 'rb') as f:
        data= pickle.load(f)

    with open('tools_MetadataExtraction/TextFound_example_attrs.pkl', 'rb') as f:
        attrs= pickle.load(f)

    TextFound = TextFindings(data, label)
    TextFound.attrs = attrs

