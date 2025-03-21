import os
from PIL import Image
import numpy as np
import argparse

def load_images_from_folder(folder):
    # List to store all the images
    images = []
    for filename in os.listdir(folder):
        if filename.endswith('.png'):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)
            images.append(img)
    return images


def check_and_resize_images(images):
    # Get the resolution of the first image
    target_size = images[0].size
    resized_images = []

    for i, img in enumerate(images):
        if img.size != target_size:
            print(f"Image {i + 1} has a different resolution. Resizing to {target_size}.")
            img = img.resize(target_size)
        resized_images.append(img)

    return resized_images, target_size


def calculate_average_image(images):
    # Convert all images into numpy arrays
    image_arrays = [np.array(img) for img in images]

    # Calculate the average image (pixel-wise mean)
    average_image_array = np.mean(image_arrays, axis=0).astype(np.uint8)

    # Return the average image as a PIL Image object
    return Image.fromarray(average_image_array)


def main():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--in_folder', type=str, required=True, help='Input folder containing WSI files')

    args = parser.parse_args()

    #for class_folder in os.listdir(args.in_folder):
    #class_folder = os.path.join(args.in_folder, class_folder)

    class_folder = args.in_folder

    if not os.path.isdir(class_folder):
        raise ValueError(f"Provided path is not a directory: {class_folder}")

    #class_folder = "../data/slide-macro-set2/CMCP"  # Provide the path to the folder with the .png images
    out_file = f"{class_folder}_mean.png"  # Provide the path to save the average image

    images = load_images_from_folder(class_folder)
    print(f"=== {len(images)} images were found in {class_folder} ===")

    # Check if all images have the same resolution, and resize if necessary
    resized_images, target_size = check_and_resize_images(images)
    print(f"Target resolution is set to {target_size}.")

    # Calculate the average image
    average_image = calculate_average_image(resized_images)

    # Save the average image
    average_image.save(out_file)
    print("The average image has been saved as ", out_file)

    print()


if __name__ == "__main__":
    main()
