import cv2
import os
import argparse

def preprocess_image(input_path, output_dir):
    # Read image
    img = cv2.imread(input_path)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    # Binarize
    _, binary = cv2.threshold(denoised, 127, 255, cv2.THRESH_BINARY)
    # Save to output directory
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, binary)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of raw check images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for preprocessed images")
    args = parser.parse_args()

    # Process all images in input directory
    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(args.input_dir, filename)
            preprocess_image(input_path, args.output_dir)