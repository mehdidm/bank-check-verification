import cv2
import numpy as np
import os
import logging
from PIL import Image


class CheckPreprocessor:
    """
    Advanced image preprocessing for bank checks
    Handles various image enhancement techniques
    """

    def __init__(self, debug=False):
        """
        Initialize preprocessor with optional debug mode

        Args:
            debug (bool): Enable detailed logging and intermediate image saving
        """
        self.debug = debug
        self.logger = logging.getLogger(__name__)

    def preprocess_image(self, image_path):
        """
        Comprehensive image preprocessing pipeline

        Args:
            image_path (str): Path to the input image

        Returns:
            numpy.ndarray: Preprocessed binary image
        """
        try:
            # Read image (support multiple formats)
            image = self._read_image(image_path)

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Noise reduction
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

            # Contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)

            # Binarization with Otsu's method
            _, binary = cv2.threshold(enhanced, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Deskew (optional, advanced rotation correction)
            binary = self._deskew(binary)

            # Debug: save intermediate images
            if self.debug:
                self._save_debug_images(gray, denoised, enhanced, binary)

            return binary

        except Exception as e:
            self.logger.error(f"Preprocessing error: {e}")
            raise

    def _read_image(self, image_path):
        """
        Read image with multiple format support

        Args:
            image_path (str): Path to the image file

        Returns:
            numpy.ndarray: Loaded image
        """
        # Handle PDF conversion if needed
        if image_path.lower().endswith('.pdf'):
            from pdf2image import convert_from_path
            images = convert_from_path(image_path)
            image = np.array(images[0])
        else:
            image = cv2.imread(image_path)

        return image

    def _deskew(self, binary_image):
        """
        Correct image skew/rotation

        Args:
            binary_image (numpy.ndarray): Binary image

        Returns:
            numpy.ndarray: Deskewed image
        """
        # Implement deskew using moments method
        coords = np.column_stack(np.where(binary_image > 0))
        angle = cv2.minAreaRect(coords)[-1]

        # Adjust for OpenCV angle measurement
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Rotate image
        (h, w) = binary_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            binary_image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated

    def _save_debug_images(self, gray, denoised, enhanced, binary):
        """
        Save intermediate preprocessing images for debugging

        Args:
            gray (numpy.ndarray): Grayscale image
            denoised (numpy.ndarray): Denoised image
            enhanced (numpy.ndarray): Contrast-enhanced image
            binary (numpy.ndarray): Binary image
        """
        debug_dir = 'data/debug_preprocessing'
        os.makedirs(debug_dir, exist_ok=True)

        cv2.imwrite(os.path.join(debug_dir, 'gray.jpg'), gray)
        cv2.imwrite(os.path.join(debug_dir, 'denoised.jpg'), denoised)
        cv2.imwrite(os.path.join(debug_dir, 'enhanced.jpg'), enhanced)
        cv2.imwrite(os.path.join(debug_dir, 'binary.jpg'), binary)

    def save_image(self, image, path):
        """
        Save processed image to specified path

        Args:
            image (numpy.ndarray): Image to save
            path (str): Output file path
        """
        cv2.imwrite(path, image)