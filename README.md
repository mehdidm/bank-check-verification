# Bank Check Data Extraction

A comprehensive tool for extracting data from bank checks using computer vision and OCR techniques.

## Features

- **Dynamic Region Detection**: Automatically detects check regions using advanced computer vision techniques and machine learning models for improved accuracy
- **Adaptive Processing**: Identifies key regions like MICR line, amount box, payee line, and signature
- **Fixed Region Fallback**: Uses predefined coordinates as fallback for challenging images
- **Advanced Preprocessing**: Includes deskewing, denoising, thresholding, and contrast enhancement
- **Optimized OCR**: Configured for best performance with check data
- **Batch Processing**: Process multiple check images at once
- **Visualization**: Generate visual reports of extracted data
- **Configurable Parameters**: Fine-tune detection parameters for specific check types

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/check_extractor.git
cd check_extractor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

The main script provides a command-line interface for processing check images:

```bash
python -m src.main --image path/to/check.jpg
```

#### Available Options

| Option | Description |
|--------|-------------|
| `--image` | Path to single check image |
| `--dir` | Path to directory containing check images |
| `--output` | Directory for output files |
| `--config` | Directory containing configuration files |
| `--detection-params` | Path to detection parameters file |
| `--check-type` | Type of check to use specific parameters (default: 'default') |
| `--method` | Region extraction method (`fixed` or `dynamic`) |
| `--transformer` | Use transformer-based OCR |
| `--no-deskew` | Disable deskewing |
| `--no-denoise` | Disable denoising |
| `--no-enhance` | Disable image enhancement |
| `--threshold` | Thresholding method (`adaptive`, `otsu`, or `none`) |

### Examples

#### Process a Single Check Image

```bash
python -m src.main --image data/checks/check1.jpg --output results
```

#### Process a Batch of Check Images

```bash
python -m src.main --dir data/checks --output results
```

#### Use Fixed Region Detection

```bash
python -m src.main --image data/checks/check1.jpg --method fixed --output results
```

#### Use Specific Check Type Parameters

```bash
python -m src.main --image data/checks/zitouna_check.jpg --check-type zitouna_bank --output results
```

## Project Structure

```
check_extractor/
├── src/
│   ├── main.py            # Main entry point and CheckExtractor class
│   ├── preprocessor.py    # Image preprocessing functionality
│   ├── region_detector.py # Region detection (dynamic and fixed)
│   ├── text_recognizer.py # OCR functionality
│   └── visualizer.py      # Result visualization
├── data/
│   └── checks/            # Check images to process
├── config/
│   ├── regions_config.json       # Fixed region coordinates
│   ├── detection_params.json     # Parameters for dynamic detection
│   └── extraction_patterns.json  # Patterns for data extraction
└── results/               # Output directory for results
```

## How Dynamic Region Detection Works

The system uses several computer vision techniques to automatically identify regions on a check:

1. **MICR Line Detection**: 
   - Focuses on the bottom part of the check
   - Uses morphological operations to enhance horizontal lines
   - Identifies the largest contour which is likely the MICR line

2. **Amount Box Detection**:
   - Analyzes the top-right part of the check
   - Uses adaptive thresholding to enhance box edges
   - Filters contours by area and aspect ratio to find rectangular boxes

3. **Payee Line Detection**:
   - Examines the middle-left part of the check
   - Applies horizontal line detection using Hough transform
   - Identifies the longest horizontal line which is likely the payee line

4. **Date Line Detection**:
   - Focuses on the top-right corner
   - Uses edge detection and Hough transform to find horizontal lines
   - Selects the topmost line as the date line

5. **Signature Detection**:
   - Analyzes the bottom-right part of the check
   - Uses adaptive thresholding to enhance the signature
   - Combines all contours to identify the signature area

6. **Written Amount Detection**:
   - Examines the middle part of the check
   - Identifies horizontal lines that typically bound the written amount
   - Extracts the region between these lines

If any region cannot be detected automatically, the system falls back to predefined coordinates based on typical check layouts.

## Configurable Parameters

The system allows fine-tuning of detection parameters for specific check types through the `detection_params.json` file. This file contains parameters for each region detection method, OCR settings, and preprocessing options.

### Detection Parameters Structure

```json
{
  "check_type_name": {
    "micr_line": {
      "focus_area": {"y1": 0.7, "y2": 1.0, "x1": 0.0, "x2": 1.0},
      "kernel_width_factor": 50,
      "min_contour_area": 100
    },
    "amount_box": {
      "focus_area": {"y1": 0.0, "y2": 0.4, "x1": 0.6, "x2": 1.0},
      "min_area": 100,
      "aspect_ratio_min": 2.0,
      "aspect_ratio_max": 5.0
    },
    "payee_line": {
      "focus_area": {"y1": 0.25, "y2": 0.5, "x1": 0.0, "x2": 0.8},
      "kernel_width_factor": 30,
      "min_line_length_factor": 0.33,
      "max_line_gap": 20
    },
    "date_line": {
      "focus_area": {"y1": 0.0, "y2": 0.2, "x1": 0.6, "x2": 1.0},
      "kernel_width": 15,
      "min_line_length_factor": 0.1,
      "max_line_gap": 10
    },
    "signature": {
      "focus_area": {"y1": 0.5, "y2": 0.8, "x1": 0.5, "x2": 1.0},
      "padding": 10
    },
    "written_amount": {
      "focus_area": {"y1": 0.35, "y2": 0.55, "x1": 0.1, "x2": 0.8},
      "kernel_width_factor": 30,
      "min_line_length_factor": 0.25,
      "max_line_gap": 20
    },
    "ocr_settings": {
      "contrast_ths": 0.2,
      "text_threshold": 0.6,
      "low_text": 0.3,
      "width_ths": 0.7,
      "mag_ratio": 2.5
    },
    "preprocessing": {
      "adaptive_threshold": true,
      "clahe": true,
      "denoise": true
    }
  }
}
```

### Parameter Descriptions

#### Region Detection Parameters
- **focus_area**: Defines the area of the image to focus on for each region (as percentages of image dimensions)
- **kernel_width_factor**: Controls the size of morphological kernels for line detection
- **min_contour_area**: Minimum area for contours to be considered valid
- **aspect_ratio_min/max**: Range of valid aspect ratios for rectangular regions
- **min_line_length_factor**: Minimum length for lines as a factor of image width
- **max_line_gap**: Maximum gap between line segments to be connected

#### OCR Settings
- **contrast_ths**: Contrast threshold for text detection
- **text_threshold**: Confidence threshold for text detection
- **low_text**: Threshold for detecting small text
- **width_ths**: Width threshold for character detection
- **mag_ratio**: Magnification ratio for small text detection

#### Preprocessing Settings
- **adaptive_threshold**: Whether to apply adaptive thresholding
- **clahe**: Whether to apply CLAHE for contrast enhancement
- **denoise**: Whether to apply denoising

## Advanced Usage

### Programmatic API

You can also use the CheckExtractor class programmatically:

```python
from src.main import CheckExtractor

# Initialize the extractor with specific check type
extractor = CheckExtractor(
    config_dir='config',
    output_dir='results',
    detection_params_path='config/detection_params.json',
    check_type='zitouna_bank'
)

# Process a single check
result = extractor.process_check(
    'data/checks/check1.jpg',
    preprocessing_params={
        'deskew': True,
        'denoise_strength': 10,
        'threshold_method': 'adaptive',
        'enhance': True
    },
    region_method='dynamic'
)

# Access extracted data
print(result['extracted_data'])
```

## Troubleshooting

### Poor Region Detection

If regions are not detected correctly:
1. Try adjusting detection parameters in the `detection_params.json` file
2. Create a specific check type configuration for your check format
3. Use fixed region detection (`--method fixed`) if you have a configuration file
4. Check if the image is properly aligned and has good contrast

### Poor OCR Results

If OCR results are not accurate:
1. Adjust OCR settings in the `detection_params.json` file
2. Ensure the preprocessing pipeline is optimized
3. Check if regions are correctly detected
4. Consider using the transformer-based OCR option for better accuracy

## License

This project is licensed under the MIT License - see the LICENSE file for details.