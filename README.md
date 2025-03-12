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

## Dynamic Region Detection Guide

### How to Use Dynamic Detection for Any Check

The system is designed to automatically detect regions on any check without requiring predefined templates. Here's how to use it effectively:

1. **Basic Usage with Default Settings**:
   ```bash
   python -m src.main --image path/to/your/check.jpg --method dynamic --output results
   ```
   This will apply the default detection parameters which are designed to work with most standard check formats.

2. **For Specific Check Types**:
   ```bash
   python -m src.main --image path/to/your/check.jpg --check-type your_bank_name --output results
   ```
   This will use parameters optimized for your specific check type (if defined in the configuration).

### Adding Support for a New Check Type

If you have a new check type that requires specific detection parameters:

1. **Create a new entry in the detection parameters file** (`config/detection_params.json`):
   ```json
   "your_bank_name": {
     "micr_line": {
       "focus_area": {"y1": 0.8, "y2": 1.0, "x1": 0.0, "x2": 1.0},
       "kernel_width_factor": 45,
       "min_contour_area": 120
     },
     "amount_box": {
       "focus_area": {"y1": 0.0, "y2": 0.3, "x1": 0.7, "x2": 1.0},
       "min_area": 110,
       "aspect_ratio_min": 2.2,
       "aspect_ratio_max": 4.8
     },
     "ocr_settings": {
       "contrast_ths": 0.2,
       "text_threshold": 0.7,
       "low_text": 0.4,
       "width_ths": 0.8,
       "mag_ratio": 2.0
     },
     "preprocessing": {
       "adaptive_threshold": true,
       "clahe": true,
       "denoise": true
     }
   }
   ```

2. **Run the extractor with your check type**:
   ```bash
   python -m src.main --image path/to/your/check.jpg --check-type your_bank_name --output results
   ```

### Tips for Improving Region Detection

1. **Adjust Focus Areas**: 
   - Each region has a `focus_area` parameter that defines where to look for that region
   - Values are percentages of the image dimensions (0.0 to 1.0)
   - For example, to look for the MICR line in the bottom 20% of the image:
     ```json
     "focus_area": {"y1": 0.8, "y2": 1.0, "x1": 0.0, "x2": 1.0}
     ```

2. **Tune Detection Parameters**:
   - `kernel_width_factor`: Controls the size of morphological kernels for line detection
   - `min_contour_area`: Minimum area for contours to be considered valid
   - `aspect_ratio_min/max`: Range of valid aspect ratios for rectangular regions
   - `min_line_length_factor`: Minimum length for lines as a factor of image width
   - `max_line_gap`: Maximum gap between line segments to be connected

3. **Optimize Preprocessing**:
   - Adjust preprocessing parameters for better image quality:
     ```json
     "preprocessing": {
       "adaptive_threshold": true,
       "clahe": true,
       "denoise": true
     }
     ```

4. **Fine-tune OCR Settings**:
   - Adjust OCR parameters for better text recognition:
     ```json
     "ocr_settings": {
       "contrast_ths": 0.2,
       "text_threshold": 0.6,
       "low_text": 0.3,
       "width_ths": 0.7,
       "mag_ratio": 2.5
     }
     ```

### Troubleshooting Region Detection

If regions are not being detected correctly:

1. **Check Image Quality**:
   - Ensure the check image is clear, well-lit, and high-resolution
   - Remove any background noise or shadows

2. **Visualize Detection Results**:
   - Examine the output visualization to see what regions were detected
   - Look for patterns in missed regions

3. **Adjust Focus Areas**:
   - If a region is consistently missed, try expanding its focus area
   - For example, if the date line is too high in the image:
     ```json
     "date_line": {
       "focus_area": {"y1": 0.0, "y2": 0.25, "x1": 0.6, "x2": 1.0},
       ...
     }
     ```

4. **Fallback to Fixed Coordinates**:
   - If dynamic detection consistently fails for a specific check type, you can use fixed coordinates as a fallback:
     ```bash
     python -m src.main --image path/to/your/check.jpg --method fixed --output results
     ```

## GUI Application

The Check Extractor now includes a graphical user interface for easier interaction with the system. The GUI provides a user-friendly way to process checks without using command-line arguments.

### Running the GUI

```bash
# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install pillow opencv-python-headless pytesseract

# Launch the GUI application
python gui_app.py
```

### GUI Features

- **Image Loading**: Browse and select individual check images for processing
- **Batch Processing**: Process multiple check images from a directory
- **Detection Method Selection**: Choose between dynamic and fixed region detection
- **Check Type Selection**: Select from available check types defined in the configuration
- **Preprocessing Options**: Configure image preprocessing settings
- **Real-time Visualization**: View the original and processed check images
- **Results Display**: See extracted data in a readable format
- **PDF Report Generation**: Automatically create professional PDF reports with check images and extracted data

### Using the GUI

1. **Load an Image**: Click the "Browse" button next to "Check Image" to select a check image
2. **Configure Settings**: Select the detection method, check type, and preprocessing options
3. **Process Check**: Click the "Process Check" button to extract data from the loaded image
4. **View Results**: The extracted data will be displayed in the "Extracted Data" section
5. **Access PDF Reports**: PDF reports are automatically generated in the results/reports directory
6. **Process Multiple Checks**: For batch processing, select a directory containing check images and click "Process Batch"

### GUI Implementation Details

The GUI is built using Tkinter, Python's standard GUI toolkit, and follows a modular design pattern:

#### Architecture

- **Main Application Class**: `CheckExtractorGUI` in `src/gui.py` handles all GUI interactions
- **Thread-Safe Processing**: Long-running operations run in background threads to keep the UI responsive
- **Event-Driven Design**: Uses Tkinter's event system for user interactions
- **MVC Pattern**: Separates the UI (View) from the check processing logic (Model) with the GUI class acting as Controller

#### Key Components

1. **Left Panel**: Contains input fields and settings controls
   - Image selection with file browser
   - Batch directory selection
   - Detection method radio buttons (Dynamic/Fixed)
   - Check type dropdown (populated from detection_params.json)
   - Preprocessing options (Deskew, Denoise, Enhance)
   - Threshold method selection
   - Action buttons (Process Check, Process Batch, Update Settings)

2. **Right Panel**: Displays the check image with scroll functionality
   - Automatically scales images to fit the view
   - Supports panning with scrollbars
   - Shows the original image or processed visualization

3. **Bottom Panel**: Shows status and results
   - Status bar for operation feedback
   - Text area for displaying extracted check data

#### Integration with Check Extractor

The GUI integrates with the existing check extraction pipeline:

1. **Configuration Loading**: Automatically loads available check types from detection_params.json
2. **Parameter Passing**: Translates GUI settings into appropriate parameters for the CheckExtractor class
3. **Result Handling**: Displays extracted data and visualizations from the processing pipeline
4. **Error Management**: Catches and displays errors in a user-friendly way

#### Asynchronous Processing

To maintain UI responsiveness during processing:

1. Check processing runs in separate threads
2. A queue system passes results back to the main thread
3. Periodic queue checking updates the UI with processing status and results
4. Progress updates are shown in the status bar

#### Customization

The GUI respects all configuration options from the detection_params.json file:

1. New check types added to the configuration automatically appear in the GUI dropdown
2. OCR settings from the configuration are applied during processing
3. Preprocessing parameters can be adjusted through the interface
4. All command-line options are accessible through the GUI

#### PDF Reports

The GUI automatically generates comprehensive PDF reports for each processed check:

1. **Individual Check Reports**: For each successfully processed check, a detailed PDF report is generated containing:
   - Check image with detected regions highlighted
   - All extracted data in a tabular format
   - Confidence scores for OCR results
   - Date and time of processing
   - Filename and check type information

2. **Batch Summary Reports**: When processing multiple checks, a summary PDF is also generated with:
   - Overview of all processed checks
   - Success/failure status for each check
   - Links to individual check reports
   - Processing timestamp and statistics

3. **Report Location**: All PDF reports are saved in the `results/reports` directory with timestamps in the filename for easy identification

## Project Structure

```
check_extractor/
├── gui_app.py             # GUI application entry point
├── src/
│   ├── gui.py             # GUI implementation
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