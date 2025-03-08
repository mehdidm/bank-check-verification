import os
import argparse
import cv2
import logging
import json
from datetime import datetime

# Import project modules
from .preprocessor import ImagePreprocessor
from .region_detector import RegionDetector
from .text_recognizer import TextRecognizer
from .data_extractor import DataExtractor
from .visualizer import ResultVisualizer

# Import helper functions
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.helpers import (
    setup_logging, load_json_config, get_image_files, 
    create_output_directory, calculate_confidence
)

class CheckExtractor:
    """
    Main class for the check data extraction pipeline.
    """
    
    def __init__(self, config_dir=None, output_dir=None, use_transformer=False):
        """
        Initialize the check extraction pipeline.
        
        Args:
            config_dir (str, optional): Directory containing configuration files.
            output_dir (str, optional): Directory for output files.
            use_transformer (bool): Whether to use transformer-based OCR.
        """
        # Set up configuration paths
        self.config_dir = config_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
        self.regions_config_path = os.path.join(self.config_dir, 'regions_config.json')
        self.patterns_config_path = os.path.join(self.config_dir, 'extraction_patterns.json')
        
        # Create output directory
        self.output_dir = output_dir or create_output_directory()
        
        # Set up logging
        setup_logging(os.path.join(self.output_dir, 'extraction.log'))
        self.logger = logging.getLogger('check_extractor.main')
        
        # Initialize pipeline components
        self.logger.info("Initializing check extraction pipeline")
        self.preprocessor = ImagePreprocessor()
        self.region_detector = RegionDetector(config_path=self.regions_config_path)
        self.text_recognizer = TextRecognizer(use_transformer=use_transformer)
        self.data_extractor = DataExtractor(patterns_config_path=self.patterns_config_path)
        self.visualizer = ResultVisualizer(output_dir=self.output_dir)
        
    def process_check(self, image_path, preprocessing_params=None, region_method='fixed'):
        """
        Process a single check image.
        
        Args:
            image_path (str): Path to the check image.
            preprocessing_params (dict, optional): Parameters for preprocessing.
            region_method (str): Method for region extraction ('fixed' or 'dynamic').
            
        Returns:
            dict: Extraction results.
        """
        self.logger.info(f"Processing check: {image_path}")
        
        # Set default preprocessing parameters if not provided
        if preprocessing_params is None:
            preprocessing_params = {
                'deskew': True,
                'denoise_strength': 10,
                'threshold_method': 'adaptive',
                'enhance': True
            }
        
        try:
            # Load and preprocess the image
            self.logger.info("Preprocessing image")
            original, processed = self.preprocessor.preprocess(
                image_path, 
                deskew=preprocessing_params['deskew'],
                denoise_strength=preprocessing_params['denoise_strength'],
                threshold_method=preprocessing_params['threshold_method'],
                enhance=preprocessing_params['enhance']
            )
            
            # Extract regions
            self.logger.info(f"Extracting regions using {region_method} method")
            regions = self.region_detector.extract_regions(processed, method=region_method)
            
            # Extract text from regions
            self.logger.info("Performing OCR on regions")
            text_data = self.text_recognizer.extract_all_text(regions)
            
            # Extract structured data
            self.logger.info("Extracting structured data")
            micr_text = text_data.get('micr_line', '')
            payee_text = text_data.get('payee_line', '')
            amount_text = text_data.get('amount_box', '')
            date_text = text_data.get('date_line', '')
            written_amount_text = text_data.get('written_amount', '')
            
            # Extract specific data from text
            extracted_data = {
                'routing_number': self.data_extractor.extract_routing_number(micr_text),
                'account_number': self.data_extractor.extract_account_number(micr_text),
                'check_number': self.data_extractor.extract_check_number(micr_text),
                'date': self.data_extractor.extract_date(date_text),
                'amount': self.data_extractor.extract_amount(amount_text),
                'payee': self.data_extractor.extract_payee(payee_text),
                'written_amount': self.data_extractor.extract_written_amount(written_amount_text),
                'raw_text': text_data  # Store all raw OCR text
            }
            
            # Calculate confidence score
            confidence = calculate_confidence(extracted_data)
            extracted_data['confidence'] = round(confidence * 100, 2)
            
            # Create visualizations
            self.logger.info("Creating visualizations and reports")
            visualization = self.visualizer.visualize_extraction_results(
                original, regions, text_data, extracted_data
            )
            
            vis_path = self.visualizer.save_visualization(
                visualization, 
                filename=f"{os.path.basename(image_path).split('.')[0]}_results.png"
            )
            
            # Create HTML report
            report_path = self.visualizer.create_report(
                original, regions, text_data, extracted_data,
                filename=f"{os.path.basename(image_path).split('.')[0]}_report.html"
            )
            
            # Save results as JSON
            results_path = os.path.join(
                self.output_dir, 
                f"{os.path.basename(image_path).split('.')[0]}_data.json"
            )
            
            # Create a copy of extracted data without raw text for JSON
            json_data = {k: v for k, v in extracted_data.items() if k != 'raw_text'}
            json_data['image_path'] = image_path
            json_data['processed_date'] = datetime.now().isoformat()
            
            with open(results_path, 'w') as f:
                json.dump(json_data, f, indent=4)
            
            self.logger.info(f"Results saved to {self.output_dir}")
            
            return {
                'extracted_data': extracted_data,
                'visualization_path': vis_path,
                'report_path': report_path,
                'json_path': results_path
            }
            
        except Exception as e:
            self.logger.error(f"Error processing check {image_path}: {e}", exc_info=True)
            return {'error': str(e)}
    
    def batch_process(self, image_dir, preprocessing_params=None, region_method='fixed'):
        """
        Process multiple check images from a directory.
        
        Args:
            image_dir (str): Directory containing check images.
            preprocessing_params (dict, optional): Parameters for preprocessing.
            region_method (str): Method for region extraction.
            
        Returns:
            list: List of extraction results for each image.
        """
        self.logger.info(f"Batch processing checks from {image_dir}")
        
        # Get list of image files
        image_files = get_image_files(image_dir)
        
        if not image_files:
            self.logger.warning(f"No image files found in {image_dir}")
            return []
        
        # Process each image
        results = []
        for image_path in image_files:
            result = self.process_check(
                image_path, 
                preprocessing_params=preprocessing_params,
                region_method=region_method
            )
            results.append(result)
        
        # Create batch summary
        summary = {
            'total_images': len(image_files),
            'successful': sum(1 for r in results if 'error' not in r),
            'failed': sum(1 for r in results if 'error' in r),
            'average_confidence': sum(r['extracted_data'].get('confidence', 0) 
                                     for r in results if 'error' not in r) / 
                                   max(1, sum(1 for r in results if 'error' not in r))
        }
        
        summary_path = os.path.join(self.output_dir, 'batch_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        self.logger.info(f"Batch processing completed. Results saved to {self.output_dir}")
        return results

def main():
    """
    Command-line interface for the check extraction pipeline.
    """
    parser = argparse.ArgumentParser(description='Check Data Extraction Tool')
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', '-i', help='Path to single check image')
    input_group.add_argument('--batch', '-b', help='Path to directory containing check images')
    
    # Configuration arguments
    parser.add_argument('--config-dir', '-c', help='Path to configuration directory')
    parser.add_argument('--output-dir', '-o', help='Path to output directory')
    
    # Processing options
    parser.add_argument('--method', '-m', choices=['fixed', 'dynamic'], default='fixed',
                        help='Region extraction method')
    parser.add_argument('--no-deskew', action='store_true', help='Disable deskewing')
    parser.add_argument('--denoise', type=int, default=10, help='Denoising strength (0-30)')
    parser.add_argument('--threshold', choices=['adaptive', 'otsu', 'binary'], 
                        default='adaptive', help='Thresholding method')
    parser.add_argument('--no-enhance', action='store_true', help='Disable contrast enhancement')
    parser.add_argument('--use-transformer', action='store_true', 
                        help='Use transformer-based OCR (requires additional dependencies)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set preprocessing parameters
    preprocessing_params = {
        'deskew': not args.no_deskew,
        'denoise_strength': args.denoise,
        'threshold_method': args.threshold,
        'enhance': not args.no_enhance
    }
    
    # Initialize the extractor
    extractor = CheckExtractor(
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        use_transformer=args.use_transformer
    )
    
    # Process single image or batch
    if args.image:
        # Process single image
        print(f"Processing check image: {args.image}")
        result = extractor.process_check(
            args.image,
            preprocessing_params=preprocessing_params,
            region_method=args.method
        )
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Extraction completed successfully. Confidence: {result['extracted_data']['confidence']}%")
            print(f"Results saved to: {result['json_path']}")
            print(f"Visualization saved to: {result['visualization_path']}")
            print(f"Report saved to: {result['report_path']}")
    else:
        # Process batch of images
        print(f"Batch processing check images from: {args.batch}")
        results = extractor.batch_process(
            args.batch,
            preprocessing_params=preprocessing_params,
            region_method=args.method
        )
        
        # Print summary
        successful = sum(1 for r in results if 'error' not in r)
        failed = sum(1 for r in results if 'error' in r)
        
        print(f"Batch processing completed.")
        print(f"Total images processed: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
        if successful > 0:
            avg_confidence = sum(r['extracted_data'].get('confidence', 0) 
                               for r in results if 'error' not in r) / successful
            print(f"Average confidence: {avg_confidence:.2f}%")
        
        print(f"Results saved to: {extractor.output_dir}")

if __name__ == "__main__":
    main()
    """
    Command-line interface for the check extraction pipeline.
    """
    parser = argparse.ArgumentParser(description='Check Data Extraction Tool')
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', '-i', help='Path to single check image')
    input_group.add_argument('--batch', '-b', help='Path to directory containing check images')
    
    # Configuration arguments
    parser.add_argument('--config-dir', '-c', help='Path to configuration directory')
    parser.add_argument('--output-dir', '-o', help='Path to output directory')
    
    # Processing options
    parser.add_argument('--method', '-m', choices=['fixed', 'dynamic'], default='fixed',
                        help='Region extraction method')
    parser.add_argument('--no-deskew', action='store_true', help='Disable deskewing')
    parser.add_argument('--denoise', type=int, default=10, help='Denoising strength (0-30)')
    parser.add_argument('--threshold', choices=['adaptive', 'otsu', 'binary'], 
                        default='adaptive', help='Thresholding method')
    parser.add_argument('--no-enhance', action='store_true', help='Disable contrast enhancement')
    parser.add_argument('--use-transformer', action='store_true', 
                        help='Use transformer-based OCR (requires additional dependencies)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set preprocessing parameters
    preprocessing_params = {
        'deskew': not args.no_deskew,
        'denoise_strength': args.denoise,
        'threshold_method': args.threshold,
        'enhance': not args.no_enhance
    }
    
    # Initialize the extractor
    extractor = CheckExtractor(
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        use_transformer=args.use_transformer
    )
    
    #