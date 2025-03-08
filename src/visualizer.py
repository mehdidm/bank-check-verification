import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

class ResultVisualizer:
    """
    Class for visualizing check processing results and extracted data.
    """
    
    def __init__(self, output_dir="output"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir (str): Directory to save visualization results.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def draw_regions(self, image, regions_config):
        """
        Draw bounding boxes on the image for each region.
        
        Args:
            image (numpy.ndarray): Original check image.
            regions_config (dict): Dictionary with region coordinates.
            
        Returns:
            numpy.ndarray: Image with drawn regions.
        """
        # Make a copy to avoid modifying the original
        result = image.copy()
        h, w = image.shape[:2]
        
        # Colors for different regions
        colors = {
            'micr_line': (0, 0, 255),      # Red
            'amount_box': (0, 255, 0),     # Green
            'payee_line': (255, 0, 0),     # Blue
            'date_line': (255, 255, 0),    # Cyan
            'signature': (255, 0, 255),    # Magenta
            'written_amount': (0, 255, 255)  # Yellow
        }
        
        # Draw each region
        for name, coords in regions_config.items():
            y_start, y_end = int(h * coords['y1']), int(h * coords['y2'])
            x_start, x_end = int(w * coords['x1']), int(w * coords['x2'])
            
            color = colors.get(name, (125, 125, 125))
            cv2.rectangle(result, (x_start, y_start), (x_end, y_end), color, 2)
            cv2.putText(result, name, (x_start, y_start-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result

    def visualize_extraction_results(self, original_image, regions, text_data, extracted_data):
        """
        Create comprehensive visualization of the extraction results.

        Args:
            original_image (numpy.ndarray): Original check image.
            regions (dict): Dictionary of extracted region images.
            text_data (dict): Dictionary of OCR results for each region.
            extracted_data (dict): Dictionary of extracted structured data.

        Returns:
            numpy.ndarray: Visualization image.
        """
        # Convert OpenCV BGR to RGB for matplotlib
        if len(original_image.shape) == 3:
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

        # Create a figure with 3 rows and 3 columns
        fig = plt.figure(figsize=(15, 12))

        # Plot original image
        ax = fig.add_subplot(3, 3, 1)
        ax.imshow(original_rgb)
        ax.set_title("Original Check")
        ax.axis('off')

        # Plot key regions
        region_titles = {
            'micr_line': 'MICR Line',
            'amount_box': 'Amount',
            'payee_line': 'Payee',
            'date_line': 'Date',
            'signature': 'Signature',
            'written_amount': 'Written Amount'
        }

        region_positions = {
            'micr_line': 2,
            'amount_box': 3,
            'payee_line': 4,
            'date_line': 5,
            'signature': 6,
            'written_amount': 7
        }

        for name, pos in region_positions.items():
            if name in regions:
                region = regions[name]
                ax = fig.add_subplot(3, 3, pos)

                if len(region.shape) == 3:
                    region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
                    ax.imshow(region_rgb)
                else:
                    ax.imshow(region, cmap='gray')

                title = region_titles.get(name, name)
                ocr_text = text_data.get(name, "").strip()
                if len(ocr_text) > 30:
                    ocr_text = ocr_text[:27] + "..."

                ax.set_title(f"{title}\nOCR: {ocr_text}")
                ax.axis('off')

        # Plot extracted data summary
        ax = fig.add_subplot(3, 3, 8)
        ax.axis('off')

        # Display extracted data
        summary_text = "Extracted Data:\n\n"
        for key, value in extracted_data.items():
            if key != "raw_text" and value:  # Skip empty values
                summary_text += f"{key.replace('_', ' ').title()}: {value}\n"

        ax.text(0, 0.5, summary_text, fontsize=10,
                verticalalignment='center', wrap=True)

        # Add confidence score or processing summary
        ax = fig.add_subplot(3, 3, 9)
        ax.axis('off')

        # Generate a simple confidence score based on data completeness
        required_fields = ['amount', 'date', 'payee', 'routing_number', 'account_number']
        complete_fields = sum(1 for field in required_fields if extracted_data.get(field))
        confidence = (complete_fields / len(required_fields)) * 100

        confidence_text = f"Extraction Confidence: {confidence:.1f}%\n\n"
        confidence_text += "Missing Fields:\n"

        missing = [field.replace('_', ' ').title() for field in required_fields
                   if not extracted_data.get(field)]

        if missing:
            for field in missing:
                confidence_text += f"- {field}\n"
        else:
            confidence_text += "None - All required fields extracted!"

        ax.text(0, 0.5, confidence_text, fontsize=10,
                verticalalignment='center', wrap=True)

        # Adjust layout and save
        plt.tight_layout()

        # Convert figure to image
        fig.canvas.draw()
        fig_image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # Use buffer_rgba()
        fig_image = fig_image.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # Reshape to (height, width, 4)
        fig_image = fig_image[:, :, :3]  # Drop the alpha channel (RGBA -> RGB)

        plt.close(fig)

        return fig_image

    def save_visualization(self, visualization, filename="extraction_results.png"):
        """
        Save visualization to a file.
        
        Args:
            visualization (numpy.ndarray): Visualization image.
            filename (str): Output filename.
            
        Returns:
            str: Path to saved file.
        """
        output_path = os.path.join(self.output_dir, filename)
        
        # Convert from RGB to BGR for OpenCV
        if visualization.shape[2] == 3:
            visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            
        cv2.imwrite(output_path, visualization)
        return output_path
    
    def create_report(self, check_image, regions, text_data, extracted_data, filename="report.html"):
        """
        Create an HTML report of extraction results.
        
        Args:
            check_image (numpy.ndarray): Original check image.
            regions (dict): Dictionary of extracted region images.
            text_data (dict): Dictionary of OCR results.
            extracted_data (dict): Dictionary of extracted structured data.
            filename (str): Output filename.
            
        Returns:
            str: Path to saved HTML report.
        """
        # Save images to be included in the report
        image_dir = os.path.join(self.output_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        
        # Save original image
        orig_path = os.path.join(image_dir, "original.png")
        cv2.imwrite(orig_path, check_image)
        
        # Save region images
        region_paths = {}
        for name, region in regions.items():
            if region is not None and region.size > 0:
                region_path = os.path.join(image_dir, f"{name}.png")
                cv2.imwrite(region_path, region)
                region_paths[name] = os.path.relpath(region_path, self.output_dir)
        
        # Create HTML content
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Check Extraction Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { text-align: center; margin-bottom: 30px; }
                .section { margin-bottom: 30px; }
                .flex-container { display: flex; flex-wrap: wrap; gap: 20px; }
                .image-card { border: 1px solid #ddd; padding: 10px; border-radius: 5px; flex: 1; min-width: 300px; }
                .data-card { border: 1px solid #ddd; padding: 15px; border-radius: 5px; background-color: #f9f9f9; }
                img { max-width: 100%; }
                table { width: 100%; border-collapse: collapse; }
                table, th, td { border: 1px solid #ddd; }
                th, td { padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Check Extraction Report</h1>
                    <p>Generated on: TIMESTAMP</p>
                </div>
                
                <div class="section">
                    <h2>Original Check</h2>
                    <img src="ORIGINAL_IMAGE" alt="Original Check">
                </div>
                
                <div class="section">
                    <h2>Extracted Regions</h2>
                    <div class="flex-container">
                        REGION_IMAGES
                    </div>
                </div>
                
                <div class="section">
                    <h2>OCR Results</h2>
                    <table>
                        <tr>
                            <th>Region</th>
                            <th>Recognized Text</th>
                        </tr>
                        OCR_RESULTS
                    </table>
                </div>
                
                <div class="section">
                    <h2>Extracted Data</h2>
                    <div class="data-card">
                        EXTRACTED_DATA
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Replace placeholders
        import datetime
        html_content = html_content.replace("TIMESTAMP", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        html_content = html_content.replace("ORIGINAL_IMAGE", os.path.relpath(orig_path, self.output_dir))
        
        # Add region images
        region_html = ""
        for name, path in region_paths.items():
            region_html += f"""
            <div class="image-card">
                <h3>{name.replace('_', ' ').title()}</h3>
                <img src="{path}" alt="{name}">
            </div>
            """
        html_content = html_content.replace("REGION_IMAGES", region_html)
        
        # Add OCR results
        ocr_html = ""
        for name, text in text_data.items():
            ocr_html += f"""
            <tr>
                <td>{name.replace('_', ' ').title()}</td>
                <td>{text}</td>
            </tr>
            """
        html_content = html_content.replace("OCR_RESULTS", ocr_html)
        
        # Add extracted data
        extracted_html = "<dl>"
        for key, value in sorted(extracted_data.items()):
            if key != "raw_text":  # Skip raw text data
                extracted_html += f"<dt><strong>{key.replace('_', ' ').title()}</strong></dt>"
                extracted_html += f"<dd>{value}</dd>"
        extracted_html += "</dl>"
        html_content = html_content.replace("EXTRACTED_DATA", extracted_html)
        
        # Save HTML file
        report_path = os.path.join(self.output_dir, filename)
        with open(report_path, 'w') as f:
            f.write(html_content)
            
        return report_path