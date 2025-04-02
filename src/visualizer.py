import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

class ResultVisualizer:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def draw_regions(self, image, regions_config):
        result = image.copy()
        h, w = image.shape[:2]
        colors = {
            'micr_line': (0, 0, 255),
            'amount_box': (0, 255, 0),
            'payee_line': (255, 0, 0),
            'date_line': (255, 255, 0),
            'written_amount': (0, 255, 255)
        }
        for name, coords in regions_config.items():
            y_start, y_end = int(h * coords['y1']), int(h * coords['y2'])
            x_start, x_end = int(w * coords['x1']), int(w * coords['x2'])
            color = colors.get(name, (125, 125, 125))
            cv2.rectangle(result, (x_start, y_start), (x_end, y_end), color, 2)
            cv2.putText(result, name, (x_start, y_start-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return result

    def visualize_extraction_results(self, original_image, regions, text_data, extracted_data):
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) if len(original_image.shape) == 3 else cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(3, 3, 1)
        ax.imshow(original_rgb)
        ax.set_title("Original Check")
        ax.axis('off')

        region_titles = {
            'micr_line': 'MICR Line',
            'amount_box': 'Amount',
            'payee_line': 'Payee',
            'date_line': 'Date',
            'written_amount': 'Written Amount'
        }
        region_positions = {
            'micr_line': 2,
            'amount_box': 3,
            'payee_line': 4,
            'date_line': 5,
            'written_amount': 6
        }
        for name, pos in region_positions.items():
            if name in regions:
                region = regions[name]
                ax = fig.add_subplot(3, 3, pos)
                if len(region.shape) == 3:
                    ax.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
                else:
                    ax.imshow(region, cmap='gray')
                ocr_text = text_data.get(name, "").strip()[:27] + "..." if len(text_data.get(name, "")) > 30 else text_data.get(name, "")
                ax.set_title(f"{region_titles.get(name, name)}\nOCR: {ocr_text}")
                ax.axis('off')

        ax = fig.add_subplot(3, 3, 8)
        ax.axis('off')
        summary_text = "Extracted Data:\n\n"
        for key, value in extracted_data.items():
            if key != "raw_text" and value:
                summary_text += f"{key.replace('_', ' ').title()}: {value}\n"
        ax.text(0, 0.5, summary_text, fontsize=10, verticalalignment='center', wrap=True)

        ax = fig.add_subplot(3, 3, 9)
        ax.axis('off')
        confidence_text = f"Confidence: {extracted_data.get('confidence', 0):.1f}%\n\nMissing Fields:\n"
        required_fields = ['amount', 'date', 'payee', 'routing_number', 'account_number']
        missing = [field.replace('_', ' ').title() for field in required_fields if not extracted_data.get(field)]
        confidence_text += "\n".join([f"- {field}" for field in missing]) if missing else "None"
        ax.text(0, 0.5, confidence_text, fontsize=10, verticalalignment='center', wrap=True)

        plt.tight_layout()
        fig.canvas.draw()
        fig_image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        plt.close(fig)
        return fig_image

    def save_visualization(self, visualization, filename="extraction_results.png"):
        output_path = os.path.join(self.output_dir, filename)
        if visualization.shape[2] == 3:
            visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, visualization)
        return output_path

    def create_report(self, check_image, regions, text_data, extracted_data, filename="report.html"):
        image_dir = os.path.join(self.output_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        orig_path = os.path.join(image_dir, "original.png")
        cv2.imwrite(orig_path, check_image)
        region_paths = {}
        for name, region in regions.items():
            if region is not None and region.size > 0:
                region_path = os.path.join(image_dir, f"{name}.png")
                cv2.imwrite(region_path, region)
                region_paths[name] = os.path.relpath(region_path, self.output_dir)

        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Check Extraction Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { text-align: center; margin-bottom: 30px; color: #333; }
                .section { margin-bottom: 30px; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                .flex-container { display: flex; flex-wrap: wrap; gap: 20px; }
                .image-card { border: 1px solid #ddd; padding: 10px; border-radius: 5px; flex: 1; min-width: 300px; background: #fff; }
                .data-card { border: 1px solid #ddd; padding: 15px; border-radius: 5px; background-color: #f9f9f9; }
                img { max-width: 100%; border-radius: 5px; }
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
                        <tr><th>Region</th><th>Recognized Text</th></tr>
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
        import datetime
        html_content = html_content.replace("TIMESTAMP", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        html_content = html_content.replace("ORIGINAL_IMAGE", os.path.relpath(orig_path, self.output_dir))
        region_html = "".join([f'<div class="image-card"><h3>{name.replace("_", " ").title()}</h3><img src="{path}" alt="{name}"></div>' for name, path in region_paths.items()])
        html_content = html_content.replace("REGION_IMAGES", region_html)
        ocr_html = "".join([f'<tr><td>{name.replace("_", " ").title()}</td><td>{text}</td></tr>' for name, text in text_data.items()])
        html_content = html_content.replace("OCR_RESULTS", ocr_html)
        extracted_html = "<dl>" + "".join([f"<dt><strong>{key.replace('_', ' ').title()}</strong></dt><dd>{value}</dd>" for key, value in sorted(extracted_data.items()) if key != "raw_text"]) + "</dl>"
        html_content = html_content.replace("EXTRACTED_DATA", extracted_html)
        report_path = os.path.join(self.output_dir, filename)
        with open(report_path, 'w') as f:
            f.write(html_content)
        return report_path