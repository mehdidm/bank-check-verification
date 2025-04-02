import os
import sys
import json
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import queue
import datetime
from fpdf import FPDF

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.main import CheckExtractor

class CheckExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Check Extractor")
        self.root.geometry("1200x800")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.queue = queue.Queue()
        self.config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
        os.makedirs(self.output_dir, exist_ok=True)

        self.check_types = self._load_check_types()
        self._create_widgets()
        self.extractor = None
        self._initialize_extractor()

    def _load_check_types(self):
        detection_params_path = os.path.join(self.config_dir, "detection_params.json")
        if os.path.exists(detection_params_path):
            try:
                with open(detection_params_path, 'r') as f:
                    params = json.load(f)
                return list(params.keys())
            except Exception as e:
                print(f"Error loading check types: {e}")
                return ["default"]
        return ["default"]

    def _initialize_extractor(self):
        try:
            self.extractor = CheckExtractor(
                config_dir=self.config_dir,
                output_dir=self.output_dir,
                detection_params_path=os.path.join(self.config_dir, "detection_params.json"),
                check_type=self.check_type_var.get()
            )
            self.status_var.set("Extractor initialized successfully.")
        except Exception as e:
            self.status_var.set(f"Error initializing extractor: {e}")
            messagebox.showerror("Initialization Error", f"Failed to initialize check extractor: {e}")

    def _create_widgets(self):
        main_frame = ctk.CTkFrame(self.root, corner_radius=10)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        left_panel = ctk.CTkFrame(main_frame, width=300, corner_radius=10)
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        left_panel.pack_propagate(False)

        right_panel = ctk.CTkFrame(main_frame, corner_radius=10)
        right_panel.pack(side="right", fill="both", expand=True)

        self._create_left_panel_widgets(left_panel)
        self._create_right_panel_widgets(right_panel)

    def _create_left_panel_widgets(self, parent):
        # Input Section
        input_frame = ctk.CTkFrame(parent, corner_radius=5)
        input_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(input_frame, text="Check Image:").pack(anchor="w", padx=5, pady=2)
        self.image_path_var = ctk.StringVar()
        image_entry = ctk.CTkEntry(input_frame, textvariable=self.image_path_var)
        image_entry.pack(fill="x", padx=5, pady=2)
        ctk.CTkButton(input_frame, text="Browse", command=self._browse_image).pack(pady=5)

        ctk.CTkLabel(input_frame, text="Batch Directory (Optional):").pack(anchor="w", padx=5, pady=2)
        self.batch_path_var = ctk.StringVar()
        batch_entry = ctk.CTkEntry(input_frame, textvariable=self.batch_path_var)
        batch_entry.pack(fill="x", padx=5, pady=2)
        ctk.CTkButton(input_frame, text="Browse", command=self._browse_batch_dir).pack(pady=5)

        # Settings Section
        settings_frame = ctk.CTkFrame(parent, corner_radius=5)
        settings_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(settings_frame, text="Detection Method:").pack(anchor="w", padx=5, pady=2)
        self.method_var = ctk.StringVar(value="dynamic")
        ctk.CTkRadioButton(settings_frame, text="Dynamic", variable=self.method_var, value="dynamic").pack(side="left", padx=5)
        ctk.CTkRadioButton(settings_frame, text="Fixed", variable=self.method_var, value="fixed").pack(side="left", padx=5)

        ctk.CTkLabel(settings_frame, text="Check Type:").pack(anchor="w", padx=5, pady=2)
        self.check_type_var = ctk.StringVar(value=self.check_types[0] if self.check_types else "default")
        ctk.CTkOptionMenu(settings_frame, variable=self.check_type_var, values=self.check_types).pack(fill="x", padx=5, pady=2)

        self.deskew_var = ctk.BooleanVar(value=True)
        self.denoise_var = ctk.BooleanVar(value=True)
        self.enhance_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(settings_frame, text="Deskew", variable=self.deskew_var).pack(side="left", padx=5)
        ctk.CTkCheckBox(settings_frame, text="Denoise", variable=self.denoise_var).pack(side="left", padx=5)
        ctk.CTkCheckBox(settings_frame, text="Enhance", variable=self.enhance_var).pack(side="left", padx=5)

        ctk.CTkLabel(settings_frame, text="Threshold Method:").pack(anchor="w", padx=5, pady=2)
        self.threshold_var = ctk.StringVar(value="adaptive")
        ctk.CTkOptionMenu(settings_frame, variable=self.threshold_var, values=["adaptive", "otsu", "none"]).pack(fill="x", padx=5, pady=2)

        self.transformer_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(settings_frame, text="Use Transformer OCR", variable=self.transformer_var).pack(anchor="w", padx=5, pady=2)

        # Action Buttons
        ctk.CTkButton(parent, text="Process Check", command=self._process_check, fg_color="#1f77b4").pack(fill="x", pady=5)
        ctk.CTkButton(parent, text="Process Batch", command=self._process_batch, fg_color="#ff7f0e").pack(fill="x", pady=5)
        ctk.CTkButton(parent, text="Update Settings", command=self._update_settings, fg_color="#2ca02c").pack(fill="x", pady=5)

        self.status_var = ctk.StringVar(value="Ready")
        ctk.CTkLabel(parent, textvariable=self.status_var).pack(pady=10)

    def _create_right_panel_widgets(self, parent):
        image_frame = ctk.CTkFrame(parent, corner_radius=5)
        image_frame.pack(fill="both", expand=True, pady=(0, 10))

        self.image_canvas = ctk.CTkCanvas(image_frame, bg="#2b2b2b")
        self.image_canvas.pack(fill="both", expand=True)

        self.output_text = ctk.CTkTextbox(parent, height=150, corner_radius=5)
        self.output_text.pack(fill="x", pady=10)
        self.image_ref = None

    def _browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if file_path:
            self.image_path_var.set(file_path)
            self._load_image(file_path)

    def _browse_batch_dir(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.batch_path_var.set(dir_path)

    def _load_image(self, image_path):
        try:
            pil_image = Image.open(image_path)
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            scale_factor = min(canvas_width / pil_image.width, canvas_height / pil_image.height, 1.0)
            new_width = int(pil_image.width * scale_factor)
            new_height = int(pil_image.height * scale_factor)
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
            self.image_ref = ImageTk.PhotoImage(pil_image)
            self.image_canvas.delete("all")
            self.image_canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.image_ref, anchor="center")
            self.status_var.set(f"Loaded image: {os.path.basename(image_path)}")
        except Exception as e:
            self.status_var.set(f"Error loading image: {e}")
            messagebox.showerror("Image Error", f"Failed to load image: {e}")

    def _update_settings(self):
        self._initialize_extractor()
        self.status_var.set("Settings updated successfully.")

    def _process_check(self):
        image_path = self.image_path_var.get()
        if not image_path or not os.path.exists(image_path):
            messagebox.showwarning("Input Error", "Please select a valid check image.")
            return
        self.status_var.set("Processing check...")
        threading.Thread(target=self._process_check_thread, args=(image_path,), daemon=True).start()

    def _process_check_thread(self, image_path):
        try:
            preprocessing_params = {
                'deskew': self.deskew_var.get(),
                'denoise_strength': 10 if self.denoise_var.get() else 0,
                'threshold_method': self.threshold_var.get(),
                'enhance': self.enhance_var.get()
            }
            result = self.extractor.process_check(
                image_path,
                preprocessing_params=preprocessing_params,
                region_method=self.method_var.get()
            )
            self.queue.put(("success", result))
        except Exception as e:
            self.queue.put(("error", str(e)))
        self.root.after(100, self._check_queue)

    def _process_batch(self):
        batch_dir = self.batch_path_var.get()
        if not batch_dir or not os.path.isdir(batch_dir):
            messagebox.showwarning("Input Error", "Please select a valid batch directory.")
            return
        self.status_var.set("Processing batch...")
        threading.Thread(target=self._process_batch_thread, args=(batch_dir,), daemon=True).start()

    def _process_batch_thread(self, batch_dir):
        try:
            image_files = [os.path.join(batch_dir, f) for f in os.listdir(batch_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
            if not image_files:
                self.queue.put(("error", "No image files found in the batch directory."))
                return
            preprocessing_params = {
                'deskew': self.deskew_var.get(),
                'denoise_strength': 10 if self.denoise_var.get() else 0,
                'threshold_method': self.threshold_var.get(),
                'enhance': self.enhance_var.get()
            }
            results = []
            for i, image_path in enumerate(image_files):
                self.queue.put(("status", f"Processing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}"))
                result = self.extractor.process_check(image_path, preprocessing_params, self.method_var.get())
                results.append((image_path, result))
            self.queue.put(("batch_success", results))
        except Exception as e:
            self.queue.put(("error", str(e)))
        self.root.after(100, self._check_queue)

    def _check_queue(self):
        try:
            while True:
                message_type, data = self.queue.get_nowait()
                if message_type == "success":
                    self._handle_success(data)
                elif message_type == "batch_success":
                    self._handle_batch_success(data)
                elif message_type == "error":
                    self._handle_error(data)
                elif message_type == "status":
                    self.status_var.set(data)
                self.queue.task_done()
        except queue.Empty:
            self.root.after(100, self._check_queue)

    def _handle_success(self, result):
        self.status_var.set("Check processed successfully.")
        self.output_text.delete("0.0", "end")
        if 'extracted_data' in result:
            data = result['extracted_data']
            self.output_text.insert("end", "Extracted Data:\n\n")
            for key, value in data.items():
                self.output_text.insert("end", f"{key}: {value}\n")
            pdf_path = self._generate_pdf_report(result)
            if pdf_path:
                self.output_text.insert("end", f"\nPDF Report: {pdf_path}\n")
        if 'visualization_path' in result and os.path.exists(result['visualization_path']):
            self._load_image(result['visualization_path'])

    def _handle_batch_success(self, results):
        self.status_var.set(f"Batch processing completed: {len(results)} images processed.")
        self.output_text.delete("0.0", "end")
        self.output_text.insert("end", f"Processed {len(results)} images:\n\n")
        pdf_reports = []
        for image_path, result in results:
            filename = os.path.basename(image_path)
            if isinstance(result, str):
                self.output_text.insert("end", f"{filename}: {result}\n")
            else:
                pdf_path = self._generate_pdf_report(result, batch_mode=True)
                if pdf_path:
                    pdf_reports.append(pdf_path)
                self.output_text.insert("end", f"{filename}: Processed successfully\n")
        if pdf_reports:
            batch_pdf_path = self._generate_batch_summary_pdf(results, pdf_reports)
            self.output_text.insert("end", f"\nBatch PDF summary: {batch_pdf_path}\n")

    def _handle_error(self, error_message):
        self.status_var.set(f"Error: {error_message}")
        messagebox.showerror("Processing Error", error_message)

    def _generate_pdf_report(self, result, batch_mode=False):
        try:
            if 'extracted_data' not in result or 'original_image_path' not in result:
                return None
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            check_filename = os.path.basename(result['original_image_path'])
            pdf.cell(0, 10, f"Check Extraction Report: {check_filename}", ln=True, align='C')
            pdf.ln(5)
            pdf.set_font("Arial", "I", 10)
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pdf.cell(0, 10, f"Generated on: {current_time}", ln=True)
            pdf.ln(5)
            image_path = result.get('visualization_path', result['original_image_path'])
            if image_path and os.path.exists(image_path):
                img = Image.open(image_path)
                pdf_img_width = 160
                pdf_img_height = pdf_img_width * (img.height / img.width)
                pdf.image(image_path, x=(210-pdf_img_width)/2, y=pdf.get_y(), w=pdf_img_width)
                pdf.ln(pdf_img_height + 10)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Extracted Data:", ln=True)
            pdf.ln(2)
            pdf.set_font("Arial", "", 10)
            for key, value in result['extracted_data /'].items():
                pdf.cell(95, 8, key, border=1)
                pdf.cell(95, 8, str(value), border=1, ln=True)
            reports_dir = os.path.join(self.output_dir, "reports")
            os.makedirs(reports_dir, exist_ok=True)
            pdf_filename = f"{os.path.splitext(check_filename)[0]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            pdf_path = os.path.join(reports_dir, pdf_filename)
            pdf.output(pdf_path)
            return pdf_path
        except Exception as e:
            if not batch_mode:
                messagebox.showerror("PDF Generation Error", f"Failed to generate PDF report: {e}")
            return None

    def _generate_batch_summary_pdf(self, results, pdf_reports):
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Batch Check Extraction Summary", ln=True, align='C')
            pdf.ln(5)
            pdf.set_font("Arial", "I", 10)
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pdf.cell(0, 10, f"Generated on: {current_time}", ln=True)
            pdf.cell(0, 10, f"Total checks processed: {len(results)}", ln=True)
            pdf.ln(10)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Processing Results:", ln=True)
            pdf.ln(2)
            pdf.set_font("Arial", "B", 10)
            pdf.cell(80, 8, "Check Image", border=1)
            pdf.cell(60, 8, "Status", border=1)
            pdf.cell(40, 8, "Report", border=1, ln=True)
            pdf.set_font("Arial", "", 10)
            for i, (image_path, result) in enumerate(results):
                filename = os.path.basename(image_path)
                status = "Error" if isinstance(result, str) else "Success"
                report = os.path.basename(pdf_reports[i]) if i < len(pdf_reports) and status == "Success" else "N/A"
                pdf.cell(80, 8, filename, border=1)
                pdf.cell(60, 8, status, border=1)
                pdf.cell(40, 8, report, border=1, ln=True)
            reports_dir = os.path.join(self.output_dir, "reports")
            os.makedirs(reports_dir, exist_ok=True)
            pdf_filename = f"batch_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            pdf_path = os.path.join(reports_dir, pdf_filename)
            pdf.output(pdf_path)
            return pdf_path
        except Exception as e:
            messagebox.showerror("PDF Generation Error", f"Failed to generate batch summary PDF: {e}")
            return None
if __name__ == "__main__":
    root = ctk.CTk()
    app = CheckExtractorGUI(root)
    root.mainloop()