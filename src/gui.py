import os
import sys
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import queue
import datetime
import time
from fpdf import FPDF

# Add parent directory to path to import check extractor modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.main import CheckExtractor

class CheckExtractorGUI:
    """
    GUI for the Check Extractor application.
    """
    
    def __init__(self, root):
        """
        Initialize the GUI.
        
        Args:
            root: The tkinter root window.
        """
        self.root = root
        self.root.title("Check Extractor")
        self.root.geometry("1200x800")
        
        # Set up the queue for thread-safe communication
        self.queue = queue.Queue()
        
        # Default paths
        self.config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load available check types from detection_params.json
        self.check_types = self._load_check_types()
        
        # Create GUI elements
        self._create_widgets()
        
        # Initialize the check extractor
        self.extractor = None
        self._initialize_extractor()
        
    def _load_check_types(self):
        """
        Load available check types from detection_params.json.
        
        Returns:
            list: List of available check types.
        """
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
        """
        Initialize the check extractor with current settings.
        """
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
        """
        Create the GUI widgets.
        """
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create left panel for controls
        left_panel = ttk.Frame(main_frame, padding="5", width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_panel.pack_propagate(False)
        
        # Create right panel for image display
        right_panel = ttk.Frame(main_frame, padding="5")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create bottom panel for status and output
        bottom_panel = ttk.Frame(main_frame, padding="5", height=150)
        bottom_panel.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
        bottom_panel.pack_propagate(False)
        
        # Left panel widgets
        self._create_left_panel_widgets(left_panel)
        
        # Right panel widgets
        self._create_right_panel_widgets(right_panel)
        
        # Bottom panel widgets
        self._create_bottom_panel_widgets(bottom_panel)
    
    def _create_left_panel_widgets(self, parent):
        """
        Create widgets for the left panel.
        
        Args:
            parent: The parent frame.
        """
        # Input section
        input_frame = ttk.LabelFrame(parent, text="Input", padding="5")
        input_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Single image input
        ttk.Label(input_frame, text="Check Image:").pack(anchor=tk.W)
        
        image_frame = ttk.Frame(input_frame)
        image_frame.pack(fill=tk.X, pady=5)
        
        self.image_path_var = tk.StringVar()
        ttk.Entry(image_frame, textvariable=self.image_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(image_frame, text="Browse", command=self._browse_image).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Batch processing
        ttk.Label(input_frame, text="Batch Directory (Optional):").pack(anchor=tk.W)
        
        batch_frame = ttk.Frame(input_frame)
        batch_frame.pack(fill=tk.X, pady=5)
        
        self.batch_path_var = tk.StringVar()
        ttk.Entry(batch_frame, textvariable=self.batch_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(batch_frame, text="Browse", command=self._browse_batch_dir).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Settings section
        settings_frame = ttk.LabelFrame(parent, text="Settings", padding="5")
        settings_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Method selection
        ttk.Label(settings_frame, text="Detection Method:").pack(anchor=tk.W)
        
        self.method_var = tk.StringVar(value="dynamic")
        method_frame = ttk.Frame(settings_frame)
        method_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(method_frame, text="Dynamic", variable=self.method_var, value="dynamic").pack(side=tk.LEFT)
        ttk.Radiobutton(method_frame, text="Fixed", variable=self.method_var, value="fixed").pack(side=tk.LEFT, padx=(10, 0))
        
        # Check type selection
        ttk.Label(settings_frame, text="Check Type:").pack(anchor=tk.W)
        
        self.check_type_var = tk.StringVar(value=self.check_types[0] if self.check_types else "default")
        ttk.Combobox(settings_frame, textvariable=self.check_type_var, values=self.check_types).pack(fill=tk.X, pady=5)
        
        # Preprocessing options
        preproc_frame = ttk.Frame(settings_frame)
        preproc_frame.pack(fill=tk.X, pady=5)
        
        self.deskew_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(preproc_frame, text="Deskew", variable=self.deskew_var).pack(side=tk.LEFT)
        
        self.denoise_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(preproc_frame, text="Denoise", variable=self.denoise_var).pack(side=tk.LEFT, padx=(10, 0))
        
        self.enhance_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(preproc_frame, text="Enhance", variable=self.enhance_var).pack(side=tk.LEFT, padx=(10, 0))
        
        # Threshold method
        ttk.Label(settings_frame, text="Threshold Method:").pack(anchor=tk.W)
        
        self.threshold_var = tk.StringVar(value="adaptive")
        ttk.Combobox(settings_frame, textvariable=self.threshold_var, values=["adaptive", "otsu", "none"]).pack(fill=tk.X, pady=5)
        
        # OCR options
        ocr_frame = ttk.Frame(settings_frame)
        ocr_frame.pack(fill=tk.X, pady=5)
        
        self.transformer_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ocr_frame, text="Use Transformer OCR", variable=self.transformer_var).pack(anchor=tk.W)
        
        # Action buttons
        action_frame = ttk.Frame(parent)
        action_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(action_frame, text="Process Check", command=self._process_check).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(action_frame, text="Process Batch", command=self._process_batch).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(action_frame, text="Update Settings", command=self._update_settings).pack(fill=tk.X)
    
    def _create_right_panel_widgets(self, parent):
        """
        Create widgets for the right panel.
        
        Args:
            parent: The parent frame.
        """
        # Image display
        image_frame = ttk.LabelFrame(parent, text="Check Image", padding="5")
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Canvas for image display with scrollbars
        canvas_frame = ttk.Frame(image_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.image_canvas = tk.Canvas(canvas_frame, bg="white")
        self.image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.image_canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        h_scrollbar = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=self.image_canvas.xview)
        h_scrollbar.pack(fill=tk.X)
        
        self.image_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Bind canvas resize event
        self.image_canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Placeholder for image
        self.image_canvas.create_text(
            150, 150, text="No image loaded", font=("Arial", 14), fill="gray"
        )
        
        # Store the image reference to prevent garbage collection
        self.image_ref = None
    
    def _create_bottom_panel_widgets(self, parent):
        """
        Create widgets for the bottom panel.
        
        Args:
            parent: The parent frame.
        """
        # Status bar
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=(5, 0))
        
        # Progress bar (hidden by default)
        self.progress_frame = ttk.Frame(parent)
        self.progress_frame.pack(fill=tk.X, pady=(0, 5))
        self.progress_frame.pack_forget()  # Hide initially
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient=tk.HORIZONTAL, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, padx=5)
        
        self.progress_label = ttk.Label(self.progress_frame, text="Processing...")
        self.progress_label.pack(pady=(2, 0))
        
        # Output text
        output_frame = ttk.LabelFrame(parent, text="Extracted Data", padding="5")
        output_frame.pack(fill=tk.BOTH, expand=True)
        
        self.output_text = tk.Text(output_frame, wrap=tk.WORD, height=5)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        output_scrollbar = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=self.output_text.yview)
        output_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.output_text.configure(yscrollcommand=output_scrollbar.set)
        
        # Create overlay for loading indicator
        self.overlay = None
        self.overlay_text = None
        self.loading_dots_count = 0
        self.loading_dots_timer = None
    
    def _browse_image(self):
        """
        Open a file dialog to select a check image.
        """
        file_path = filedialog.askopenfilename(
            title="Select Check Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        
        if file_path:
            self.image_path_var.set(file_path)
            self._load_image(file_path)
    
    def _browse_batch_dir(self):
        """
        Open a directory dialog to select a batch directory.
        """
        dir_path = filedialog.askdirectory(title="Select Batch Directory")
        
        if dir_path:
            self.batch_path_var.set(dir_path)
    
    def _load_image(self, image_path):
        """
        Load and display an image in the canvas.
        
        Args:
            image_path: Path to the image file.
        """
        try:
            # Load image with PIL
            pil_image = Image.open(image_path)
            
            # Resize image to fit canvas while maintaining aspect ratio
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            # Calculate scaling factor
            width_ratio = canvas_width / pil_image.width
            height_ratio = canvas_height / pil_image.height
            scale_factor = min(width_ratio, height_ratio, 1.0)  # Don't upscale
            
            new_width = int(pil_image.width * scale_factor)
            new_height = int(pil_image.height * scale_factor)
            
            # Resize image
            if scale_factor < 1.0:
                pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to Tkinter PhotoImage
            self.image_ref = ImageTk.PhotoImage(pil_image)
            
            # Clear canvas and display image
            self.image_canvas.delete("all")
            self.image_canvas.create_image(
                canvas_width // 2, canvas_height // 2,
                image=self.image_ref, anchor=tk.CENTER
            )
            
            # Configure canvas scrolling region
            self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
            
            self.status_var.set(f"Loaded image: {os.path.basename(image_path)}")
        except Exception as e:
            self.status_var.set(f"Error loading image: {e}")
            messagebox.showerror("Image Error", f"Failed to load image: {e}")
    
    def _on_canvas_configure(self, event):
        """
        Handle canvas resize event.
        
        Args:
            event: The configure event.
        """
        # If an image is loaded, redisplay it to fit the new canvas size
        if self.image_path_var.get():
            self._load_image(self.image_path_var.get())
    
    def _update_settings(self):
        """
        Update the check extractor settings.
        """
        try:
            self._initialize_extractor()
            self.status_var.set("Settings updated successfully.")
        except Exception as e:
            self.status_var.set(f"Error updating settings: {e}")
            messagebox.showerror("Settings Error", f"Failed to update settings: {e}")
    
    def _process_check(self):
        """
        Process a single check image.
        """
        image_path = self.image_path_var.get()
        
        if not image_path:
            messagebox.showwarning("Input Error", "Please select a check image.")
            return
        
        if not os.path.exists(image_path):
            messagebox.showwarning("Input Error", "The selected image file does not exist.")
            return
        
        # Update status and show loading indicator
        self.status_var.set("Processing check...")
        self._show_loading_indicator("Processing check")
        self.root.update_idletasks()
        
        # Start processing in a separate thread
        threading.Thread(target=self._process_check_thread, args=(image_path,), daemon=True).start()
    
    def _process_check_thread(self, image_path):
        """
        Thread function for processing a check.
        
        Args:
            image_path: Path to the check image.
        """
        try:
            # Prepare preprocessing parameters
            preprocessing_params = {
                'deskew': self.deskew_var.get(),
                'denoise_strength': 10 if self.denoise_var.get() else 0,
                'threshold_method': self.threshold_var.get(),
                'enhance': self.enhance_var.get()
            }
            
            # Process the check
            result = self.extractor.process_check(
                image_path,
                preprocessing_params=preprocessing_params,
                region_method=self.method_var.get()
            )
            
            # Update GUI with result
            self.queue.put(("success", result))
        except Exception as e:
            self.queue.put(("error", str(e)))
        
        # Schedule GUI update
        self.root.after(100, self._check_queue)
    
    def _process_batch(self):
        """
        Process a batch of check images.
        """
        batch_dir = self.batch_path_var.get()
        
        if not batch_dir:
            messagebox.showwarning("Input Error", "Please select a batch directory.")
            return
        
        if not os.path.isdir(batch_dir):
            messagebox.showwarning("Input Error", "The selected batch directory does not exist.")
            return
        
        # Update status and show loading indicator
        self.status_var.set("Processing batch...")
        self._show_loading_indicator("Processing batch")
        self.root.update_idletasks()
        
        # Start processing in a separate thread
        threading.Thread(target=self._process_batch_thread, args=(batch_dir,), daemon=True).start()
    
    def _process_batch_thread(self, batch_dir):
        """
        Thread function for processing a batch of checks.
        
        Args:
            batch_dir: Path to the batch directory.
        """
        try:
            # Get image files in the directory
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                image_files.extend([os.path.join(batch_dir, f) for f in os.listdir(batch_dir) if f.lower().endswith(ext)])
            
            if not image_files:
                self.queue.put(("error", "No image files found in the batch directory."))
                return
            
            # Prepare preprocessing parameters
            preprocessing_params = {
                'deskew': self.deskew_var.get(),
                'denoise_strength': 10 if self.denoise_var.get() else 0,
                'threshold_method': self.threshold_var.get(),
                'enhance': self.enhance_var.get()
            }
            
            # Process each image
            results = []
            for i, image_path in enumerate(image_files):
                progress_msg = f"Processing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}"
                self.queue.put(("status", progress_msg))
                
                # Update progress label
                self.queue.put(("update_progress", progress_msg))
                
                try:
                    result = self.extractor.process_check(
                        image_path,
                        preprocessing_params=preprocessing_params,
                        region_method=self.method_var.get()
                    )
                    results.append((image_path, result))
                except Exception as e:
                    results.append((image_path, f"Error: {e}"))
            
            # Update GUI with results
            self.queue.put(("batch_success", results))
        except Exception as e:
            self.queue.put(("error", str(e)))
        
        # Schedule GUI update
        self.root.after(100, self._check_queue)
    
    def _check_queue(self):
        """
        Check the queue for messages from worker threads.
        """
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
                elif message_type == "update_progress":
                    self.progress_label.config(text=data)
                    # Also update the overlay text if it exists
                    if self.overlay_text:
                        base_message = data.split('...')[0] if '...' in data else data
                        self._animate_loading_dots(base_message)
                
                self.queue.task_done()
        except queue.Empty:
            # No more messages, schedule next check
            self.root.after(100, self._check_queue)
    
    def _handle_success(self, result):
        """
        Handle successful check processing.
        
        Args:
            result: The processing result.
        """
        # Hide loading indicator
        self._hide_loading_indicator()
        
        self.status_var.set("Check processed successfully.")
        
        # Display extracted data
        self.output_text.delete(1.0, tk.END)
        
        if 'extracted_data' in result:
            data = result['extracted_data']
            self.output_text.insert(tk.END, "Extracted Data:\n\n")
            
            for key, value in data.items():
                self.output_text.insert(tk.END, f"{key}: {value}\n")
                
            # Generate PDF report
            pdf_path = self._generate_pdf_report(result)
            if pdf_path:
                self.output_text.insert(tk.END, f"\nPDF Report: {pdf_path}\n")
                self.status_var.set(f"Check processed successfully. PDF report generated at {os.path.basename(pdf_path)}")
        else:
            self.output_text.insert(tk.END, "No data extracted.")
        
        # Load the visualization if available
        if 'visualization_path' in result and os.path.exists(result['visualization_path']):
            self._load_image(result['visualization_path'])
    
    def _handle_batch_success(self, results):
        """
        Handle successful batch processing.
        
        Args:
            results: List of (image_path, result) tuples.
        """
        # Hide loading indicator
        self._hide_loading_indicator()
        
        self.status_var.set(f"Batch processing completed: {len(results)} images processed.")
        
        # Display summary
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, f"Processed {len(results)} images:\n\n")
        
        pdf_reports = []
        for image_path, result in results:
            filename = os.path.basename(image_path)
            
            if isinstance(result, str):  # Error message
                self.output_text.insert(tk.END, f"{filename}: {result}\n")
            else:
                # Generate PDF report for successful processing
                pdf_path = self._generate_pdf_report(result, batch_mode=True)
                if pdf_path:
                    pdf_reports.append(pdf_path)
                self.output_text.insert(tk.END, f"{filename}: Processed successfully\n")
        
        # Generate batch summary PDF
        if pdf_reports:
            batch_pdf_path = self._generate_batch_summary_pdf(results, pdf_reports)
            self.output_text.insert(tk.END, f"\nBatch PDF summary: {batch_pdf_path}\n")
        
        self.output_text.insert(tk.END, f"\nResults saved to: {self.output_dir}")
        
        # Show message box
        messagebox.showinfo("Batch Processing", f"Processed {len(results)} images. Results saved to {self.output_dir}")
    
    def _handle_error(self, error_message):
        """
        Handle processing error.
        
        Args:
            error_message: The error message.
        """
        # Hide loading indicator
        self._hide_loading_indicator()
        
        self.status_var.set(f"Error: {error_message}")
        messagebox.showerror("Processing Error", error_message)
        
    def _show_loading_indicator(self, message="Processing"):
        """
        Show a loading indicator overlay on the image canvas and progress bar.
        
        Args:
            message: The message to display during loading.
        """
        # Show progress bar
        self.progress_frame.pack(fill=tk.X, pady=(0, 5), after=self.status_var.master.master)
        self.progress_bar.start(10)  # Start the progress bar animation
        self.progress_label.config(text=f"{message}...")
        
        # Create an overlay on the image canvas if an image is loaded
        if self.image_ref:
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            # Create semi-transparent overlay
            self.overlay = self.image_canvas.create_rectangle(
                0, 0, canvas_width, canvas_height,
                fill="gray", stipple="gray50", tags="overlay"
            )
            
            # Create loading text
            self.overlay_text = self.image_canvas.create_text(
                canvas_width // 2, canvas_height // 2,
                text=f"{message}...", font=("Arial", 16, "bold"),
                fill="white", tags="overlay"
            )
            
            # Start animation for loading dots
            self._animate_loading_dots(message)
    
    def _animate_loading_dots(self, message):
        """
        Animate the loading dots in the overlay text.
        
        Args:
            message: The base message to display.
        """
        if self.overlay_text:
            self.loading_dots_count = (self.loading_dots_count + 1) % 4
            dots = "." * self.loading_dots_count
            self.image_canvas.itemconfig(self.overlay_text, text=f"{message}{dots}")
            
            # Schedule next animation frame
            self.loading_dots_timer = self.root.after(500, lambda: self._animate_loading_dots(message))
    
    def _hide_loading_indicator(self):
        """
        Hide the loading indicator overlay and progress bar.
        """
        # Hide progress bar
        self.progress_bar.stop()  # Stop the progress bar animation
        self.progress_frame.pack_forget()
        
        # Remove overlay from canvas
        if self.overlay:
            self.image_canvas.delete("overlay")
            self.overlay = None
            self.overlay_text = None
        
        # Cancel loading dots animation timer
        if self.loading_dots_timer:
            self.root.after_cancel(self.loading_dots_timer)
            self.loading_dots_timer = None

    def _generate_pdf_report(self, result, batch_mode=False):
        """
        Generate a PDF report for the processed check.
        
        Args:
            result: The processing result.
            batch_mode: Whether this is being called as part of batch processing.
            
        Returns:
            str: Path to the generated PDF file, or None if generation failed.
        """
        try:
            if 'extracted_data' not in result or 'original_image_path' not in result:
                return None
            
            # Create PDF object
            pdf = FPDF()
            pdf.add_page()
            
            # Set up fonts
            pdf.set_font("Arial", "B", 16)
            
            # Title
            check_filename = os.path.basename(result['original_image_path'])
            pdf.cell(0, 10, f"Check Extraction Report: {check_filename}", ln=True, align='C')
            pdf.ln(5)
            
            # Date and time
            pdf.set_font("Arial", "I", 10)
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pdf.cell(0, 10, f"Generated on: {current_time}", ln=True)
            pdf.ln(5)
            
            # Check image
            if 'visualization_path' in result and os.path.exists(result['visualization_path']):
                image_path = result['visualization_path']
            elif os.path.exists(result['original_image_path']):
                image_path = result['original_image_path']
            else:
                image_path = None
                
            if image_path:
                # Add image to PDF (with proper scaling)
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "Check Image:", ln=True)
                
                # Get image dimensions
                img = Image.open(image_path)
                img_width, img_height = img.size
                
                # Calculate aspect ratio
                aspect = img_height / img_width
                
                # Set image width to 160mm (A4 width is 210mm, leaving margins)
                pdf_img_width = 160
                pdf_img_height = pdf_img_width * aspect
                
                # Add image centered
                pdf.image(image_path, x=(210-pdf_img_width)/2, y=pdf.get_y(), w=pdf_img_width)
                pdf.ln(pdf_img_height + 10)
            
            # Extracted data
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Extracted Data:", ln=True)
            pdf.ln(2)
            
            # Data table
            pdf.set_font("Arial", "", 10)
            data = result['extracted_data']
            
            # Define table structure
            col_width = 95
            row_height = 8
            
            # Add data rows
            for key, value in data.items():
                pdf.set_font("Arial", "B", 10)
                pdf.cell(col_width/2, row_height, key, border=1)
                pdf.set_font("Arial", "", 10)
                pdf.cell(col_width, row_height, str(value), border=1, ln=True)
            
            # Confidence scores if available
            if 'confidence_scores' in result:
                pdf.ln(5)
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "Confidence Scores:", ln=True)
                pdf.ln(2)
                
                pdf.set_font("Arial", "", 10)
                for field, score in result['confidence_scores'].items():
                    pdf.set_font("Arial", "B", 10)
                    pdf.cell(col_width/2, row_height, field, border=1)
                    pdf.set_font("Arial", "", 10)
                    pdf.cell(col_width/2, row_height, f"{score:.2f}%", border=1, ln=True)
            
            # Save the PDF
            base_filename = os.path.splitext(os.path.basename(result['original_image_path']))[0]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create reports directory if it doesn't exist
            reports_dir = os.path.join(self.output_dir, "reports")
            os.makedirs(reports_dir, exist_ok=True)
            
            pdf_filename = f"{base_filename}_{timestamp}.pdf"
            pdf_path = os.path.join(reports_dir, pdf_filename)
            
            pdf.output(pdf_path)
            return pdf_path
        except Exception as e:
            print(f"Error generating PDF report: {e}")
            if not batch_mode:  # Only show error in single check mode
                messagebox.showerror("PDF Generation Error", f"Failed to generate PDF report: {e}")
            return None
    
    def _generate_batch_summary_pdf(self, results, pdf_reports):
        """
        Generate a summary PDF for batch processing.
        
        Args:
            results: List of (image_path, result) tuples.
            pdf_reports: List of paths to individual PDF reports.
            
        Returns:
            str: Path to the generated summary PDF file.
        """
        try:
            # Create PDF object
            pdf = FPDF()
            pdf.add_page()
            
            # Set up fonts
            pdf.set_font("Arial", "B", 16)
            
            # Title
            pdf.cell(0, 10, "Batch Check Extraction Summary", ln=True, align='C')
            pdf.ln(5)
            
            # Date and time
            pdf.set_font("Arial", "I", 10)
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pdf.cell(0, 10, f"Generated on: {current_time}", ln=True)
            pdf.cell(0, 10, f"Total checks processed: {len(results)}", ln=True)
            pdf.ln(10)
            
            # Summary table
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Processing Results:", ln=True)
            pdf.ln(2)
            
            # Table headers
            pdf.set_font("Arial", "B", 10)
            pdf.cell(80, 8, "Check Image", border=1)
            pdf.cell(60, 8, "Status", border=1)
            pdf.cell(40, 8, "Report", border=1, ln=True)
            
            # Table rows
            pdf.set_font("Arial", "", 10)
            for i, (image_path, result) in enumerate(results):
                filename = os.path.basename(image_path)
                
                if isinstance(result, str):  # Error message
                    status = "Error"
                    report = "N/A"
                else:
                    status = "Success"
                    if i < len(pdf_reports):
                        report = os.path.basename(pdf_reports[i])
                    else:
                        report = "N/A"
                
                pdf.cell(80, 8, filename, border=1)
                pdf.cell(60, 8, status, border=1)
                pdf.cell(40, 8, report, border=1, ln=True)
            
            # Save the PDF
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create reports directory if it doesn't exist
            reports_dir = os.path.join(self.output_dir, "reports")
            os.makedirs(reports_dir, exist_ok=True)
            
            pdf_filename = f"batch_summary_{timestamp}.pdf"
            pdf_path = os.path.join(reports_dir, pdf_filename)
            
            pdf.output(pdf_path)
            return pdf_path
        except Exception as e:
            print(f"Error generating batch summary PDF: {e}")
            messagebox.showerror("PDF Generation Error", f"Failed to generate batch summary PDF: {e}")
            return None

def main():
    """
    Main function to run the GUI application.
    """
    root = tk.Tk()
    app = CheckExtractorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
