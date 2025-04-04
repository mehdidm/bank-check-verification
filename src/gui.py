import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import os
from data_extractor import DataExtractor

class CheckProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Check Processor")
        self.root.geometry("1200x900")  # Even larger default window
        
        # Define color scheme
        self.colors = {
            'light': {
                'bg': '#f5f5f5',
                'fg': '#333333',
                'accent': '#2196F3',
                'success': '#4CAF50',
                'error': '#f44336',
                'frame_bg': '#ffffff'
            },
            'dark': {
                'bg': '#1e1e1e',
                'fg': '#ffffff',
                'accent': '#64B5F6',
                'success': '#81C784',
                'error': '#E57373',
                'frame_bg': '#2d2d2d'
            }
        }
        
        # Initialize dark mode state
        self.dark_mode = tk.BooleanVar(value=False)
        
        # Configure styles
        self.setup_styles()
        
        self.data_extractor = DataExtractor()
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="30")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for responsiveness
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(2, weight=1)  # Make image frame expandable
        
        # Header frame
        self.header_frame = ttk.Frame(self.main_frame)
        self.header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 30))
        self.header_frame.columnconfigure(1, weight=1)
        
        # Title with modern font
        self.title_label = ttk.Label(self.header_frame, 
                                   text="Check Image Processor", 
                                   style="Title.TLabel")
        self.title_label.grid(row=0, column=0, sticky=tk.W)
        
        # Dark mode toggle
        self.dark_mode_btn = ttk.Checkbutton(self.header_frame, 
                                           text="Dark Mode", 
                                           variable=self.dark_mode,
                                           command=self.toggle_theme,
                                           style="Switch.TCheckbutton")
        self.dark_mode_btn.grid(row=0, column=2, sticky=tk.E)
        
        # Button frame with modern styling
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=1, column=0, pady=(0, 30))
        
        # Enhanced buttons with icons (using Unicode symbols as placeholders)
        self.load_button = ttk.Button(self.button_frame, 
                                    text="📂 Load Check Image", 
                                    style="Accent.TButton",
                                    command=self.load_image)
        self.load_button.grid(row=0, column=0, padx=(0, 15))
        
        self.process_button = ttk.Button(self.button_frame, 
                                       text="⚡ Process Check",
                                       style="Accent.TButton",
                                       command=self.process_check)
        self.process_button.grid(row=0, column=1)
        self.process_button.state(['disabled'])
        
        # Image frame with shadow effect
        self.image_frame = ttk.LabelFrame(self.main_frame, 
                                        text="Check Image Preview", 
                                        padding="20",
                                        style="Card.TLabelframe")
        self.image_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 30))
        
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Results frame with modern styling
        self.results_frame = ttk.LabelFrame(self.main_frame, 
                                          text="Extracted Data", 
                                          padding="20",
                                          style="Card.TLabelframe")
        self.results_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))
        
        # Results text with custom font and colors
        self.results_text = tk.Text(self.results_frame, 
                                  height=10, 
                                  width=70,
                                  font=("Segoe UI", 11),
                                  wrap=tk.WORD,
                                  padx=10,
                                  pady=10)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Modern scrollbar
        self.scrollbar = ttk.Scrollbar(self.results_frame, 
                                     orient=tk.VERTICAL,
                                     command=self.results_text.yview)
        self.scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_text.configure(yscrollcommand=self.scrollbar.set)
        
        # Status bar with modern styling
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.main_frame, 
                                  textvariable=self.status_var,
                                  style="Status.TLabel")
        self.status_bar.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(20, 0))
        
        self.current_image_path = None
        
    def setup_styles(self):
        style = ttk.Style()
        
        # Configure modern styles
        style.configure("Title.TLabel",
                      font=("Segoe UI", 24, "bold"),
                      padding=(0, 10))
        
        style.configure("Accent.TButton",
                      font=("Segoe UI", 11),
                      padding=(20, 10))
        
        style.configure("Card.TLabelframe",
                      font=("Segoe UI", 11))
        
        style.configure("Card.TLabelframe.Label",
                      font=("Segoe UI", 12, "bold"))
        
        style.configure("Status.TLabel",
                      font=("Segoe UI", 10),
                      padding=(10, 5))
        
        # Configure hover effects
        style.map("Accent.TButton",
                background=[('active', self.colors['light']['accent'])])
    
    def toggle_theme(self):
        theme = 'dark' if self.dark_mode.get() else 'light'
        colors = self.colors[theme]
        
        # Update window colors
        self.root.configure(bg=colors['bg'])
        self.main_frame.configure(style=f"{theme}.TFrame")
        
        # Update text colors and backgrounds
        self.results_text.configure(
            bg=colors['frame_bg'],
            fg=colors['fg'],
            insertbackground=colors['fg']
        )
        
        # Update styles for dark mode
        style = ttk.Style()
        style.configure("TFrame", background=colors['bg'])
        style.configure("TLabel", background=colors['bg'], foreground=colors['fg'])
        style.configure("TButton", background=colors['accent'])
        style.configure("TLabelframe", background=colors['frame_bg'])
        style.configure("TLabelframe.Label", background=colors['frame_bg'], foreground=colors['fg'])

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if file_path:
            self.current_image_path = file_path
            image = Image.open(file_path)
            # Larger display size with maintained aspect ratio
            display_size = (800, 500)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo
            self.process_button.state(['!disabled'])
            self.status_var.set(f"✅ Loaded: {os.path.basename(file_path)}")
            
    def process_check(self):
        if not self.current_image_path:
            self.status_var.set("Error: Please load an image first")
            return
            
        try:
            self.status_var.set("Processing...")
            self.root.update()  # Update GUI to show processing status
            
            results = self.data_extractor.process_check_image(self.current_image_path)
            
            # Display results with improved formatting
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "📋 Extracted Check Data:\n\n")
            for key, value in results.items():
                formatted_key = key.replace('_', ' ').title()
                self.results_text.insert(tk.END, f"• {formatted_key}: {value}\n")
            
            self.status_var.set("Processing completed successfully")
                
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"❌ Error processing check:\n{str(e)}")

def main():
    root = tk.Tk()
    app = CheckProcessorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
