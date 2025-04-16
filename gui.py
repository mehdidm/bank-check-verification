import tkinter as tk
from tkinter import filedialog, ttk
import json
from PIL import Image, ImageTk  # Added for image display
import subprocess  # To run main.py

class CheckProcessorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Check Processor")

        self.config_file = "config/config.json"
        self.check_types = self.load_check_types()
        self.model_types = ["dynamic", "fixed", "yolov5", "yolov8", "faster_rcnn", "efficientdet"]
        self.image_path = None  # To store the selected image path
        self.image_display = None # To store displayed image

        # Image Selection
        self.image_label = tk.Label(master, text="No image selected")
        self.image_label.pack()
        self.select_button = tk.Button(master, text="Select Image", command=self.select_image)
        self.select_button.pack()

        # Check Type Selection
        self.check_type_label = tk.Label(master, text="Select Check Type:")
        self.check_type_label.pack()
        self.check_type_var = tk.StringVar(master)
        self.check_type_var.set(self.check_types[0] if self.check_types else "default")  # Default value
        self.check_type_dropdown = ttk.Combobox(master, textvariable=self.check_type_var, values=self.check_types)
        self.check_type_dropdown.pack()

        # Model Type Selection
        self.model_type_label = tk.Label(master, text="Select Model Type:")
        self.model_type_label.pack()
        self.model_type_var = tk.StringVar(master)
        self.model_type_var.set(self.model_types[0])  # Default value
        self.model_type_dropdown = ttk.Combobox(master, textvariable=self.model_type_var, values=self.model_types)
        self.model_type_dropdown.pack()

        # Process Button
        self.process_button = tk.Button(master, text="Process Image", command=self.process_image)
        self.process_button.pack()

        # Output Text Area
        self.output_label = tk.Label(master, text="Extracted Information:")
        self.output_label.pack()
        self.output_text = tk.Text(master, height=10, width=50)
        self.output_text.pack()

    def load_check_types(self):
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                return list(config.get("check_types", {}).keys())
        except FileNotFoundError:
            print(f"Configuration file not found: {self.config_file}")
            return ["default"]
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in configuration file: {e}")
            return ["default"]

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.image_path = file_path
            self.image_label.config(text=f"Selected Image: {file_path}")
            # Display the image
            self.display_image(file_path)

    def display_image(self, image_path):
        try:
            img = Image.open(image_path)
            img.thumbnail((200, 200))  # Resize for display
            img_tk = ImageTk.PhotoImage(img)
            if self.image_display:
                self.image_display.destroy()
            self.image_display = tk.Label(self.master, image=img_tk)
            self.image_display.image = img_tk  # Keep a reference
            self.image_display.pack()
        except Exception as e:
            print(f"Error displaying image: {e}")


    def process_image(self):
        if not self.image_path:
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "Please select an image first.")
            return

        check_type = self.check_type_var.get()
        model_type = self.model_type_var.get()

        # Construct the command to run main.py
        command = [
            "python", "main.py",
            "--config", self.config_file,
            "--image", self.image_path,
            "--check_type", check_type,
            "--model_type", model_type
        ]

        try:
            # Execute the command and capture the output
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            output = result.stdout
            # Assuming main.py prints the extracted information to stdout
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, output)

        except subprocess.CalledProcessError as e:
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"Error processing image: \n{e.stderr}")
        except Exception as e:
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"An unexpected error occurred: \n{e}")


root = tk.Tk()
gui = CheckProcessorGUI(root)
root.mainloop()