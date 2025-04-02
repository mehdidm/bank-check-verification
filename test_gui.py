import customtkinter as ctk
root = ctk.CTk()
root.title("Test GUI")
root.geometry("400x300")
label = ctk.CTkLabel(root, text="Hello, GUI!")
label.pack(pady=20)
root.mainloop()
