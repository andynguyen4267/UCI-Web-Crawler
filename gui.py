import tkinter as tk
from milestone2 import *
import sys

class GUI:
    def __init__(self, master):
        self.master = master
        master.title("Search Engine")

        # input
        self.input_label = tk.Label(master, text="Enter query:")
        self.input_label.pack()
        self.input_entry = tk.Entry(master)
        self.input_entry.pack()

        # submit
        self.submit_button = tk.Button(master, text="Submit", command=self.submit)
        self.submit_button.pack()

    def submit(self):
        input_text = self.input_entry.get()
        if input_text == '0':
            self.master.quit()
        else:
            project1 = Milestone2()
            paths = project1.get_paths(input_text)
            project1.get_urls(paths)
            
        self.input_entry.delete(0,tk.END)
        
        