import tkinter as tk

# Add a minimal GlobalSettings for the MRE to be self-contained
class GlobalSettings:
    class Simulation:
        SCROLL_SPEED = 0.2

class ScrollableFrame(tk.Frame):
    """A scrollable frame that works consistently across platforms"""
    def __init__(self, container: tk.Widget, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        # Create canvas and scrollbar
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)

        # Create frame inside canvas for content
        self.scrolled_frame = tk.Frame(self.canvas)

        self.scrolled_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrolled_frame, anchor="nw")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Bind canvas resize
        self.canvas.bind('<Configure>', lambda e: self.canvas.itemconfig(self.canvas.find_withtag("all")[0], width=e.width))

        # Pack scrollbar and canvas
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Bind mousewheel events
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)  # Linux
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)  # Linux

    def _on_mousewheel(self, event):
        """Cross-platform mouse wheel scrolling."""
        if event.num == 4 or event.num == 5:  # Linux
            delta = -1 if event.num == 4 else 1
        else:  # Windows/macOS
            delta = -1 * (event.delta // 120)

        # Apply scroll speed multiplier here
        delta = int(delta * GlobalSettings.Simulation.SCROLL_SPEED)
        self.canvas.yview_scroll(delta, "units") # Corrected: use self.canvas

    def add_widget(self, widget):
        """Add a widget to the scrolled frame"""
        widget.pack(in_=self.scrolled_frame)

# --- Example Usage (for testing) ---
root = tk.Tk()
root.geometry("400x300")

container = tk.Frame(root)
container.pack(fill=tk.BOTH, expand=True)
scrollable_frame = ScrollableFrame(container)
scrollable_frame.pack(fill=tk.BOTH, expand=True)

# Add some widgets to the scrollable frame
for i in range(50):
    label = tk.Label(scrollable_frame.scrolled_frame, text=f"Label {i}")
    label.pack()

    entry = tk.Entry(scrollable_frame.scrolled_frame)
    entry.insert(0, f"Entry {i}")
    entry.pack()

    button = tk.Button(scrollable_frame.scrolled_frame, text=f"Button {i}")
    button.pack()

root.mainloop()