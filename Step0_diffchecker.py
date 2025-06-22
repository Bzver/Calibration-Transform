import os

import tkinter as tk
from tkinter import filedialog

import cv2
import matplotlib.pyplot as plt

def get_first_frame(video_path):
    """Extracts the first frame from a video file."""
    abs_video_path = os.path.abspath(video_path) # Ensures path is absolute
    if not os.path.exists(abs_video_path):
        print(f"Error: Video file not found at {abs_video_path}")
        return None

    cap = cv2.VideoCapture(abs_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {abs_video_path}")
        cap.release()
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read the first frame from {abs_video_path}")
        return None
    return frame

def compare_and_show_first_frames(video1_path, video2_path):
    print(f"Loading first frame from: {video1_path}")
    frame1 = get_first_frame(video1_path)
    print(f"Loading first frame from: {video2_path}")
    frame2 = get_first_frame(video2_path)

    if frame1 is None or frame2 is None:
        print("Could not proceed with comparison due to errors in loading one or both frames.")
        return

    # Resize frame2 to match frame1's dimensions for proper comparison
    h1, w1 = frame1.shape[:2]
    print(f"Resizing second frame to match first frame dimensions: {w1}x{h1}")
    frame2_resized = cv2.resize(frame2, (w1, h1))

    # Convert frames to grayscale for comparison
    print("Converting frames to grayscale for comparison...")
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection to grayscale frames
    print("Applying Canny edge detection to frames...")
    edges1 = cv2.Canny(frame1_gray, 100, 150)
    edges2 = cv2.Canny(frame2_gray, 100, 150)

    # Display the frames and the difference using Matplotlib
    fig = plt.figure(figsize=(15, 7)) # Adjusted figure size for better layout

    # Original frames side-by-side
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    ax1.set_title(f'Video 1: First Frame\n{os.path.basename(video1_path)}')
    ax1.axis('off')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2RGB))
    ax2.set_title(f'Video 2: First Frame (Resized)\n{os.path.basename(video2_path)}')
    ax2.axis('off')

    # Edge-detected frames side-by-side
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(edges1, cmap='gray')
    ax3.set_title(f'Video 1: Edges')
    ax3.axis('off')

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(edges2, cmap='gray')
    ax4.set_title(f'Video 2: Edges')
    ax4.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.suptitle('Video Frame Comparison', fontsize=16)
    plt.show(block=True) # Keep plot window open until closed by user

def browse_file(entry_widget):
    """Opens a file dialog and updates the entry widget with the selected file path."""
    
    filename = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")])
    if filename:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, filename)

if __name__ == "__main__":

    root = tk.Tk()
    root.title("Video Frame Diff Checker")
    root.geometry("550x150") # Set initial window size

    # Video 1 selection
    label_video1 = tk.Label(root, text="Video 1:")
    label_video1.grid(row=0, column=0, padx=10, pady=5, sticky="w")
    entry_video1 = tk.Entry(root, width=50)
    entry_video1.grid(row=0, column=1, padx=10, pady=5)
    button_browse1 = tk.Button(root, text="Browse", command=lambda: browse_file(entry_video1))
    button_browse1.grid(row=0, column=2, padx=10, pady=5)

    # Video 2 selection
    label_video2 = tk.Label(root, text="Video 2:")
    label_video2.grid(row=1, column=0, padx=10, pady=5, sticky="w")
    entry_video2 = tk.Entry(root, width=50)
    entry_video2.grid(row=1, column=1, padx=10, pady=5)
    button_browse2 = tk.Button(root, text="Browse", command=lambda: browse_file(entry_video2))
    button_browse2.grid(row=1, column=2, padx=10, pady=5)

    # Compare button
    button_compare = tk.Button(root, text="Compare Frames",
                               command=lambda: compare_and_show_first_frames(entry_video1.get(), entry_video2.get()))
    button_compare.grid(row=2, column=1, pady=10)

    root.mainloop()