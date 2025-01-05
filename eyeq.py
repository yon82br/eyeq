#!/usr/bin/env python
# coding: utf-8


import functools
import subprocess
import tkinter as tk
from functools import partial
from glob import glob
from pathlib import Path
from time import ctime
from tkinter import filedialog, ttk
import logging

import cv2
import numpy as np
import pandas as pd
from aicspylibczi import CziFile as czi
from matplotlib import pyplot as plt
from matplotlib.pyplot import Normalize
from pandas import DataFrame as Df
from pandas import Series as Ser
from PIL import Image
from PIL.ImageTk import PhotoImage
from scipy.ndimage import grey_dilation as g_dil
from tqdm.auto import tqdm


# In[4]:


## image utils
def piximg(x: np.ndarray, f: int) -> np.ndarray:
    """pixelate downscale img x by factor f"""
    assert isinstance(f, int), 'f must be integer'
    return np.array(Image.fromarray(x).resize((x.shape[0] // f, x.shape[1] // f)))


def g2bgr(x: np.ndarray) -> np.ndarray:
    """gray (1 channel) to bgr (3 channels)"""
    return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)


def bgr2g(x: np.ndarray) -> np.ndarray:
    """bgr (3 channels) to gray (1 channel)"""
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)


def gfilt(x: np.ndarray, s: float) -> np.ndarray:
    '''applies gaussian filter with kernel size s to array x'''
    assert ((isinstance(s, int) or isinstance(s, float)) and s > 0), 's must be number larger that 0'
    return cv2.GaussianBlur(x, (0, 0), s)


def dog(x: np.ndarray, a: float, b: float) -> np.ndarray:
    '''returns Difference of Gaussians (DoG) on array x, between kernel sizes a  and b'''
    assert ((isinstance(a, int) or isinstance(a, float)) and s > 0), 'a must be number larger that 0'
    assert ((isinstance(b, int) or isinstance(b, float)) and s > 0), 'b must be number larger that 0'
    return gfilt(x, a) - gfilt(x, b)


def qnorm(x):
    '''linearly stretches x to range 0:1, x being list like'''
    return Normalize()(x)


def erode(x: np.ndarray, s: int, n: int):
    """erode x with kernel size s*s with n iterations"""
    kern = np.ones((s, s))
    return cv2.erode(x, kernel=kern, iterations=n)


# In[5]:


import logging


# In[6]:


## advanced image manipulations
def find_contours(
    img, lb=127, ub=255, return_contours=0, to_draw=None, width=3, color=None
):
    """finds contours with option to either return the contours with hierarchy or return image to_draw with contours in the image
    params lb ub : lower and upper bounds
    param retcont : boolean, whether to return the contours in tuple of lists, or return image with contours drawn on.
    param to_draw : image to draw the contours on, optional.
    param width : width of contours to draw. negative value fills the contour."""
    t = img
    ret, thresh = cv2.threshold(t, lb, ub, 0)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if to_draw is None:
        to_draw = t
    if return_contours:
        return contours, hierarchy
    cv2.drawContours(
        to_draw.copy(), contours, -1, color if color else (0, 255, 0), thickness=width
    )
    return to_draw


def max_diameter_contour(query_image, display_image=None) -> (float, np.ndarray):
    """finds contours in rp, chooses one with largest enclosing circle radius
    param query_image : image to find the contours int
    param display_image : image to draw upon
    returns tuple (diameter, image with enclosing circle as array)
    """
    contours, _ = find_contours(query_image, return_contours=True)
    # find larget contour
    try:
        contour_index = np.argmax(
            [cv2.minEnclosingCircle(contours[i])[1] for i in range(len(contours))]
        )
    except:
        return 0, display_image
    (x, y), radius = cv2.minEnclosingCircle(contours[contour_index])
    center = (int(x), int(y))
    if display_image is None:
        display_image = query_image
    if len(display_image.shape) == 2:
        display_image = g2bgr(display_image)
    retshow = cv2.circle(display_image.copy(), center, int(radius), (0, 255, 0), 2)
    return radius * 2, retshow


def get_lens_contour(image: np.ndarray, todraw=None, mask=False) -> np.ndarray:
    """attempts to find the contour of the lens and the iris"""
    if todraw is None:
        todraw = image.copy()
    if not mask:
        image = g_dil(((image > 6) * 255).astype(np.uint8), 4)
        image = erode(image, 2, 4)
        image = 255 - erode(255 - image, 5, 3)
    contours, hierarchy = find_contours(image, return_contours=True)
    try:
        big_contour_index = np.argmax(
            [cv2.contourArea(contours[i]) for i in range(len(contours))]
        )
    except Exception as e:
        print(e)
        return image * 0
    big_contour = contours[big_contour_index]
    inners = [
        x for i, x in enumerate(contours) if hierarchy[0, i, 3] == big_contour_index
    ]
    try:
        inner = inners[
            np.argmax([cv2.contourArea(inners[i]) for i in range(len(inners))])
        ]
    except:
        inner = None
    if not mask:
        image = gfilt(image, 6)
    contours, hierarchy = find_contours(image, return_contours=True)
    big_contour_index = np.argmax(
        [cv2.contourArea(contours[i]) for i in range(len(contours))]
    )
    big_contour = contours[big_contour_index]
    k = cv2.drawContours(
        np.zeros_like(image),
        contours=[big_contour],
        contourIdx=0,
        color=(255, 255, 255),
        thickness=-1,
    )
    if not inner is None:
        mask = cv2.drawContours(
            k.copy(), contours=[inner], contourIdx=-1, color=(0, 0, 0), thickness=-1
        )
        cont_disp = [big_contour, inner]
    else:
        mask = k
        cont_disp = [big_contour]
    if len(todraw.shape) < 3:
        todraw = g2bgr(todraw)

    show = cv2.drawContours(
        todraw, contours=cont_disp, contourIdx=-1, color=(0, 255, 0), thickness=2
    )
    return show, mask


def stitch(imgs, vertical=1):
    img1, img2 = imgs
    rotate = lambda x: x if vertical else x.T
    if len(img1.shape) == 2:
        img1 = g2bgr(rotate(img1))
    if len(img2.shape) == 2:
        img2 = g2bgr(rotate(img2))

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Get matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find affine transformation
    affine_matrix, inliers = cv2.estimateAffine2D(dst_pts, src_pts)

    # Get dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Calculate the new dimensions after transformation
    corners = np.array(
        [[0, 0], [0, h2 - 1], [w2 - 1, h2 - 1], [w2 - 1, 0]], dtype=np.float32
    ).reshape(-1, 1, 2)
    transformed_corners = cv2.transform(corners, affine_matrix)
    min_x = min(0, np.min(transformed_corners[:, 0, 0]))
    max_x = max(w1, np.max(transformed_corners[:, 0, 0]))
    min_y = min(0, np.min(transformed_corners[:, 0, 1]))
    max_y = max(h1, np.max(transformed_corners[:, 0, 1]))

    # Create the stitched image
    stitched_width = int(max_x - min_x)
    stitched_height = int(max_y - min_y)
    result = np.zeros((stitched_height, stitched_width, 3), dtype=np.uint8)

    # Adjust the affine matrix for the new image size
    affine_matrix[0, 2] -= min_x
    affine_matrix[1, 2] -= min_y

    # Apply affine transformation to img2
    img2_transformed = cv2.warpAffine(
        img2, affine_matrix, (stitched_width, stitched_height)
    )

    # Copy img1 to the result
    result[-int(min_y) : h1 - int(min_y), -int(min_x) : w1 - int(min_x)] = img1

    # Create a mask for smooth blending
    mask = np.zeros((stitched_height, stitched_width), dtype=np.float32)
    cv2.warpAffine(
        np.ones((h2, w2), dtype=np.float32),
        affine_matrix,
        (stitched_width, stitched_height),
        dst=mask,
    )
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=20, sigmaY=20)

    # Blend the images
    result = (
        result * (1 - mask[:, :, np.newaxis])
        + img2_transformed * mask[:, :, np.newaxis]
    )

    return (0, bgr2g(result.astype(np.uint8)))


def get_fish_contour(r):
    """attempts find fish contours"""
    image = r.copy()
    r = (r * (dog(r, 10, 11) > 100)).astype(
        float
    )  # remove transparent areas by removing thin areas
    r[r >= 200] = np.nan  # remove bright colours
    r = g_dil(r, 0) - g_dil(r, 10)  # find expanded area where there is a fish
    r = np.where(np.isnan(r), 0, r)  # remove nans
    r = dog(r, 1, 7)  # find sharp edges
    r = cv2.Canny(
        gfilt(cv2.Canny(gfilt((qnorm(r) * 255).astype(np.uint8), 1), 10, 80), 5), 20, 20
    )  # find high complexity areas
    r = gfilt(g_dil(r, 8), 3)  # expand the high complexity areas
    return max_diameter_contour(r, image)


def get_eye_contour(r):
    """attempts to find eye contours"""
    image = r.copy()
    r = -g_dil(-r, 10)  # expand dark areas
    r = find_contours(gfilt(r, 3), 50, 200)  # find contours after blurring the image
    r = g_dil(r, 10)  # expand bright areas
    r = ((r < 30) * 255).astype(np.uint8)  # make mask of the dark areas
    r = erode(r, 15, 3)  # remove small area masks
    r = g_dil(r, 30)  # expand remaining mask
    return max_diameter_contour(r, image)


# In[7]:


## data loading and unloading
def project_czi(path, method=np.max, channel=1, retstack=0, full_stack=0) -> np.ndarray:
    """returns projection of czi in path, using method for the projection, in channel
    param retstack - flag, enter 1 to get the whole stack
    param fullstack - flag, enter 1 to get all the stacks within the file"""
    tfile = czi(path)
    stack = tfile.read_image(B=0, H=0, T=0)[0]
    stack = stack[tuple(slice(None) if x in 'CZYX' else 0 for x in tfile.dims)]
    return np.apply_along_axis(method, 1, stack)

def unstack_projected_czi(x: pd.Series) -> pd.DataFrame:
    """
    given series with zooms as index, and holding stack of czi projections in each cell
    returns series where each cell holds only one projection, and the index is {zoom}_{channel}
    """
    projections = Df(x.map(lambda x: {i:y for i, y in enumerate(x)}).to_dict()).unstack()
    projections.index = projections.index.map(lambda x: f'{x[0]}_{x[1]}')
    return projections

def save_df(x, path) -> None:
    """save df as np.rec, to save space and loading time"""
    np.save(path, x.to_records(), allow_pickle=True)
    return

def load_df(path, index_cols = []) -> pd.DataFrame:
    """load np.rec object from path into df"""
    return Df.from_records(np.load(path, allow_pickle=True), index = index_cols)


# In[8]:


## widgets and layout
def is_notebook() -> bool:
    """checks if the script is beaing run in a notebook"""
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


def pack_widgets(ls, side="right", padx=2, pady=0) -> None:
    """accepts a list of widget objects and packs them all sequentially"""
    [x.pack(side=side, padx=padx, pady=pady) for x in ls]
    return


def pack_widgs(x) -> None:
    """consistently pack widgets side by side within a frame"""
    pack_widgets(x, padx=2, pady=0, side="left")


def pack_frames(x) -> None:
    """consistently pack frames one on top of the other"""
    pack_widgets(x, padx=0, pady=5, side="top")


def button(master, *args, **kwargs) -> tk.Button:
    """utility to make consistent tk buttons"""
    return ttk.Button(master=master, *args, **kwargs)


def get_screensize() -> (int, int):
    """gets screen measures according to temporarily constructed tk widget"""
    root = tk.Tk()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    return w, h


def tabulate_images_widget(
    ims, master=None, show_labels=True, pack=True
) -> [tk.Tk, tk.Frame]:
    """given df with images, builds window with label per cell, displaying the images, and optional labels on top and right."""
    win = tk.Tk() if master is None else master

    # Create a Frame to hold the content
    frame = ttk.Frame(master=win)
    imsize = get_screensize()[0] // ims.shape[1]
    width = ims.shape[1] * imsize
    height = ims.shape[0] * imsize

    # Add a 'X' button to close the frame
    close_button = tk.Button(frame, command=frame.destroy, background="red", text="   ")
    close_button.pack(side=tk.TOP, anchor="ne")

    # Create a canvas and scrollbars inside the frame
    canvas = tk.Canvas(frame, width=width, height=height)
    vbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
    hbar = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=canvas.xview)
    canvas.config(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
    vbar.pack(side=tk.RIGHT, fill=tk.Y)
    hbar.pack(side=tk.TOP, fill=tk.X)

    # Create a scrollable content frame
    scrollable_frame = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    # Resize and convert images to Tkinter PhotoImage
    Ims = (
        ims.map(Image.fromarray)
        .map(lambda x: x.resize((imsize, imsize)))
        .map(partial(PhotoImage, master=frame))
    )
    labels = Ims.map(lambda x: tk.Label(master=scrollable_frame, image=x))

    if show_labels:
        # Create header labels
        for col, header in enumerate(ims.columns):
            tk.Label(
                master=scrollable_frame, text=str(header), font=("Arial", 10, "bold")
            ).grid(row=0, column=col + 1)
        # Create index labels
        for row, index in enumerate(ims.index):
            tk.Label(
                master=scrollable_frame, text=str(index), font=("Arial", 10, "bold")
            ).grid(row=row + 1, column=0)
        # Add images with an offset for the labels
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                labels.iloc[i, j].grid(row=i + 1, column=j + 1)
    else:
        # Add images without labels
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                labels.iloc[i, j].grid(row=i, column=j)

    # Update scroll region
    scrollable_frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

    canvas.pack(fill=tk.BOTH, expand=True)
    if pack:
        frame.pack(fill=tk.BOTH, expand=True)

    # Attach the persistent reference to the frame
    frame.images = Ims
    frame.labels = labels
    frame.imsize = imsize

    return win if master is None else frame


class ThresholdApp:
    def __init__(self, master, ims):
        self.master = master
        self.ims = ims
        self.locked_columns = []  # Columns that are "frozen"
        self.buttons = {}  # Store buttons
        self.frozen_values = {}  # Store frozen threshold values
        self.frozen_and_non_frozen_values = (
            {}
        )  # Store non-frozen column threshold values

        self.create_widgets()

    def create_widgets(self):
        """Initialize widgets for the thresholding app."""
        # Create scale widget
        frame = tabulate_images_widget(self.ims, master=self.master, pack=False)
        self.labels = frame.labels
        self.scale = self.create_scale_window()

        # Create "Set" buttons for each column
        button_frame = tk.Frame(self.master)
        button_frame.pack(side=tk.TOP, pady=10)

        for col in range(self.ims.shape[1]):
            self.create_set_button(button_frame, col)

        # Create the images tabulation frame
        frame.pack()

        # Attach window close handler
        self.master.protocol("WM_DELETE_WINDOW", self.on_window_close)

    def create_scale_window(self):
        """Create the scale widget."""
        scale = tk.Scale(self.master, from_=0, to=255, orient=tk.HORIZONTAL)
        scale.set(128)  # Default threshold
        scale.pack(fill=tk.X, padx=10, pady=10)
        scale.bind(
            "<Motion>", lambda event: self.update_threshold_continuously(scale.get())
        )
        return scale

    def create_set_button(self, frame, column_index):
        """Create 'Set' button for each column."""
        button = ttk.Button(
            frame,
            text=f"Set channel {column_index + 1}",
            command=partial(self.toggle_lock_column, column_index),
        )
        button.pack(side=tk.LEFT, padx=5)
        self.buttons[column_index] = button

    def toggle_lock_column(self, column_index):
        """Toggle locking/unlocking a column and update button text."""
        if column_index in self.locked_columns:
            self.locked_columns.remove(column_index)
            self.buttons[column_index].config(text=f"Set channel {column_index + 1}")
            self.frozen_values.pop(column_index, None)
        else:
            self.locked_columns.append(column_index)
            threshold = self.scale.get()
            self.buttons[column_index].config(
                text=f"Channel {column_index + 1} Locked at {threshold}"
            )
            self.frozen_values[column_index] = threshold

    def update_threshold_continuously(self, threshold_value):
        """Update images based on the threshold, excluding locked columns."""
        threshold = int(threshold_value)
        for i in range(self.ims.shape[0]):
            for j in range(self.ims.shape[1]):
                if j not in self.locked_columns:
                    img = self.ims.iloc[i, j]
                    thresholded = (np.array(img) > threshold) * 255
                    photo = ImageTk.PhotoImage(
                        image=Image.fromarray(thresholded.astype(np.uint8))
                    )
                    lbl = self.labels.iloc[i, j]
                    lbl.config(image=photo)
                    lbl.image = photo

    def on_window_close(self):
        """Handle the window close event, saving threshold values."""
        # Print frozen values
        print(f"Frozen values: {self.frozen_values}")

        # Save non-frozen values
        for col in range(self.ims.shape[1]):
            if col not in self.locked_columns:
                self.frozen_and_non_frozen_values[col] = self.scale.get()

        print(f"Non-frozen values: {self.frozen_and_non_frozen_values}")
        self.frozen_values = self.frozen_values | self.frozen_and_non_frozen_values

        # Close the window
        self.master.destroy()


def open_threshold_window(master, ims):
    """Helper function to open the threshold window."""
    show_thresh = tk.Toplevel(master)
    app = ThresholdApp(show_thresh, ims)
    return app


# In[9]:


def summarize(obj):
    """Summarize large objects like NumPy arrays and pandas DataFrames."""
    if isinstance(obj, np.ndarray):
        return f"np.ndarray(shape={obj.shape}, dtype={obj.dtype})"
    elif isinstance(obj, pd.DataFrame):
        return f"pd.DataFrame(shape={obj.shape}, columns={list(obj.columns)[:3]}...)"  # Show only first 3 columns
    elif isinstance(obj, pd.Series):
        return f"pd.Series(length={len(obj)}, dtype={obj.dtype})"
    else:
        return repr(obj)  # Default to the object's string representation


def log_calls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        summarized_args = [summarize(arg) for arg in args]
        summarized_kwargs = {k: summarize(v) for k, v in kwargs.items()}
        logger.info(
            f"{ctime()} :: Calling {func.__name__} with args: {summarized_args}, kwargs: {summarized_kwargs}"
        )
        result = func(*args, **kwargs)
        return result

    return wrapper


def log_all_methods(cls):
    for attr_name, attr_value in cls.__dict__.items():
        if callable(attr_value) and not attr_name.startswith(
            "__"
        ):  # Skip special methods
            setattr(cls, attr_name, log_calls(attr_value))
    return cls


# In[10]:


@log_all_methods
class QWidget:
    def __init__(self):
        # initiate window
        # add and pack buttons

        self.root = tk.Tk()
        # self.root.configure(bg = '#0A0A20')
        style = ttk.Style()
        style.theme_use("clam")
        # style.configure("TFrame", background = '#0A0020')
        style.configure("TButton", background = 'SlateGray3', foreground = 'ivory')
        self.root.title("Image Manipulation Widget")
        self.root.protocol(
            "WM_DELETE_WINDOW", lambda: self.close_q(self.root, main=1)
        )  # bind window colse to a prompt if to close the window

        # set naming template
        ## FIX THIS SECTION
        self.naming_convention = "group_age_zoom_part_id_comment_collector"
        self.columns = ["path"] + self.naming_convention.split("_")
        self.instab_columns = "path_group_id_age_zoom_channel".split("_")
        self.stab_zooms = ["X2", "X2b", "X10", "X20"]  # list of zooms to accept
        self.view_zooms = [
            "X2_0",
            "X2b_0",
            "X10_0",
            "X20_0",
            "X20_1",
        ]  # list of zooms to show in image showing window
        self.view_processed = [
            "X2_0s",
            "X2_0s_contour",
            "X10_0_contour",
            "X20_0_contour",
            "X20_1_contour",
        ]
        self.masks_nums = ["X2_0s_length", "X10_0_length", "X20_0_mask"]
        self.proj_path = None
        self.manual_ops = {
            "X2_0s": self.manual_stitch,
            "X2_0s_contour": self.paint_length,
            "X10_0_contour": self.paint_length,
            "X20_0_contour": self.paint_masks,
        }
        self.auto_ops = {
            "X2_0": self.auto_stitch,
            "X2_0s": self.auto_fish_length,
            "X10_0": self.auto_eye_length,
            "X20_0": self.auto_lens_area,
        }

        # constants
        self.ref_thresh = 6
        self.flu_thresh = 6
        self.vertical = True

        # buttons
        self.buttons_frame = ttk.Frame(self.root)
        self.load_dir_button = button(
            self.buttons_frame,
            text="load and verify directory",
            command=self.load_dir,
        )
        self.make_auto_button = button(
            self.buttons_frame, text="automate masks", command=self.automate_masks
        )
        self.show_button = button(
            self.buttons_frame,
            text="show data",
            command=self.show_data,
        )
        self.load_data_button = button(
            self.buttons_frame, text="load data", command=self.load_data
        )
        self.show_thresh_button = button(
            self.buttons_frame, text="show thresholded", command=self.show_thresh
        )
        self.get_results_button = button(
            self.buttons_frame, text="get results", command=self.export_results
        )
        pack_widgs(
            [
                self.load_dir_button,
                self.show_button,
                self.make_auto_button,
                self.load_data_button,
                self.show_thresh_button,
                self.get_results_button,
            ]
        )
        pack_frames([self.buttons_frame])

        # label for information messages, such as progress, warnings etc.
        self.infolabel = tk.Label(self.root, text=self.naming_convention)
        pack_frames([self.infolabel])

    def __repr__(self):
        return "QWidget"

    def close_q(self, root, main=0):
        # function to prompt destroying a window
        # param root : Tk or Toplevel widget to destroy
        # param main : flag if to destroy the whole widget (main window) and quit the script or just destroy a specific window
        win = tk.Toplevel(master=root)
        close_button = button(
            win, text="Close", command=lambda: self.close(self.root, main=main)
        )
        close_button.pack(side="right", padx=2, pady=5)
        cancel_button = button(win, text="Cancel", command=win.destroy)
        cancel_button.pack(side="left", padx=2)

    def close(self, root, main=0):
        """
        function that destroys the window given in argument root
        param main : flag if to quit python program (when closing main window) or just destroy specific window
        """
        root.quit()
        root.destroy()
        if (
            main and not is_notebook()
        ):  # if python runs not in a notebook, kill the kernel
            quit()
            sexit()

    def start(self):
        """rerun widget"""
        self.root.mainloop()

    def checkpoint(self):
        """save projections as numpy"""
        if self.proj_path is None:
            self.proj_path = filedialog.asksaveasfile().name
            self.log_path = Path(self.proj_path).parent / 'eyeq.log'
            logging.basicConfig(filename = self.log_path)
        save_df(self.projections, self.proj_path)
        return

    def load_data(self):
        """loads the projections from numpy"""
        self.proj_path = filedialog.askopenfile().name
        if isinstance(self.proj_path, str):  # if user cancels, cancel gracefully
            self.projections = load_df(
                self.proj_path, index_cols=["age", "group", "id"]
            )
            self.log_path = Path(self.proj_path).parent / 'eyeq.log'
            logging.basicConfig(filename = self.log_path, level = self.log_level)
        return

    def load_dir(self):
        """
        accepts directory holding czi files, groups, projects and tabulates them, then saves the tabulated projections
        """
        self.dir = filedialog.askdirectory()
        files = glob(str(Path(self.dir) / "*.czi"))
        self.files = Df(
            [[x, *Path(x).stem.split("_")[: len(self.columns)]] for x in files],
            columns=self.columns,
        )
        self.files.loc[
            (self.files.zoom == "X2") & (self.files.part == "b"), "zoom"
        ] += "b"  # mark for stitching
        self.samples_paths = self.files.groupby(["age", "group", "id"]).apply(
            lambda x: x.set_index("zoom")["path"]
        )
        self.projections = self.samples_paths.map(partial(project_czi, method=np.max))
        if isinstance(self.projections, pd.Series):
            self.projections = self.projections.unstack(fill_value=[])
        self.projections = self.projections.apply(unstack_projected_czi, axis=1)
        self.projections = self.projections[self.view_zooms]
        # self.projections = self.projections.map(lambda x: QWidget.EMPTY_IMAGE if pd.isna(x) is True else x)
        self.checkpoint()
        return

    EMPTY_IMAGE = np.array([[0, 0], [0, 255]]).astype(
        np.uint8
    )  # placeholder for empty image slots

    def np2photo(self, x, master=None):
        """converts PIL.Image to tk.PhotoImage, for image showing window."""
        if type(x) is np.ndarray:
            tmp = PhotoImage(
                Image.fromarray(piximg(x, 3)),
                master=master if master else self.images_frame,
            )
            return tmp
        return tk.Image(QWidget.EMPTY_IMAGE)

    def show_data(self):
        """
        shows showing columns from the projections df.
        all the (relevant) subwidgets are saved within self.show_window
        """
        view_zooms = self.projections.columns[
            self.projections.columns.isin([*self.view_zooms, *self.view_processed])
        ]
        show_projections = self.projections[view_zooms]
        show_projections = show_projections.map(
            lambda x: QWidget.EMPTY_IMAGE if pd.isna(x) is True else x
        )
        self.show_window = tabulate_images_widget(
            show_projections,
            master=self.root,
        )
        np.vectorize(
            lambda i, j: self.show_window.labels.iloc[i, j].bind(
                "<Double-Button-1>", partial(self.redirect_op, i, j, manual=True)
            )
        )(*np.indices(self.show_window.labels.shape))
        return

    def redirect_op(self, i, j, event, manual=True):
        """
        redirects click events and auto masking to correct function.
        """
        row = self.show_window.labels.index[i]
        column = self.show_window.labels.columns[j]
        opdict = self.manual_ops if manual else self.auto_ops
        func = opdict.get(column, lambda *x: None)
        func(i, j, row, column)
        return

    def paint_length(self, i, j, row, column):
        """
        prompts user to draw the length of the object in given image. must be drawn in white.
        """
        # get original image (infer based on zoom. choose mask names wisely for this)
        zoom = "_".join(column.split("_")[:2])
        image = self.projections.at[row, zoom].clip(max=254)
        # save image in temporary file in original directory
        tmp = Path(self.proj_path).parent / "_tmp.tmp.png"
        Image.fromarray(image).save(tmp)
        try:
            # open mspaint with temporary file
            subprocess.Popen(["mspaint.exe", tmp])
            # ask user to press ok when done, or cancel if cancel job
            answer = tk.messagebox.askokcancel(
                "Question", "When you are finished with paint, save and then press OK."
            )
            if not answer:
                logger.info(f'{row} {column} tried manual operation and cancelled')
                # tmp.unlink(missing_ok = True)
                return
            # read image
            with Image.open(tmp) as processed_image:
                processed_image = processed_image.copy()
        except Exception as e:
            logger.error("exception was raised", exc_info = True)
            # tmp.unlink(missing_ok = True)
            raise e
        # delete file
        tmp.unlink(missing_ok=True)
        # make necessary conversions to image (BGR2G...)
        processed_image = ((bgr2g(np.array(processed_image)) == 255) * 255).astype(
            np.uint8
        )
        # do calculation, recieve length and new visual
        length, visual = max_diameter_contour(processed_image, image)
        self.projections.at[row, zoom + "_length"] = length
        # update images and checkpoint
        self.update_image(visual, i, j, row, column)
        return

    def paint_masks(self, i, j, row, column):
        """
        mostly general besides the very end. prompts the user to draw positive then negative mask
        of desired object in the image, then updates display and data accordingly.
        """
        tmp = Path(self.proj_path).parent / "_tmp.tmp.png"
        image = self.projections.at[row, "X20_0"]
        # save tmp image
        Image.fromarray(cv2.equalizeHist(image).clip(1, 254)).save(tmp)
        # open ms paint, confirm user finished
        subprocess.Popen(["mspaint.exe", tmp])
        answer = tk.messagebox.askokcancel(
            "Question",
            "When you are finished with paint for positive mask, save and then press OK.",
        )
        if not answer:
            logger.info(f'{row} {column} tried manual operation and cancelled')
            tmp.unlink(missing_ok=True)
            return
        # save image as positive mask
        with Image.open(tmp) as mask:
            positive_mask = bgr2g(np.array(mask)) == 255
        # resave image after masking, open mspaint, confirm user finished
        Image.fromarray(cv2.equalizeHist(image * positive_mask)).save(tmp)
        subprocess.Popen(["mspaint.exe", tmp])
        answer = tk.messagebox.askokcancel(
            "Question",
            "When you are finished with paint for negative mask, save and then press OK.",
        )
        if not answer:
            logger.info(f'{row} {column} tried manual operation and cancelled')
            tmp.unlink(missing_ok=True)
            return
        # open image, save to array and delete tmp file
        with Image.open(tmp) as mask:
            negative_mask = bgr2g(np.array(mask)) != 255
        tmp.unlink(missing_ok=True)
        # combine with negative mask
        mask = positive_mask & negative_mask
        # update mask in projections
        self.projections.at[row, "X20_0_mask"] = mask
        # get contour
        contoured_image, _ = get_lens_contour(
            (mask * 255).astype(np.uint8), image, mask=True
        )
        # update mask images and checkpoint
        self.update_image(contoured_image, i, j, row, column)

    def manual_stitch(self, i, j, row, column):
        """
        displays X2_0 and X2b_0 one on top of the other in mspaint, and asks the user to place them properly instead of automatic stitching.
        update display and data, adn reattempts to calculate fish length at the end.
        """
        tmp = Path(self.proj_path).parent / "_tmp.tmp.png"
        image = np.vstack(self.projections.loc[row, ["X2_0", "X2b_0"]])
        # save image
        Image.fromarray(image).save(tmp)
        # open mspaint, ask for confirmation when done
        subprocess.Popen(["mspaint.exe", tmp])
        answer = tk.messagebox.askokcancel(
            "Question", "When you are finished with paint, save and then press OK."
        )
        if not answer:
            logger.info(f'{row} {column} tried manual operation and cancelled')
            tmp.unlink(missing_ok=True)
            return
        # open file, save copy to variable
        with Image.open(tmp) as file:
            processed_image = np.array(file)
        tmp.unlink(missing_ok=True)
        # update image
        self.update_image(processed_image, i, j, row, column)
        # recalculate fish length and update those images as well
        self.auto_fish_length(i, j, row, column)

    def auto_stitch(self, i, j, row, column):
        ## get images, drop na
        images = self.projections.loc[row, ["X2_0", "X2b_0"]].dropna()
        if len(images) <= 1:
            self.update_image(images if len(images) else np.nan, i, j, row, "X2_0s")
            return
        ## calculate stitch
        try:
            image = stitch(images, vertical=self.vertical)
            if image[0]:
                raise Exception(f"error occured with stitching row {row}")
            image = image[1]
        except Exception as e:
            logger.error(f'failed stitching {row = } {column = }', exc_info = True)
            image = images[0]
            raise e
        ## update image
        self.update_image(image, i, j, row, "X2_0s")

    def auto_fish_length(self, i, j, row, column):
        ## get image
        image = self.projections.at[row, column]
        if isinstance(image, float):
            logger.info(f'{row} {column} image is empty')
            self.update_image(image, i, j, row, column)
            return
        # contour = get_fish_contour(image)
        length, circled_fish = get_fish_contour(image)
        self.projections.at[row, "X2_0s_length"] = length
        self.update_image(circled_fish, i, j, row, "X2_0s_contour")

    def auto_eye_length(self, i, j, row, column):
        ## get image
        image = self.projections.at[row, column]
        if isinstance(image, float):
            logger.info(f'{row} {column} image is empty')
            self.update_image(image, i, j, row, column)
            return
        # contour = get_eye_contour(image)
        length, circled_eye = get_eye_contour(image)
        self.projections.at[row, "X10_0_length"] = length
        self.update_image(circled_eye, i, j, row, "X10_0_contour")

    def auto_lens_area(self, i, j, row, column):
        ## get image
        image = self.projections.at[row, column]
        if isinstance(image, float):
            logger.info(f'{row} {column} image is empty')
            self.update_image(image, i, j, row, column)
            return
        contoured_lens, mask = get_lens_contour(image)
        mask = mask == 255
        self.projections.at[row, "X20_0_mask"] = mask
        self.update_image(contoured_lens, i, j, row, "X20_0_contour")
        if "X20_1" in self.view_zooms:
            try:
                self.update_image(
                    mask * self.projections.at[row, "X20_1"], i, j, row, "X20_1_contour"
                )
            except Exception as e:
                print(e)

    def automate_masks(self):
        ### initiate empty columns to populate
        self.projections[self.view_processed] = np.nan
        self.projections[self.masks_nums] = np.nan
        self.projections = self.projections.astype(object)
        ### deal with case (likely case) where there is no display
        if not hasattr(self, "show_window"):
            self.show_data()
        else:
            self.show_window.destroy()
            self.show_data()
        self.vertical = tk.simpledialog.askinteger(
            "stitching direction", "to stitch horizontally, type 0. vertical, type 1."
        )
        # perform auto calculations on relevant categories.
        for column, func in self.auto_ops.items():
            for i in tqdm(range(len(self.show_window.images))):
                ## get j from show_window labels
                j = self.show_window.images.columns.tolist().index(column)
                self.redirect_op(i, j, None, manual=False)

    def update_image(self, image: np.ndarray, i, j, row, column):
        """updates the display at location i,j and data at location row,column, according to image."""
        # get label and update images and projections, checkpoint
        photoimage = QWidget.EMPTY_IMAGE if isinstance(image, float) else image
        label = self.show_window.labels.at[row, column]
        self.show_window.images.at[row, column] = PhotoImage(
            Image.fromarray(photoimage).resize((self.show_window.imsize,) * 2),
            master=self.show_window,
        )
        try:
            self.projections.at[row, column] = image
        except:
            logger.critical('failed to update image.', exc_info = True)
            self.debug_obj = image
            self.close(self.root)
            return
        self.checkpoint()
        # reconfig label with new image
        label.config(image=self.show_window.images.at[row, column])

    def show_thresh(self):
        self.thresh_window = open_threshold_window(
            self.root,
            (
                self.projections[["X20_0", "X20_1"]].T * self.projections["X20_0_mask"]
            ).T.dropna(),
        )
        self.ref_thresh, self.flu_thresh = self.thresh_window.frozen_values.values()

    def export_results(self):
        self.results_path = filedialog.asksaveasfile()
        if not self.results_path:
            return
        else:
            self.results_path = self.results_path.name
        res = self.projections[["X2_0s_length", "X10_0_length"]]
        res.loc[:, "eye_area"] = self.projections["X20_0_mask"].map(
            lambda x: x.sum() if isinstance(x, np.ndarray) else x
        )
        res.loc[:, "eye_ref_area"] = self.projections.apply(
            axis=1,
            func=lambda x: ((x["X20_0"] > self.ref_thresh) & x["X20_0_mask"]).sum()
            if isinstance(x["X20_0"], np.ndarray)
            else np.nan,
        )
        res.loc[:, "eye_flu_area"] = self.projections.apply(
            axis=1,
            func=lambda x: ((x["X20_1"] > self.flu_thresh) & x["X20_0_mask"]).sum()
            if isinstance(x["X20_1"], np.ndarray)
            else np.nan,
        )
        res.to_csv(self.results_path)


# In[11]:


if __name__ == '__main__':
    widg = QWidget()
    logger = logging.getLogger(__name__)
    widg.log_level = logging.INFO
    logger.info(f'{ctime()} :: starting widget')
    widg.start()
    logger.info(f'{ctime()} :: closed program')


# In[ ]:




