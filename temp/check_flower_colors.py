import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
potato_flower_dir = os.path.join(BASE_DIR, "plant_disease_detection", "plant_disease_dataset", "Potato", "Flower")
tomato_flower_dir = os.path.join(BASE_DIR, "plant_disease_detection", "plant_disease_dataset", "Tomato", "Flower")

def get_flower_color_stats(img_path):
    img = cv2.imread(img_path)
    if img is None: return 0, 0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Yellow color range (h: 15-35, s: 80-255, v: 80-255)
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_pct = (np.count_nonzero(yellow_mask) / (img.shape[0] * img.shape[1])) * 100.0

    # White / Light purple range (h: 0-180, s: 0-60, v: 160-255)
    lower_white = np.array([0, 0, 160])
    upper_white = np.array([180, 70, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    white_pct = (np.count_nonzero(white_mask) / (img.shape[0] * img.shape[1])) * 100.0

    return yellow_pct, white_pct

out_file = os.path.join(BASE_DIR, "temp", "flower_color_stats.txt")
with open(out_file, "w", encoding="utf-8") as out:
    def p(text=""):
        print(text)
        out.write(text + "\n")

    p("=== POTATO FLOWERS ===")
    for f in os.listdir(potato_flower_dir):
        if not f.lower().endswith(('.jpg','.jpeg','.png')): continue
        ypct, wpct = get_flower_color_stats(os.path.join(potato_flower_dir, f))
        p(f"File: {f:18s} | Yellow %: {ypct:5.2f}% | White/Pale %: {wpct:5.2f}%")

    p("\n=== TOMATO FLOWERS ===")
    for f in os.listdir(tomato_flower_dir):
        if not f.lower().endswith(('.jpg','.jpeg','.png')): continue
        ypct, wpct = get_flower_color_stats(os.path.join(tomato_flower_dir, f))
        p(f"File: {f:18s} | Yellow %: {ypct:5.2f}% | White/Pale %: {wpct:5.2f}%")

