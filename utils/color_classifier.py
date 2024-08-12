import cv2
import numpy as np

def classify_color(hsv_crop, threshold=10000):
    color_ranges = {
        'red': [(0, 70, 50), (10, 255, 255)],
        'green': [(40, 40, 40), (70, 255, 255)],
        'blue': [(100, 150, 0), (140, 255, 255)],
        'yellow': [(20, 100, 100), (30, 255, 255)],
        'white': [(0, 0, 200), (180, 20, 255)],
        'black': [(0, 0, 0), (180, 255, 30)],
        'gray': [(0, 0, 100), (180, 50, 200)]
    }

    color_pixel_counts = {color: 0 for color in color_ranges}

    for color, (lower, upper) in color_ranges.items():
        lower_np = np.array(lower, dtype="uint8")
        upper_np = np.array(upper, dtype="uint8")
        mask = cv2.inRange(hsv_crop, lower_np, upper_np)
        
        count = cv2.countNonZero(mask)
        color_pixel_counts[color] = count

    detected_colors = [color for color, count in color_pixel_counts.items() if count > threshold]
    detected_colors = ", ".join(detected_colors)

    return detected_colors

def recognize_color(image: np.ndarray = None):
    assert image is not None
    car_crop = image
    hsv_crop = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
    car_color = classify_color(hsv_crop)
    return car_color

if __name__ == "__main__":
    recognize_color()
