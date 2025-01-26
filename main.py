import os
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
import time

def compress_image(image_path, max_size_kb=100):
    """Compress image if it's larger than max_size_kb."""
    file_size = os.path.getsize(image_path) / 1024  # size in KB
    if file_size > max_size_kb:
        print(f"Image size is {file_size:.2f} KB, compressing it...")
        img = cv2.imread(image_path)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]  # Set JPEG quality to 80%
        result, encoded_img = cv2.imencode('.jpg', img, encode_param)
        if result:
            compressed_image_path = image_path.replace(".jpg", "_compressed.jpg")
            with open(compressed_image_path, 'wb') as f:
                f.write(encoded_img)
            print(f"Image compressed and saved as {compressed_image_path}")
            return compressed_image_path
    return image_path  # No compression needed

def save_large_dataframe_to_excel(df, writer, sheet_name):
    """Save large DataFrame to Excel with multiple sheets."""
    sheet_chunk_size = 1048576  # Excel's row limit per sheet
    num_chunks = len(df) // sheet_chunk_size + 1

    for i in range(num_chunks):
        chunk = df.iloc[i * sheet_chunk_size : (i + 1) * sheet_chunk_size]
        chunk.to_excel(writer, sheet_name=f"{sheet_name}_{i + 1}", index=False)

def run_yolo_object_detection(image_path, yolo_config, yolo_weights, yolo_classes):
    """Perform YOLO-based object detection."""
    net = cv2.dnn.readNet(yolo_weights, yolo_config)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)
    
    detected_objects = []
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                detected_objects.append((yolo_classes[class_id], confidence, (x, y, w, h)))
                print(f"Object detected: {yolo_classes[class_id]} with confidence {confidence * 100:.2f}%")
    return detected_objects

def process_and_simulate(image_path, output_file, ground_truth_distance=None):
    start_time = time.time()  # Start the timer to measure processing time

    # --- Step 1: Load and Preprocess Image ---
    image_path = compress_image(image_path)  # Compress the image if needed
    img = cv2.imread(image_path)  # Load the image in color (RGB)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    scale_factor = 0.5  # Resize to 50%
    img_resized = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    img_smoothed = gaussian_filter(img_resized, sigma=2)
    img_normalized = cv2.normalize(img_smoothed, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    if len(img_normalized.shape) == 3:
        img_normalized = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2GRAY)

    height, width = img_normalized.shape

    # --- Step 2: Simulate Dummy Data ---
    camera_data = [(x, y, img_normalized[y, x]) for y in range(height) for x in range(width)]
    camera_df = pd.DataFrame(camera_data, columns=["X", "Y", "Intensity"])

    lidar_data = [(x, y, np.random.uniform(0, 10)) for y in range(height) for x in range(width) if img_normalized[y, x] > 0]
    lidar_df = pd.DataFrame(lidar_data, columns=["X", "Y", "Z"])

    num_objects = 10
    ranges = np.random.uniform(1, 20, num_objects)
    angles = np.random.uniform(0, 360, num_objects)
    velocities = np.random.uniform(-10, 10, num_objects)
    radar_data = []
    for r, a, v in zip(ranges, angles, velocities):
        x = r * np.cos(np.radians(a))
        y = r * np.sin(np.radians(a))
        radar_data.append((r, a, v, x, y))
    radar_df = pd.DataFrame(radar_data, columns=["Range", "Angle", "Velocity", "X", "Y"])

    # Save to Excel with multiple sheets
    sheet_chunk_size = 1000000  # Set the chunk size for the sheet
    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
        for i in range(0, len(camera_df), sheet_chunk_size):
            chunk = camera_df.iloc[i:i + sheet_chunk_size]
            chunk.to_excel(writer, sheet_name=f"Camera_{i // sheet_chunk_size + 1}", index=False)

        save_large_dataframe_to_excel(lidar_df, writer, "Lidar Data")
        radar_df.to_excel(writer, sheet_name="Radar Data", index=False)

    print(f"Dummy data saved to {output_file}")

    # --- Step 3: Object Detection ---
    _, thresh = cv2.threshold(img_normalized, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(largest_contour)
    object_size_in_pixels = w * h
    estimated_distance = 1000 / object_size_in_pixels
    meters_per_unit = 24.28
    estimated_distance_meters = estimated_distance * meters_per_unit
    print(f"Estimated distance to object: {estimated_distance_meters:.2f} meters")

    net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    blob = cv2.dnn.blobFromImage(img_resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)


    for out in detections:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Only consider confident detections
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Bounding box corners
                x = center_x - w // 2
                y = center_y - h // 2

                # Calculate object size in pixels
                object_size_in_pixels = w * h

                # Estimate distance (example formula: inverse proportional to size in pixels)
                estimated_distance = 1000 / object_size_in_pixels if object_size_in_pixels > 0 else float('inf')
                meters_per_unit = 89.29  # Conversion factor (modify as needed for your application)
                estimated_distance_meters = estimated_distance * meters_per_unit

                # Draw bounding box and label
                cv2.rectangle(img_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{classes[class_id]}: {confidence:.2f}, {estimated_distance_meters:.2f}m"
                cv2.putText(img_resized, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                print(f"Detected object: {classes[class_id]} | Confidence: {confidence:.2f} | Distance: {estimated_distance_meters:.2f} meters")


    if ground_truth_distance:
        error = abs(estimated_distance_meters - ground_truth_distance)
        print(f"Error in distance estimation: {error:.2f} meters")

    left_img = cv2.imread("left.jpg", cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread("right.jpg", cv2.IMREAD_GRAYSCALE)

    if left_img is None or right_img is None:
        print("Error: Unable to load stereo images.")
        return

    if left_img.shape != right_img.shape:
        right_img = cv2.resize(right_img, (left_img.shape[1], left_img.shape[0]))

    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(left_img, right_img)

    Q = np.array([
        [1, 0, 0, -0.5],
        [0, 1, 0, -0.5],
        [0, 0, 0, 0.5],
        [0, 0, -1/100, 0]
    ], dtype=np.float32)

    depth_map = cv2.reprojectImageTo3D(disparity, Q)

    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)

    z = img_normalized.astype(np.float64) + np.random.normal(scale=5, size=img_normalized.shape)

    fig = go.Figure(data=[go.Surface(z=z, x=x_grid, y=y_grid, colorscale='Viridis', colorbar=dict(title='Intensity'))])

    camera = dict(eye=dict(x=1.5, y=1.5, z=1.5))
    fig.update_layout(
        title='Enhanced 3D Image Plot',
        scene=dict(
            xaxis=dict(title='X', showgrid=False, zeroline=False),
            yaxis=dict(title='Y', showgrid=False, zeroline=False),
            zaxis=dict(title='Intensity', showgrid=False, zeroline=False),
        ),
        width=900,
        height=900,
        scene_camera=camera,
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(label="Default View", method="relayout", args=[{"scene.camera": camera}]),
                    dict(label="Top-Down View", method="relayout", args=[{"scene.camera.eye": {"x": 0, "y": 0, "z": 2}}]),
                    dict(label="Side View", method="relayout", args=[{"scene.camera.eye": {"x": 2, "y": 0, "z": 0}}]),
                ]
            )
        ]
    )

    fig.show()
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")


# Input data
image_path = "input.jpg"
output_file = "output_data.xlsx"
ground_truth_distance = 1.52  # Optional, if available

process_and_simulate(image_path, output_file, ground_truth_distance)

