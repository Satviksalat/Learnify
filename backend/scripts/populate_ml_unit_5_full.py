
import json
import os

# PATHS
EXAM_FILE = os.path.join("backend", "data", "exam_questions.json")

# FULL EXPERT CONTENT FOR ML UNIT 5
ML_UNIT_5_DATA = {
    "unit": "ML Unit 5: Computer Vision with OpenCV",
    "sections": {
        "Part A (1-Mark)": [
            {"question": "Define Computer Vision.", "answer": "A field of AI that enables computers to derive meaningful information from digital images and videos."},
            {"question": "What is OpenCV?", "answer": "Open Source Computer Vision Library: A huge open-source library for computer vision and ML."},
            {"question": "Define a Pixel.", "answer": "The smallest unit of a digital image, representing a single point of color."},
            {"question": "RGB vs Grayscale.", "answer": "RGB has 3 channels (Red, Green, Blue). Grayscale has 1 channel (Intensity of light)."},
            {"question": "Define Image Resolution.", "answer": "The detail an image holds, usually measured in pixels (Width x Height)."},
            {"question": "What is FPS?", "answer": "Frames Per Second. The frequency at which consecutive images appear on a display."},
            {"question": "What is a Haar Cascade?", "answer": "A machine learning object detection method used to identify objects in images or video."},
            {"question": "Define Face Detection.", "answer": "The technology that determines the location and size of human faces in arbitrary images."},
            {"question": "Define Object Detection.", "answer": "Identifying and locating objects of a certain class (like cars, humans) in an image."},
            {"question": "What is ROI?", "answer": "Region of Interest. A portion of an image that you want to filter or perform some operation on."},
            {"question": "What is a Bounding Box?", "answer": "A rectangular box drawn around an object in an image to define its location."},
            {"question": "Define Thresholding.", "answer": "A method of image segmentation that converts a grayscale image into a binary image (Black/White)."},
            {"question": "What is Edge Detection?", "answer": "An image processing technique for finding the boundaries of objects within images."},
            {"question": "Define Convolution.", "answer": "A mathematical operation where a kernel (filter) slides over an image to extract features."},
            {"question": "What is a Kernel (Filter)?", "answer": "A small matrix used for blurring, sharpening, embossing, or edge detection."},
            {"question": "Define Image Segmentation.", "answer": "Partitioning an image into multiple segments (sets of pixels) to simplify representation."},
            {"question": "What is Optical Flow?", "answer": "The pattern of apparent motion of objects between two frames caused by the movement of the object or camera."},
            {"question": "Define Background Subtraction.", "answer": "A technique to extract moving foreground objects from a static background camera feed."},
            {"question": "What is Feature Matching?", "answer": "Finding corresponding points/features between two images of the same scene."},
            {"question": "Define Blob Detection.", "answer": "Identifying regions in a digital image that differ in properties, such as brightness or color, from surrounding regions."}
        ],
        "Part B (2-Marks)": [
            {"question": "Image Classification vs Object Detection.", "answer": "• Classification: 'Is there a cat?' (Label).\n• Detection: 'Where is the cat?' (Label + Location/Box)."},
            {"question": "Haar Cascade Features.", "answer": "• Rectangular filters (Edge, Line, Four-rectangle).\n• Used to detect simple contrast structures (like eyes vs bridge of nose)."},
            {"question": "What is Grayscaling?", "answer": "• Converting a color image to shades of grey.\n• Reduces computational complexity (3 channels -> 1 channel)."},
            {"question": "Goal of Edge Detection.", "answer": "• To capture important events and changes in properties of the world.\n• Simplifies image data by preserving structural properties."},
            {"question": "What is a Classifier?", "answer": "• An algorithm that maps input data to a specific category.\n• Example: Haar Cascade Classifier for faces."},
            {"question": "Explain 'Viola-Jones' algorithm.", "answer": "• The first real-time face detection framework.\n• Components: Haar Features, Integral Image, Adaboost, Cascading."},
            {"question": "Real-time Face Tracking.", "answer": "• Detecting a face in the first frame.\n• Continuously locating it in subsequent frames of a video stream."},
            {"question": "Positive vs Negative Images.", "answer": "• Positive: Images containing the object you want to detect (e.g., Faces).\n• Negative: Images NOT containing the object (e.g., Walls, Trees)."},
            {"question": "Role of Integral Image.", "answer": "• A data structure to quickly calculate sum of pixel values in a grid.\n• Speeds up Haar Feature calculation significantly."},
            {"question": "Adaboost in Haar Cascades.", "answer": "1. Selects the best features from thousands of available features.\n2. Combines weak classifiers into a strong classifier."},
            {"question": "OpenCV Library usage.", "answer": "• Provide common infrastructure for CV apps.\n• To accelerate the use of machine perception in commercial products."},
            {"question": "Video Capture Object.", "answer": "• A class in OpenCV (cv2.VideoCapture) to handle video streams.\n• Can read from a file or directly from a webcam (Index 0)."},
            {"question": "Blur vs Sharpen.", "answer": "• Blur: Smoothens image, reduces noise (Low-pass filter).\n• Sharpen: Enhances edges and details (High-pass filter)."},
            {"question": "What is Pupil Detection?", "answer": "• A specific type of object detection focused on the eye structure.\n• Used in Gaze Tracking and Drowsiness Detection."},
            {"question": "RGB to Grayscale Formula.", "answer": "• Weighted Method: Gray = 0.299*R + 0.587*G + 0.114*B.\n• Matches human eye sensitivity (more sensitive to Green)."}
        ],
        "Part C (3-Marks)": [
            {"question": "Computer Vision Pipeline.", "answer": "1. **Definition:** Workflow to process images.\n2. **Explanation:** Input Image -> Preprocessing (Resize/Gray) -> Feature Extraction -> Classification -> Output.\n3. **Example:** Camera -> Gray -> Find Face -> Draw Box."},
            {"question": "How Digital Images are Stored.", "answer": "1. **Definition:** Grid of numbers.\n2. **Explanation:** An image is a 2D (Grayscale) or 3D (Color) matrix of pixel intensities (0-255).\n3. **Example:** Black is 0, White is 255. A 10x10 RGB image is a 10x10x3 array."},
            {"question": "Haar Cascade Training Process.", "answer": "1. **Definition:** Creating the XML file.\n2. **Explanation:** 1. Collect Positive/Negative images. 2. Extract Haar features. 3. Use Adaboost to select best features. 4. Build Cascade.\n3. **Example:** Training a 'Open Hand' detector."},
            {"question": "Steps in Face Detection.", "answer": "1. **Definition:** Using `detectMultiScale`.\n2. **Explanation:** Load Image -> Convert to Gray -> Load Haar XML -> Run `detectMultiScale` -> Loop through faces -> Draw Rect.\n3. **Example:** The standard 5-line OpenCV code block."},
            {"question": "Canny Edge Detection Steps.", "answer": "1. **Definition:** Popular edge detection algorithm.\n2. **Explanation:** 1. Noise Reduction (Gaussian). 2. Gradient Calculation. 3. Non-maximum Suppression. 4. Hysteresis Thresholding.\n3. **Example:** Finding the outlines of a building."},
            {"question": "Thresholding Techniques.", "answer": "1. **Definition:** Binarization methods.\n2. **Explanation:** Simple: Fixed cutoff (e.g., >127). Adaptive: Cutoff varies based on local neighborhood (lighting changes).\n3. **Example:** Adaptive is better for scanning documents with shadows."},
            {"question": "Background Subtraction Method.", "answer": "1. **Definition:** Detecting motion.\n2. **Explanation:** Frame(Current) - Frame(Background) = Difference. Large difference = Motion/Foreground.\n3. **Example:** Security camera recording only when someone walks in."},
            {"question": "Object Tracking vs Detection.", "answer": "1. **Definition:** Tracking vs Finding.\n2. **Explanation:** Detection happens in every frame (slow). Tracking locates in frame 1 and estimates motion in frame 2 (fast).\n3. **Example:** Detection is finding a ball. Tracking is following it."},
            {"question": "CNN vs Traditional CV.", "answer": "1. **Definition:** Deep Learning vs Manual Features.\n2. **Explanation:** Trad CV: Manual feature design (SIFT/Haar). CNN: Learns features automatically from data.\n3. **Example:** CNNs (like YOLO) are now standard for high accuracy."},
            {"question": "Drawing Shapes in OpenCV.", "answer": "1. **Definition:** Annotation.\n2. **Explanation:** `cv2.rectangle()`, `cv2.circle()`, `cv2.putText()`. Modifies the image matrix in-place.\n3. **Example:** Drawing a Green box around a detected face."},
            {"question": "Color Spaces: HSV vs RGB.", "answer": "1. **Definition:** Ways to represent color.\n2. **Explanation:** RGB: Red/Green/Blue (Hardware based). HSV: Hue/Saturation/Value (Perception based).\n3. **Example:** HSV is easier for color tracking (e.g., 'Track Blue Objects')."},
            {"question": "Eye Detection Logic.", "answer": "1. **Definition:** Hierarchical detection.\n2. **Explanation:** First detect Face (ROI). Then search for Eyes strictly INSIDE the Face ROI. Reduces false positives.\n3. **Example:** Prevents detecting a 'knot in a tree' as an eye."},
            {"question": "Challenges in Computer Vision.", "answer": "1. **Definition:** Real-world difficulties.\n2. **Explanation:** 1. Illumination variation (Light/Dark). 2. Occlusion (Object hidden). 3. Viewpoint variation (Rotation).\n3. **Example:** Face ID failing in the dark or with a mask."},
            {"question": "Pupil Detection Logic.", "answer": "1. **Definition:** Finding the dark center of the eye.\n2. **Explanation:** 1. Detect Eye. 2. Blur. 3. Threshold (Find darkest point). 4. Blob Detect or Contour find.\n3. **Example:** Driver drowsiness monitoring."},
            {"question": "Reading/Writing Images.", "answer": "1. **Definition:** I/O operations.\n2. **Explanation:** `cv2.imread('file.jpg')` loads to memory. `cv2.imwrite('new.jpg', img)` saves from memory.\n3. **Example:** Saving a screenshot from a video feed."}
        ],
        "Part D (5-Marks)": [
            {
                "question": "Detailed Explanation of Computer Vision.",
                "answer": "1. **Definition:**\n   An interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos.\n\n2. **Goal:**\n   To automate tasks that the human visual system can do (Identification, Tracking, Measurement).\n\n3. **Core Concept:**\n   Pattern Recognition on pixel data. Converting visual signals into structured data.\n\n4. **Technique / Method:**\n   Traditional: Image Processing (Filtering, Edge detection). Modern: Deep Learning (CNNs).\n\n5. **Applications:**\n   Self-Driving Cars (Lane detection), Medical Imaging (X-Ray analysis), Facial Recognition (Security)."
            },
            {
                "question": "Detailed Explanation of OpenCV.",
                "answer": "1. **Definition:**\n   Open Source Computer Vision Library. The standard library for real-time computer vision.\n\n2. **Goal:**\n   To provide a common infrastructure for CV applications and accelerate the use of machine perception.\n\n3. **Core Concept:**\n   Efficient C++ implementation with Python wrappers. Handles Image Processing, Video analysis, and ML.\n\n4. **Technique / Method:**\n   Provides 2500+ optimized algorithms (Haar, SIFT, Canny, Gaussian Blur, etc.).\n\n5. **Applications:**\n   Used by giants like Google, Yahoo, Microsoft, Intel and startups for visual apps."
            },
            {
                "question": "Detailed Explanation of Haar Cascade Classifiers.",
                "answer": "1. **Definition:**\n   An effective object detection method proposed by Viola and Jones in 2001.\n\n2. **Goal:**\n   To detect objects (mainly faces) rapidly in real-time video streams.\n\n3. **Core Concept:**\n   Cascade of Classifiers. Simple features are checked first. If fail, discard window. If pass, check complex features.\n\n4. **Technique / Method:**\n   Uses Haar-like features (Edge/Line features) + Integral Image (Fast Sum) + Adaboost (Selection).\n\n5. **Applications:**\n   Digital Cameras (Focus on face), Access Control Systems, SnapChat filters."
            },
            {
                "question": "Detailed Explanation of Face Detection System.",
                "answer": "1. **Definition:**\n   A computer technology being used in a variety of applications that identifies human faces in digital images.\n\n2. **Goal:**\n   To answer 'Is there a face?' and 'Where is it?' (Detect location and size).\n\n3. **Core Concept:**\n   Scanning the image with a sliding window and checking for face-like features (Eyes darker than bridge of nose).\n\n4. **Technique / Method:**\n   1. Load Image. 2. Grayscale. 3. Load XML (Haar). 4. DetectMultiScale. 5. Return Rectangles.\n\n5. **Applications:**\n   Autofocus in cameras, Face tagging in Facebook, counting people in a crowd."
            },
            {
                "question": "Object Detection in Images Detailed.",
                "answer": "1. **Definition:**\n   A technique involving identifying and locating distinct objects of interest in an image.\n\n2. **Goal:**\n   To assign a Class Label (e.g., 'Car') and a Bounding Box coordinates to every object.\n\n3. **Core Concept:**\n   Distinguishing Foreground objects from Background using learned features.\n\n4. **Technique / Method:**\n   Classical: HOG (Histogram of Oriented Gradients) + SVM. Modern: YOLO (You Only Look Once), SSD.\n\n5. **Applications:**\n   Autonomous driving (Pedestrian detect), Retail (shelf inventory), Security (Weapon detect)."
            },
            {
                "question": "Real-time Face Tracking Detailed.",
                "answer": "1. **Definition:**\n   The process of locating a moving face across multiple frames in a video stream in real-time.\n\n2. **Goal:**\n   To maintain the identity and position of the face continuously without redetecting from scratch every frame.\n\n3. **Core Concept:**\n   Temporal Coherence. The face in frame 2 is likely close to where it was in frame 1.\n\n4. **Technique / Method:**\n   detectMultiScale on Frame 1. Update position based on motion/color histograms (CamShift) in Frame 2+.\n\n5. **Applications:**\n   Video conferencing (Zoom background blur), AR Effects, Gaze interaction."
            },
            {
                "question": "Detailed Explanation of Edge Detection.",
                "answer": "1. **Definition:**\n   An image processing technique for finding the boundaries of objects within images.\n\n2. **Goal:**\n   To detect discontinuities in brightness, which typically correspond to boundaries of objects.\n\n3. **Core Concept:**\n   Gradient. Looking for sharp changes in pixel intensity (e.g., White to Black).\n\n4. **Technique / Method:**\n   Sobel Operator (First derivative), Canny Edge Detector (Multi-stage, robust).\n\n5. **Applications:**\n   Fingerprint recognition, Medical Imaging (Bone fracture), Satellite Image analysis."
            },
            {
                "question": "Detailed Explanation of Pupil/Eye Detection.",
                "answer": "1. **Definition:**\n   A fine-grained detection task to locate the eyes and pupils within a detected face.\n\n2. **Goal:**\n   To understand gaze direction, eye openness (drowsiness), or biometric identification (Iris scan).\n\n3. **Core Concept:**\n   Geometric constraints. Eyes are always in the upper half of the face.\n\n4. **Technique / Method:**\n   1. Detect Face. 2. Extract Face ROI. 3. Detect Eyes XML. 4. Thresholding to find black pupil.\n\n5. **Applications:**\n   Driver Drowsiness Alarms, Eye-controlled mouse for disabled, Lie detection."
            },
            {
                "question": "Challenges in Computer Vision Detailed.",
                "answer": "1. **Definition:**\n   The fundamental problems that make interpreting 3D world from 2D images difficult.\n\n2. **Goal:**\n   To create robust systems that work in the wild, not just in the lab.\n\n3. **Core Concept:**\n   The 'Semantic Gap'. Pixels are just numbers; deriving meaning is hard.\n\n4. **Technique / Method:**\n   Main hurdles: Illumination (Shadows/Glare), Scale (Size changes), Deformation (Non-rigid bodies), Occlusion.\n\n5. **Applications:**\n   Explains why self-driving cars still struggle in heavy snow or chaotic traffic."
            },
            {
                "question": "Case Study: Intruder Detection System.",
                "answer": "1. **Definition:**\n   A security application using CV to detect unauthorized presence.\n\n2. **Goal:**\n   To monitor a static scene and trigger an alert when a human enters.\n\n3. **Core Concept:**\n   Motion Detection via Background Subtraction.\n\n4. **Technique / Method:**\n   1. Save first frame (Emply room). 2. Loop current frames. 3. Diff = Current-First. 4. If Diff > Threshold -> Alarm.\n\n5. **Applications:**\n   Home Security Cameras (Ring), Restricted Area monitoring, Wildlife Traps."
            }
        ]
    }
}

def populate_ml_unit5():
    try:
        if not os.path.exists(EXAM_FILE):
             print("Error: exam_questions.json not found.")
             return

        with open(EXAM_FILE, 'r') as f:
            data = json.load(f)

        # 1. Remove ANY existing "Unit 5: Computer Vision"
        data = [u for u in data if "Unit 5: Computer Vision with OpenCV" not in u['unit']]
        
        # 2. Append the NEW Full Unit
        data.append(ML_UNIT_5_DATA)
        print("Successfully replaced ML Unit 5 with 60 Expert Questions.")

        # 3. Save
        with open(EXAM_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print("Saved to exam_questions.json.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    populate_ml_unit5()
