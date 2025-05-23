import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from joblib import dump, load
import time
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold

# DATASET_PATH = "/Users/e1111742/Documents/FYP/asl_dataset"
DATASET_PATH = '/home/vikas/Downloads/PCSE25-65/asl_dataset'
MODEL_PATH = "sign_language_model.joblib"
IMG_SIZE = (64, 64) 
RANDOM_STATE = 42
TEST_SIZE = 0.2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

def detect_skin(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def extract_hand(image):
    skin_mask = detect_skin(image)
    
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    max_contour = max(contours, key=cv2.contourArea, default=None)
    
    if max_contour is None or cv2.contourArea(max_contour) < 1000:  # Minimum area threshold
        return image
    
    result = image.copy()
    
    x, y, w, h = cv2.boundingRect(max_contour)
    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return result

def extract_hog_features(image):
    if len(image.shape) > 2 and image.shape[2] > 1:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    
    features = hog.compute(gray)
    return features

def load_and_preprocess_data(dataset_path, img_size):
    print("Loading and preprocessing data...")
    start_time = time.time()
    
    images = []
    labels = []
    class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    for class_dir in class_dirs:
        class_path = os.path.join(dataset_path, class_dir)
        print(f"Processing class: {class_dir}")
        
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        max_images = 500
        if len(image_files) > max_images:
            image_files = image_files[:max_images]
        
        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            try:
                processed_image = extract_hand(image)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                processed_image = image  # Fallback to original image
                
            processed_image = cv2.resize(processed_image, img_size)
            
            features = extract_hog_features(processed_image)
            
            images.append(features)
            labels.append(class_dir)
    
    X = np.array(images)
    y = np.array(labels)
    
    X = X.reshape(X.shape[0], -1)
    
    print(f"Data loading completed in {time.time() - start_time:.2f} seconds")
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(set(y))}")
    
    return X, y, list(set(y))

def plot_learning_curve(X, y, filename='learning_curve.png'):
    print("Generating learning curve...")
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))
    ])
    
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    train_sizes = np.linspace(0.1, 1.0, 10)  
    train_scores = []
    val_scores = []
    
    for train_size in train_sizes:
        n_samples = int(len(X) * train_size)
        
        X_subset = X[:n_samples]
        y_subset = y_encoded[:n_samples]
        
        fold_train_scores = []
        fold_val_scores = []
        for train_idx, val_idx in cv.split(X_subset, y_subset):
            X_train, X_val = X_subset[train_idx], X_subset[val_idx]
            y_train, y_val = y_subset[train_idx], y_subset[val_idx]
            
            model.fit(X_train, y_train)
            
            train_acc = accuracy_score(y_train, model.predict(X_train))
            val_acc = accuracy_score(y_val, model.predict(X_val))
            
            fold_train_scores.append(train_acc)
            fold_val_scores.append(val_acc)
        
        train_scores.append(np.mean(fold_train_scores))
        val_scores.append(np.mean(fold_val_scores))
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, 'o-', label='Training accuracy')
    plt.plot(train_sizes, val_scores, 'o-', label='Validation accuracy')
    plt.title('Learning Curve (RandomForest)')
    plt.xlabel('Training Set Size (proportion)')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    print(f"Learning curve saved as '{os.path.join(PLOTS_DIR, filename)}'")
    plt.close()
    
    return train_scores, val_scores

def train_model(X, y):
    print("Training model...")
    start_time = time.time()
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))
    ])
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    print(f"Model accuracy: {accuracy:.4f}")
    
    dump({'model': model, 'label_encoder': label_encoder}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    return model, label_encoder, X_test, y_test, y_pred

def plot_confusion_matrix(y_true, y_pred, class_names, filename='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    print(f"Confusion matrix saved as '{os.path.join(PLOTS_DIR, filename)}'")
    plt.close()

def visualize_predictions(model, label_encoder, dataset_path, class_names, num_images=10):
    test_images = []
    processed_images = []
    true_labels = []
    
    for class_dir in class_names:
        class_path = os.path.join(dataset_path, class_dir)
        if not os.path.isdir(class_path):
            continue
            
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            continue
        image_file = np.random.choice(image_files)
        image_path = os.path.join(class_path, image_file)
        
        image = cv2.imread(image_path)
        if image is None:
            continue
            
        # Store original image for display
        test_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Process image for prediction (now without background removal)
        processed = extract_hand(image)
        processed_images.append(processed)
        
        true_labels.append(class_dir)
        
        if len(test_images) >= num_images:
            break
    
    pred_labels = []
    for image in processed_images:
        resized = cv2.resize(image, IMG_SIZE)
        features = extract_hog_features(resized)
        features = features.reshape(1, -1)
        
        pred_idx = model.predict(features)[0]
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
        pred_labels.append(pred_label)
    
    plt.figure(figsize=(20, 15))
    for i, (orig_image, proc_image, true_label, pred_label) in enumerate(zip(test_images, processed_images, true_labels, pred_labels)):
        if i >= num_images:
            break
            
        plt.subplot(2, num_images, i+1)
        plt.imshow(orig_image)
        plt.title(f"Original", fontsize=10)
        plt.axis('off')
        
        plt.subplot(2, num_images, i+1+num_images)
        plt.imshow(cv2.cvtColor(proc_image, cv2.COLOR_BGR2RGB))
        
        title_color = 'green' if true_label == pred_label else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=title_color, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'predictions.png'))
    print(f"Predictions visualization saved as '{os.path.join(PLOTS_DIR, 'predictions.png')}'")
    plt.close()

def webcam_recognition():
    print("\nStarting webcam recognition. Press 'q' to quit.")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model file {MODEL_PATH} not found. Train the model first.")
        return
    
    print("Loading model...")
    model_data = load(MODEL_PATH)
    model = model_data['model']
    label_encoder = model_data['label_encoder']
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    _, frame = cap.read()
    frame_height, frame_width = frame.shape[:2]
    
    margin = 80 
    roi_top = margin
    roi_bottom = frame_height - 20
    roi_left = 20
    roi_right = frame_width - 20
    
    prev_frame_time = 0
    new_frame_time = 0
    
    prediction_history = []
    max_history = 10
    
    font_scale = 1.5
    font_thickness = 3
    text_color = (0, 255, 0)  
    
    show_processed = True
    
    print("Press 'p' to toggle between original/processed view")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time
        fps_text = f"FPS: {int(fps)}"
        
        cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 3)
        
        roi = frame[roi_top:roi_bottom, roi_left:roi_right]
        
        if roi.size > 0:
            display_frame = frame.copy()
            
            processed_roi = extract_hand(roi)
            
            roi_resized = cv2.resize(processed_roi, IMG_SIZE)
            
            features = extract_hog_features(roi_resized)
            features = features.reshape(1, -1)
            
            pred_idx = model.predict(features)[0]
            pred_label = label_encoder.inverse_transform([pred_idx])[0]
            
            prediction_history.append(pred_label)
            if len(prediction_history) > max_history:
                prediction_history.pop(0)
            
            if prediction_history:
                from collections import Counter
                most_common = Counter(prediction_history).most_common(1)[0][0]
                
                text = f"Prediction: {most_common}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                text_x = (frame_width - text_size[0]) // 2
                cv2.putText(display_frame, text, (text_x, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
            
            if show_processed:
                display_frame[roi_top:roi_bottom, roi_left:roi_right] = processed_roi
                
                skin_mask = detect_skin(roi)
                mask_display = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)
                mask_h, mask_w = skin_mask.shape[:2]
                display_frame[20:20+mask_h//3, 20:20+mask_w//3] = cv2.resize(mask_display, (mask_w//3, mask_h//3))
        
        cv2.putText(display_frame, fps_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        mode_text = "Mode: Processed" if show_processed else "Mode: Original"
        cv2.putText(display_frame, mode_text, (frame_width - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(display_frame, "Press 'p' to toggle view", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(display_frame, "Press 'q' to quit", (10, frame_height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Sign Language Recognition', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            show_processed = not show_processed
            print(f"Switched to {'processed' if show_processed else 'original'} view")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam recognition stopped.")


def interactive_menu():
    print("\n=== Sign Language Recognition System ===")
    print("1. Train new model")
    print("2. Load existing model")
    print("3. Open webcam recognition (press 'c')")
    print("4. Exit")
    
    choice = input("Enter your choice (1-4): ")
    
    return choice

def main():
    print("Starting sign language recognition pipeline...")
    
    model_exists = os.path.exists(MODEL_PATH)
    model = None
    label_encoder = None
    
    while True:
        choice = interactive_menu()
        
        if choice == '1':
            print("\nTraining new model...")
            
            X, y, class_names = load_and_preprocess_data(DATASET_PATH, IMG_SIZE)
            
            train_scores, val_scores = plot_learning_curve(X, y, 'learning_curve.png')
            
            model, label_encoder, X_test, y_test, y_pred = train_model(X, y)
            
            plot_confusion_matrix(y_test, y_pred, label_encoder.classes_, 'confusion_matrix.png')
            
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
            
            visualize_predictions(model, label_encoder, DATASET_PATH, class_names)
            
        elif choice == '2':
            if not os.path.exists(MODEL_PATH):
                print(f"\nModel file {MODEL_PATH} not found. Please train a model first.")
                continue
                
            print("\nLoading existing model...")
            model_data = load(MODEL_PATH)
            model = model_data['model']
            label_encoder = model_data['label_encoder']
            print("Model loaded successfully!")
            
        elif choice == '3':
            print("\nPress 'c' to start webcam recognition, or 'q' to quit.")
            
            if model is None:
                if not os.path.exists(MODEL_PATH):
                    print(f"Model file {MODEL_PATH} not found. Please train or load a model first.")
                    continue
                    
                print("Loading model...")
                model_data = load(MODEL_PATH)
                model = model_data['model']
                label_encoder = model_data['label_encoder']
            
            blank = np.ones((300, 500, 3), dtype=np.uint8) * 200  # Light gray background
            cv2.putText(blank, "Press 'c' to start webcam", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(blank, "Press 'q' to quit", (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.imshow('Sign Language Recognition', blank)
            
            webcam_active = False
            cap = None
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    if cap is not None:
                        cap.release()
                    cv2.destroyAllWindows()
                    break
                    
                elif key == ord('c'):
                    if webcam_active:
                        if cap is not None:
                            cap.release()
                        webcam_active = False
                        
                        blank = np.ones((300, 500, 3), dtype=np.uint8) * 200
                        cv2.putText(blank, "Press 'c' to start webcam", (50, 100), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                        cv2.putText(blank, "Press 'q' to quit", (50, 150), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                        cv2.imshow('Sign Language Recognition', blank)
                    else:
                        cap = cv2.VideoCapture(0)
                        if not cap.isOpened():
                            print("Error: Could not open webcam.")
                            break
                        webcam_active = True
                
                if webcam_active and cap is not None:
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Failed to capture image")
                        webcam_active = False
                        continue
                    
                    frame_height, frame_width = frame.shape[:2]
                    
                    margin = 80
                    roi_top = margin
                    roi_bottom = frame_height - 20
                    roi_left = 20
                    roi_right = frame_width - 20
                    
                    cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 3)
                    
                    roi = frame[roi_top:roi_bottom, roi_left:roi_right]
                    
                    if roi.size > 0:
                        processed_roi = extract_hand(roi)
                        
                        display_frame = frame.copy()
                        display_frame[roi_top:roi_bottom, roi_left:roi_right] = processed_roi
                        
                        roi_resized = cv2.resize(processed_roi, IMG_SIZE)
                        
                        features = extract_hog_features(roi_resized)
                        features = features.reshape(1, -1)
                        
                        pred_idx = model.predict(features)[0]
                        pred_label = label_encoder.inverse_transform([pred_idx])[0]
                        
                        text = f"Prediction: {pred_label}"
                        font_scale = 1.5
                        font_thickness = 3
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                        text_x = (frame_width - text_size[0]) // 2
                        cv2.putText(display_frame, text, (text_x, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
                    
                        cv2.putText(display_frame, "Press 'c' to stop webcam", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(display_frame, "Press 'q' to quit", (10, frame_height - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        cv2.imshow('Sign Language Recognition', display_frame)
                    else:
                        cv2.putText(frame, "Press 'c' to stop webcam", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, "Press 'q' to quit", (10, frame_height - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        cv2.imshow('Sign Language Recognition', frame)
            
        elif choice == '4':
         
            print("\nExiting program. Goodbye!")
            break
            
        else:
            print("\nInvalid choice. Please try again.")
    
    print("\nSign language recognition pipeline completed successfully!")


if __name__ == "__main__":
    main()
