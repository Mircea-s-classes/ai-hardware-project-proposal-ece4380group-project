import numpy as np
from PIL import Image
import tensorflow as tf

def classify_hand_gesture(model_path, image_path):
    """
    Load a TFLite model and classify a hand gesture from an image.
    
    Args:
        model_path: Path to the .tflite model file
        image_path: Path to the input image
    
    Returns:
        prediction: The classification result
    """
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get input shape
    input_shape = input_details[0]['shape']
    height, width = input_shape[1], input_shape[2]
    
    print(f"Model expects input shape: {input_shape}")
    print(f"Model input dtype: {input_details[0]['dtype']}")
    
    # Load and preprocess the image
    img = Image.open(image_path)
    img = img.convert('RGB')  # Ensure RGB format
    img = img.resize((width, height))
    
    # Convert to numpy array (raw pixel values 0-255)
    # The model's Rescaling layer will normalize internally
    img_array = np.array(img, dtype=np.float32)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    print(f"Input array shape: {img_array.shape}")
    print(f"Input array range: [{img_array.min():.1f}, {img_array.max():.1f}]")
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Get the predicted class (index with highest probability)
    predicted_class = np.argmax(output_data[0])
    confidence = output_data[0][predicted_class]
    
    # Define gesture labels - MUST match your training directory names in alphabetical order
    # TensorFlow loads classes alphabetically from directory names
    gesture_labels = ['fist', 'like', 'no_gesture', 'palm', 'point']
    
    print(f"\nPredicted class index: {predicted_class}")
    if predicted_class < len(gesture_labels):
        print(f"Predicted gesture: {gesture_labels[predicted_class]}")
    print(f"Confidence: {confidence:.4f}")
    print(f"\nAll class probabilities:")
    for i, prob in enumerate(output_data[0]):
        label = gesture_labels[i] if i < len(gesture_labels) else f"Class {i}"
        print(f"  Class {i} ({label}): {prob:.6f}")
    
    # Check if all probabilities are similar (indicates a problem)
    probs = output_data[0]
    if np.std(probs) < 0.01:
        print("\n⚠️  WARNING: All probabilities are very similar!")
        print("    This suggests the model isn't learning properly or input is wrong.")
    
    # Check if one class dominates everything
    if confidence > 0.9 and predicted_class == 2:
        print("\n⚠️  WARNING: Model heavily biased toward class 2 (no_gesture)")
        print("    This often means:")
        print("    1. Class imbalance in training data")
        print("    2. Input preprocessing mismatch")
        print("    3. Model needs retraining")
    
    return predicted_class, confidence, output_data[0]


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python tester.py <model_path> <image_path>")
        print("Example: python tester.py hand_gesture.tflite data/Test/fist/fisttest_2.jpg")
        sys.exit(1)
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    try:
        predicted_class, confidence, probabilities = classify_hand_gesture(model_path, image_path)
        print(f"\nFinal result: Class {predicted_class} with {confidence*100:.2f}% confidence")
    except Exception as e:
        print(f"Error: {e}")