YOLOv8 Custom Object Detection Model
This repository contains a YOLOv8 object detection model trained on a custom dataset to identify the person class. The model was trained using the Ultralytics framework and the results are summarized below.

Training Progress
The model was trained for a number of epochs, and the training and validation loss curves, as well as key performance metrics, are shown in the results plot. The curves demonstrate good convergence, with both the bounding box loss and classification loss decreasing steadily over the training period.

!(results.png)

Performance Metrics
Confusion Matrix
The confusion matrix provides a clear breakdown of the model's performance. The normalized matrix shows that the model correctly identified 90 of the person class (True Positives) and 100 of the background class (True Negatives), with only a 10 false negative rate for the person class.

!(confusion_matrix_normalized.png)

Precision-Recall Curve
The model's Precision-Recall curve indicates high performance, achieving a mean Average Precision (mAP@0.5) of 0.921.

!(BoxPR_curve.png)

F1 Score
The F1-Confidence curve helps in selecting an optimal confidence threshold for the model. The peak F1 score of 0.88 is achieved at a confidence threshold of approximately 0.322. This value provides a strong balance between precision and recall for the model.

!(BoxF1_curve.png)

Precision and Recall Curves
The precision-confidence and recall-confidence curves further illustrate the model's performance characteristics. As the confidence threshold increases, the precision remains high while the recall drops, which is a typical trade-off for object detection models.

Precision-Confidence Curve: The model's precision remains high across the confidence range.
!(BoxP_curve.png)

Recall-Confidence Curve: The recall for the model decreases as the confidence threshold increases.
!(BoxR_curve.png)

Usage
To use this model, you will need to have the Ultralytics library installed. You can then load the trained weights and perform inference on new images or video streams.

from ultralytics import YOLO

# Load a custom trained model
model = YOLO('path/to/your/best.pt')

# Predict on an image
results = model('path/to/your/image.jpg')
