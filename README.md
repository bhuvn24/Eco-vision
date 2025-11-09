# Waste Management Using CNN Model

## Project Overview
This project focuses on building a Convolutional Neural Network (CNN) model to classify waste into categories such as "Organic" and "Recyclable." The goal is to automate waste classification processes, making them more efficient and accurate. By leveraging deep learning frameworks such as TensorFlow and Keras, this project showcases the potential of artificial intelligence in solving environmental challenges.

## Key Features
- Automated classification of waste images using a CNN.
- Data preprocessing and augmentation for enhanced model performance.
- Visualization of data distribution and model performance metrics.
- Real-world applications for sustainable waste management practices.

## Technical Details

### 1. Data Preparation
- Dataset: Images are organized into `TRAIN` and `TEST` directories.
- Preprocessing:
  - Images are resized and normalized to improve training efficiency.
  - OpenCV (`cv2`) is used for image reading and transformation.
- Augmentation: Techniques such as rotation, flipping, and scaling are applied to improve generalization.

### 2. Model Architecture
The CNN is implemented using TensorFlow/Keras and includes the following layers:
- Convolutional Layers: Extract spatial features from input images using filters.
- MaxPooling Layers: Reduce the spatial dimensions of feature maps.
- Batch Normalization: Normalize activations to stabilize and speed up training.
- Dropout Layers: Prevent overfitting by randomly dropping nodes during training.
- Dense Layers: Fully connected layers for final classification.

### 3. Training Details
- Loss Function: Categorical Crossentropy to minimize classification errors.
- Optimizer: Adam optimizer for adaptive learning rate adjustments.
- Metrics: Accuracy and precision-recall metrics to evaluate performance.
- Visualization: Training progress and performance metrics are plotted using Matplotlib.

### 4. Results
- Achieved high accuracy in classifying waste into "Organic" and "Recyclable" categories.
- Insights into the dataset’s composition were visualized using pie charts.

## Weekly Progress

### Week 1 Progress
- Dataset Preparation:
  - Dataset Link : https://www.kaggle.com/datasets/techsash/waste-classification-data
  - The dataset was organized into `TRAIN` and `TEST` directories.
  - Images were read and converted to RGB format using OpenCV.
- Model Architecture:
  - Implemented a CNN model using TensorFlow and Keras.
  - Used layers like `Conv2D`, `MaxPooling2D`, `BatchNormalization`, `Dropout`, `Flatten`, and `Dense`.
- Basic Visualization:
  - Loaded and displayed sample images from the dataset.

### Week 2 Progress
- Data Augmentation:
  - Applied transformations using `ImageDataGenerator` to enhance model performance.
- Model Compilation and Training:
  - Defined loss function and optimizer.
  - Trained the model on the dataset.
- Evaluation:
  - Assessed model accuracy and loss.
  - Visualized training progress.

## Applications
- Automating waste segregation to reduce manual effort.
- Improving recycling efficiency through accurate classification.
- Promoting sustainable practices in waste management systems.

## Installation and Usage

### Prerequisites
- Python 3.x
- TensorFlow
- OpenCV
- Matplotlib
- NumPy
- Pandas

### Installation
1. Clone this repository:
   ```
   git clone https://github.com/your-repo/waste-management-cnn.git
   ```
2. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```

### Usage
1. Prepare the dataset by organizing images into `TRAIN` and `TEST` directories.
2. Run the notebook or script to preprocess the data and train the model.
   ```
   python train_model.py
   ```
3. Evaluate the model and visualize results.
   ```
   python evaluate_model.py
   ```
### Streamlit Application
  ![WhatsApp Image 2025-02-15 at 19 07 58_13a26f70](https://github.com/user-attachments/assets/b5fa6314-bcef-4784-8039-51a57f4138a8)
  ![WhatsApp Image 2025-02-15 at 19 08 20_58f18486](https://github.com/user-attachments/assets/9317e6ba-2585-4592-8679-776f8f33708f)
  ![WhatsApp Image 2025-02-15 at 19 10 15_08c67a73](https://github.com/user-attachments/assets/fb524bcf-369e-4ace-afcc-058243ff2e4a)




## Future Scope
- Extend the classification categories to include more waste types.
- Deploy the model as a web application or mobile app for broader accessibility.
- Integrate IoT devices for real-time waste classification.

## Acknowledgments
- TensorFlow and Keras documentation.
- OpenCV community for image processing resources.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

