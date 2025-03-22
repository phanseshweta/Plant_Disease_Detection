 **Plant Disease Detection**
This project uses machine learning to detect diseases in tomato plants from images. It is built using TensorFlow/Keras and includes a Convolutional Neural Network (CNN) model trained on the Tomato Plant Disease Dataset. The goal of this project is to help farmers and gardeners quickly identify diseases in tomato plants, enabling timely intervention and reducing crop losses.
--------------------------------------------------------------------------------------------
Output Example

Here’s an example of the model’s predictions on a tomato plant image:
![Screenshot 2025-03-22 114347](https://github.com/user-attachments/assets/9614fb1e-6a9e-4e2d-9a75-2be25da689c1)**

--------------------------------------------------------------------------------------------
**Key Features**
Disease Detection: Accurately identifies various tomato plant diseases from images.

CNN Model: Uses a deep learning model based on Convolutional Neural Networks (CNNs) for high accuracy.

User-Friendly: Simple scripts for training, evaluation, and prediction.

Scalable: Can be extended to detect diseases in other plants or crops.
--------------------------------------------------------------------------------------------

** How It Works**
Input: The user provides an image of a tomato plant.

Preprocessing: The image is resized and normalized to match the input requirements of the CNN model.

Prediction: The trained CNN model analyzes the image and predicts the disease (or identifies the plant as healthy).

Output: The model returns the predicted disease along with confidence scores.
--------------------------------------------------------------------------------------------

**Technical Details**
Model Architecture
The CNN model consists of the following layers:

Input Layer: Accepts images of size (128, 128, 3).

Conv2D Layers: Multiple convolutional layers with Batch Normalization and MaxPooling for feature extraction.

Global Average Pooling Layer: Reduces spatial dimensions.

Dense Layers: Fully connected layers with Dropout to prevent overfitting.

Output Layer: Softmax activation for multi-class classification.

Dataset
The model is trained on the Tomato Plant Disease Dataset, which contains images of tomato plants with the following diseases:

Bacterial spot

Early blight

Late blight

Leaf Mold

Septoria leaf spot

Spider mites (Two-spotted spider mite)

Target Spot

Tomato mosaic virus

Tomato Yellow Leaf Curl Virus

Healthy plants
----------------------------------------------------------------------------------------
Performance
The model achieves the following performance metrics:

Training Accuracy: 98.15%

Training Loss: 0.117

Validation Accuracy: 96.20%

Validation Loss: 0.166
----------------------------------------------------------------------------------------
How to Use
1. Clone the Repository
Clone the repository to your local machine:

bash
git clone https://github.com/your-username/tomato-disease-detection.git
cd tomato-disease-detection
2. Install Dependencies
Install the required Python libraries:

bash
pip install -r requirements.txt
3. Prepare the Dataset
Organize your dataset into the following structure:


dataset/
├── train/                # Training images
│   ├── Bacterial_spot/   # Images of plants with Bacterial spot
│   ├── Early_blight/     # Images of plants with Early blight
│   ├── healthy/          # Images of healthy plants
│   └── ...               # Other disease classes
├── val/                  # Validation images
│   ├── Bacterial_spot/   # Images of plants with Bacterial spot
│   ├── Early_blight/     # Images of plants with Early blight
│   ├── healthy/          # Images of healthy plants
│   └── ...               # Other disease classes
4. Train the Model
Train the CNN model using the Jupyter Notebook:

Open the train.ipynb file in Jupyter Notebook.

Run all the cells to preprocess the data, train the model, and save the trained model as bestModel.keras.

5. Run the Flask App
Deploy the model using the Flask app:

Start the Flask app:

bash
python app.py
Open your browser and navigate to http://127.0.0.1:5000/.

Upload an image of a tomato plant using the web interface.

The app will display the predicted disease.

-------------------------------------------------------------------------------------------
📂 Project Structure

tomato-disease-detection/
├── ipynb_checkpoints/        # Jupyter Notebook checkpoints (auto-generated)
├── __pycache__/              # Python compiled files (auto-generated)
├── dataset/                  # Dataset directory
│   ├── train/                # Training images
│   └── val/                  # Validation images
├── static/                   # Static files for the Flask app (CSS, JS, images)
├── templates/                # HTML templates for the Flask app
├── app.py                    # Flask app for deployment (Python Source File)
├── bestModel.keras           # Trained Keras model file
├── README.md                 # Project documentation (Markdown Source File)
├── test_image.jpg            # Sample image for testing (JPG File)
└── train.ipynb               # Jupyter Notebook for training the model
-------------------------------------------------------------------------------------

 Acknowledgments
Thanks to TensorFlow for providing the machine learning framework.

Dataset sourced from PlantVillage.
--------------------------------------------------------------------------------------


