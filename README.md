**Plant Disease Detection**

This project leverages machine learning to detect diseases in tomato plants using image data. Built with TensorFlow/Keras, it employs a Convolutional Neural Network (CNN) trained on the Tomato Plant Disease Dataset. The aim is to assist farmers and gardeners in quickly identifying plant diseases, enabling early intervention and reducing crop losses.


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*Example Predictions*

https://github.com/user-attachments/assets/6d3b0d32-a4f4-4bbb-befd-796447505fc4


-------------------------------------------------------------------------------------------

*Key Feature*

  >Disease Detection: Accurately classifies a range of tomato plant diseases from images.

  >CNN Architecture: Employs deep learning using Convolutional Neural Networks for high 
   precision.

  >User-Friendly: Simple scripts for training, evaluation, and prediction.

  >Scalable Design: Easily extendable to support disease detection in other crops or plants.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

*How It Works*
>Input: User uploads an image of a tomato plant.

>Preprocessing: Image is resized and normalized to fit the CNN’s input format.

>Prediction: The trained model analyzes the image and predicts the disease class or 
 identifies the plant as healthy.

>Output: The prediction is displayed along with confidence scores.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

*Model Architecture*
>The CNN model includes the following layers:

>Input Layer: Accepts images of shape (128, 128, 3).

>Convolutional Layers: Feature extraction using Conv2D layers with Batch Normalization and MaxPooling.

>Global Average Pooling: Reduces spatial dimensions while preserving key features.

>Dense Layers: Fully connected layers with Dropout to prevent overfitting.

>Output Layer: Softmax activation for multi-class classification.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

*Dataset*
The model is trained on the Tomato Plant Disease Dataset, which includes the following classes:

    Bacterial Spot
    
    Early Blight
    
    Late Blight
    
    Leaf Mold
    
    Septoria Leaf Spot
    
    Spider Mites (Two-Spotted Spider Mite)
    
    Target Spot
    
    Tomato Mosaic Virus
    
    Tomato Yellow Leaf Curl Virus
    
    Healthy Plants

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

*Performance Metrics*
Training Accuracy: 98.15%

>Training Loss: 0.117

>Validation Accuracy: 96.20%

>Validation Loss: 0.166

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

*How to Use*
1. Clone the Repository
   git clone https://github.com/phanseshweta/Plant_Disease_Detection.git
   cd Plant_Disease_Detection
2. Install Dependencies
   pip install -r requirements.txt
3. Prepare the Dataset
   Structure your dataset as follows:


![image](https://github.com/user-attachments/assets/11ff6b6b-990d-4141-85fe-6779b48e3f5e)

4. Train the Model
  Open train.ipynb in Jupyter Notebook and run all cells to:
  
  Preprocess the data
  
  Train the model
  
  Save the trained model as bestModel.keras

5. Run the Flask App
  Start the web app using:
  python app.py
  Then open your browser and go to: http://127.0.0.1:5000
  
  Upload a tomato plant image through the interface to see the predicted disease.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

*Project Structure*

![image](https://github.com/user-attachments/assets/6aa63aad-ba56-478c-a3a6-1523c0134f33)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

*Acknowledgments*
Dataset sourced from PlantVillage


