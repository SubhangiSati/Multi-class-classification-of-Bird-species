# Bird Species Classifier

The Bird Species Classifier is an application built using a Convolutional Neural Network (CNN) to classify images of birds into one of 525 different species. It allows users to upload an image of a bird and receive a prediction of the bird species along with the confidence score.

### Key Features:

- **Image Upload:** Users can upload an image of a bird directly to the application.
  
- **Classification:** The application uses a trained CNN model to classify the uploaded image into one of 525 bird species.

- **Confidence Score:** Along with the predicted bird species, the application provides a confidence score indicating the certainty of the prediction.

### Model Training:

The CNN model used in this application has been trained on a dataset containing images of 525 different bird species. The model has been trained using various optimizers, including Adam, Nadam, SGD, RMSprop, and Adagrad, to optimize its performance.

### Custom Metrics:

The model's performance is evaluated using precision, recall, and F1-score metrics, which provide measures of the model's accuracy in classifying bird species.

### Usage:

1. **Upload Image:** Click on the "Choose an image..." button to upload an image of a bird.

2. **Prediction:** After uploading the image, click on the "Predict" button to classify the bird species.

3. **Result:** The application will display the predicted bird species along with the confidence score.

### Requirements:

- Python 3.x
- Streamlit
- TensorFlow
- PIL (Python Imaging Library)
- NumPy
- scikit-learn

### Running the Application:

1. Install the required dependencies mentioned above.
   
2. Clone the repository or download the provided files.

3. Run the Streamlit app script (`app.py` or any appropriate filename) using the following command:
   
   ```bash
   streamlit run app.py
   ```

4. Access the application through the provided URL in the terminal.

### Dataset: https://www.kaggle.com/datasets/gpiosenka/100-bird-species/code

The dataset used to train the model contains a diverse collection of bird images spanning 525 different species. Each image is labeled with its corresponding bird species to facilitate supervised learning.

### Model Training:

The model is trained using the following steps:

- **Data Augmentation:** Images are augmented using techniques such as rotation, shifting, shearing, zooming, and flipping to increase the diversity of the training data.
  
- **Model Architecture:** The CNN model consists of multiple convolutional layers followed by max-pooling layers to extract features from the input images.
  
- **Optimizer:** The model is compiled using various optimizers, including Adam, Nadam, SGD, RMSprop, and Adagrad, to optimize its performance.

### Evaluation:

The trained model is evaluated on a separate test dataset to assess its performance. Metrics such as accuracy, precision, recall, and F1-score are computed to measure the model's effectiveness in classifying bird species.

### Future Enhancements:

1. **Improving Model Accuracy:** Continuously train and fine-tune the model to improve its accuracy in classifying bird species.

2. **Interactive Visualization:** Enhance the user interface with interactive features such as displaying similar bird species, bird habitat information, and bird calls.

3. **Multi-Image Classification:** Extend the application to support batch processing of multiple bird images for classification.

4. **Mobile Application:** Develop a mobile version of the application for users to classify bird species on the go.

### Acknowledgments:

- The dataset used in this project is sourced from Kaggle and https://www.kaggle.com/datasets/gpiosenka/100-bird-species/code.

- Special thanks to Ayan Sar and Purvika Joshi for their contributions to the project.

### Contributors:

- Subhangi Sati

- Purvika Joshi

- Ayan Sar

