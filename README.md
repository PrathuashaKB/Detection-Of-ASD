# Detection-Of-ASD
Autism Spectrum Disorder (ASD) Detection is a machine learning-based approach that analyzes images and structured data to identify ASD traits, aiding in early diagnosis and assessment.

### About :

Autism Spectrum Disorder (ASD) is a developmental disorder affecting communication and behavior. Early detection is crucial for timely intervention, which can significantly improve the quality of life for individuals with ASD. This project aims to develop a system for ASD detection using two different approaches:

1. **Image-based detection**: Utilizes deep learning techniques to analyze facial features from images to identify ASD characteristics. This approach leverages Convolutional Neural Networks (CNNs) to extract patterns that may indicate ASD. 
2. **CSV-based detection**: Uses structured data containing behavioral and demographic attributes to classify ASD cases. Machine learning models are applied to analyze key features and make predictions based on clinical data.

By comparing both methods, this project provides insights into the effectiveness of image-based and structured data-based approaches in ASD diagnosis. The goal is to explore how artificial intelligence can assist in ASD detection and potentially support healthcare professionals in their assessments.

### Methodology :

### Image-based Approach :
[CODE](https://github.com/PrathuashaKB/Detection-Of-ASD/tree/main/ASD%20Detection%201)
- Preprocessing of images (resizing, normalization, augmentation)
- Model: Convolutional Neural Networks (CNNs) trained on ASD image datasets
- Evaluation: Accuracy, precision, recall, and F1-score
  
[IMAGE DATASET](https://github.com/PrathuashaKB/Detection-Of-ASD/tree/main/ASD%20Detection%201/Autism_data) : Contains unlabeled images for ASD and non-ASD individuals.
<img src="https://github.com/PrathuashaKB/Detection-Of-ASD/blob/main/images/design1.PNG" width="100%"> 

### CSV-based Approach :
[CODE](https://github.com/PrathuashaKB/Detection-Of-ASD/tree/main/ASD%20Detection%202)
- Data preprocessing (handling missing values, encoding categorical variables, feature selection)
- Model: Machine Learning classifiers (Random Forest, SVM, etc.)
- Evaluation: Performance metrics and validation techniques
  
[CSV DATASET](https://github.com/PrathuashaKB/Detection-Of-ASD/blob/main/ASD%20Detection%202/Autism-Child-Data1.csv) : Tabular data with features relevant to ASD diagnosis.
<img src="https://github.com/PrathuashaKB/Detection-Of-ASD/blob/main/images/design2.PNG" width="100%"> 

### Technologies Used :

1. Programming Languages: Python

2. Machine Learning & Deep Learning: TensorFlow, Keras, Scikit-learn

3. Data Processing & Visualization: Pandas, NumPy, Matplotlib, Seaborn

4. Image Processing: OpenCV, PIL

5. Model Deployment (Optional): Flask, Streamlit

#### "Suggestions and project improvement are invited"

#### Prathuasha K B

