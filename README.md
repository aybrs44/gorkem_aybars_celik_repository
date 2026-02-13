Görkem, ilk GitHub projen için profesyonel bir README.md içeriği hazırladım. Doğruluk oranını belirttiğin gibi %90 olarak ekledim. GitHub sayfasında README.md dosyasını düzenle kısmına gelip şu metni yapıştırabilirsin:

SVM-Based Face Detection with Fourier Transform (FFT)
This project focuses on distinguishing between real human faces and AI-generated (GAN) faces using Support Vector Machines (SVM) and Fast Fourier Transform (FFT) techniques within the MATLAB environment.

🛠 Technical Overview
Feature Extraction: Images are analyzed in the frequency domain using FFT to detect structural artifacts typically found in synthetic images.

Classification: A Support Vector Machine (SVM) model is trained on these frequency features to classify images as "Real" or "Fake".

Dataset: Synthetic faces are gathered from thispersondoesnotexist.com using a custom Python automation script.

📊 Performance
Accuracy: The model achieved an approximately 90% accuracy rate in detecting AI-generated faces during testing.

📂 Repository Structure
face_detection_svm.m: The primary MATLAB script for SVM training and image classification.

image_downloader.py: A Python script designed to automatically fetch test images from thispersondoesnotexist.com.

README.md: Project documentation and technical summary.

🚀 How to Run
Run the image_downloader.py script to generate the required dataset.

Execute the .m file in MATLAB to train the model and evaluate the results.
