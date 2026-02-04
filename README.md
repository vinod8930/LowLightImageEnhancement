# 🌙 Low-Light Image Enhancement using Deep Learning

This project implements a **CNN-based low-light image enhancement system** that improves the visibility and structural quality of images captured under poor lighting conditions.  
The model is trained using paired low- and normal-light images and deployed through an interactive **Streamlit web application** for real-time inference.

---

## 📌 Problem Statement
Images captured in low-light environments often suffer from poor visibility, low contrast, and noise, which negatively affect both human perception and downstream computer vision tasks. Traditional image enhancement techniques such as histogram equalization often fail to preserve structural details. This project addresses the problem using a **deep learning–based approach**.

---

## 🧠 Proposed Solution
- A **Convolutional Neural Network (CNN)** is trained to learn a direct mapping from low-light images to normal-light images.
- The model is trained in a **supervised manner** using paired images from the LOL dataset.
- The trained model is evaluated using **PSNR** and **SSIM** metrics and compared against traditional enhancement techniques.
- A **Streamlit web interface** is developed for easy testing and demonstration.

---

## 📂 Dataset
- **LOL Dataset (our485)**
  - 485 paired low-light and normal-light images
  - Used for supervised training and evaluation

Dataset structure:
our485/
├── low/ # low-light images
└── high/ # corresponding normal-light images

yaml
Copy code

---

## ⚙️ Model Architecture
- Convolutional layers with ReLU activation
- Batch Normalization
- Residual connection for stable learning
- Sigmoid activation in the output layer
- Loss function: **Mean Squared Error (MSE)**

---

## 📊 Evaluation Metrics
The model is evaluated quantitatively using:

- **PSNR (Peak Signal-to-Noise Ratio)**
- **SSIM (Structural Similarity Index)**

### Performance Comparison

| Method | PSNR (dB) | SSIM |
|------|----------|------|
| CLAHE | 7.57 | 0.28 |
| CNN (Proposed) | **17.02** | **0.75** |

The results show that the proposed CNN significantly outperforms traditional enhancement methods.

---

## 🌐 Web Application (Streamlit)
The project includes a Streamlit-based web interface that allows users to:

- Select images from the dataset
- Upload custom low-light images
- View original and enhanced images
- Save enhanced outputs automatically


<img width="1914" height="915" alt="image" src="https://github.com/user-attachments/assets/205c575e-f720-4792-8264-078ffcfba7c0" />

<img width="1883" height="748" alt="image" src="https://github.com/user-attachments/assets/09960fc3-7cff-45cb-90a3-7346725c14d6" />



---

## 🛠️ Project Structure
LowLightImageEnhancement/
├── train.py # Model training
├── evaluate.py # PSNR & SSIM evaluation
├── compare_clahe.py # Classical vs CNN comparison
├── resave_model.py # Convert model to SavedModel format
├── app.py # Streamlit web application
├── our485/ # Dataset
├── output/ # Enhanced images
├── requirements.txt
└── README.md

yaml
Copy code

---

## 🚀 How to Run

### 1️⃣ Create virtual environment

python -m venv .venv
.\.venv\Scripts\activate

2️⃣ Install dependencies


pip install -r requirements.txt

3️⃣ Train the model

python train.py

4️⃣ Evaluate performance

python evaluate.py
python compare_clahe.py

5️⃣ Run Streamlit app

python -m streamlit run app.py

---
##📦 Requirements
arduino
tensorflow==2.15.0
opencv-python
numpy
scikit-image
streamlit
pillow
tqdm

---
##🔍 Limitations
Performance degrades for extremely dark images with minimal visible information

Color shifts may occur in some scenarios

Model performance depends on dataset distribution

---
##🔮 Future Enhancements
U-Net or GAN-based architectures

Zero-reference enhancement (Zero-DCE)

Real-time webcam enhancement

Mobile or cloud deployment

---
##🎓 Academic Relevance
This project demonstrates:

Deep learning–based image-to-image translation

Quantitative evaluation of enhancement techniques

End-to-end system design and deployment

Suitable for Senior Design / Final Year Project submission.

---
##📎 Author

Vinod Penkey
GitHub: https://github.com/vinod8930
