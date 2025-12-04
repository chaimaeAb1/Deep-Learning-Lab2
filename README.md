# Deep Learning Lab 2 – CNN, Faster R-CNN, Transfer Learning & Vision Transformer (MNIST)

## Objective
The goal of this lab is to explore different deep learning architectures for image classification using the MNIST dataset.  
We implemented:

- A custom **Convolutional Neural Network (CNN)**
- A **Faster R-CNN** detection model adapted to classification
- Two **Transfer Learning models** (VGG16 & AlexNet)
- A **Vision Transformer (ViT)** implemented from scratch

The full pipeline includes training, validation, testing, comparison of models using accuracy, F1-score, confusion matrix, and training time.

---

#  Dataset  
We used the MNIST dataset provided on Kaggle:  
https://www.kaggle.com/datasets/hojjatk/mnist-dataset

Loaded via a custom IDX parser (60,000 train images and 10,000 test images).

---

#  PART 1 — CNN Classifier

###  Custom CNN Architecture
The CNN consists of:
- 3 convolutional blocks (Conv → ReLU → BatchNorm → MaxPool)
- Adaptive Average Pooling
- A fully connected classifier with dropout

**Performance:**
- **Test Accuracy:** 0.9923  
- **Test F1-score:** 0.9923  
- **Training Time:** 53 seconds  
- **Conclusion:** Best overall model for MNIST.

###  Confusion Matrix  
Very few errors, near-perfect diagonal.

---

#  PART 1 — Faster R-CNN (Detection → Classification)

A Faster R-CNN MobileNetV3 model was adapted for MNIST:

- Artificial bounding box around each digit
- Dataset converted to detection format (COCO-style)
- Fine-tuned for 2 epochs

**Performance:**
- **Accuracy:** 0.275  
- **Training Time:** 4.86 seconds  
- **Conclusion:**  
  Faster R-CNN is designed for object detection, *not* classification.  
  Poor performance is expected and demonstrates the limitations of RCNN for MNIST.

---

#  PART 1 — Transfer Learning (VGG16 & AlexNet)

We froze convolutional layers and trained only the classifier head.

###  VGG16 (Transfer Learning)
- **Test Accuracy:** 0.9678  
- **Test F1-score:** 0.9674  
- **Training Time:** 1068 seconds  
- **Conclusion:** High accuracy but extremely slow and unnecessary for MNIST.

###  AlexNet (Transfer Learning)
- **Test Accuracy:** 0.951  
- **Test F1-score:** 0.950  
- **Training Time:** 167 seconds  
- **Conclusion:** Much faster, but less accurate than CNN.

---

#  PART 2 — Vision Transformer (ViT)

Implemented from scratch following the tutorial:
https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c

Architecture:
- Patch embedding via Conv2d
- 6 Transformer encoder layers
- Multi-head self-attention
- Final linear classifier

**Performance:**
- **Test Accuracy:** 0.9727  
- **Test F1-score:** 0.9725  
- **Training Time:** 85 seconds  
- **Conclusion:**  
  ViT performs well, but CNN remains superior on MNIST due to small image size and limited dataset.

---

#  Global Comparison Table

| Model            | Test Accuracy | Test F1 | Training Time (s) | Notes |
|------------------|--------------|---------|---------------------|-------|
| **CNN**          | **0.9923**   | **0.9923** | 53  |  Best model |
| Faster R-CNN     | 0.2750       | —       | 4.8 |  Not adapted for classification |
| VGG16 (TL)       | 0.9678       | 0.9674  | 1068 | Too heavy |
| AlexNet (TL)     | 0.9510       | 0.9503  | 167 | Fast but weaker |
| ViT              | 0.9727       | 0.9725  | 85 | Strong, but CNN still better |

---

#  Final Conclusion

- CNN is the most efficient architecture for MNIST (best accuracy, fastest training).
- Faster R-CNN is not suitable for classification tasks → intentionally poor results.
- Transfer Learning models (VGG/AlexNet) do not outperform a small custom CNN.
- Vision Transformer performs well but requires more data to surpass CNN on low-resolution images.

