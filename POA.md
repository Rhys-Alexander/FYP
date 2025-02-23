# POA

Act in sprints, do a simple cycle to get v1 then go back and build again

first sprint just compare the architectures

Keep working on the building of the project and make notes as I go

dont convert to 2d

model the architecture to use 3d

### Data Acquisition and Preprocessing Pipeline Development

**Milestones:**

- Obtain the fMRI dataset and perform initial exploratory data analysis.
- Develop and test the preprocessing pipeline.

**Key Tasks:**

- **Data Acquisition:**
  - Download fMRI data (ensure you have the necessary permissions and understand the data format, e.g., NIfTI).
- **Preprocessing Steps:**
  - Load data using libraries like nibabel or nilearn.
  - Normalize voxel intensities and apply spatial registration (if necessary).
  - Handle noise reduction and skull stripping (if applicable).
- **Adapting 3D to 2D:**
  - Option 1: Extract representative 2D slices (axial, sagittal, and coronal views) from each 3D volume.
  - Option 2: Compute 2D projections (e.g., maximum intensity projections) that capture relevant features.
  - Save the processed slices in a format (PNG, JPEG, or tensors) that can be easily fed into a 2D CNN.
- **Best Practices:**
  - Use reproducible code (Jupyter notebooks or scripts).
  - Log preprocessing parameters for reproducibility.

### Model Adaptation and Fine-Tuning Setup

**Milestones:**

- Choose a pre-trained 2D CNN (e.g., ResNet, VGG, or EfficientNet) available in PyTorch.
- Adapt the model architecture to your task.

**Key Tasks:**

- **Model Adaptation:**
  - Replace the final classification layer(s) with a new layer that outputs the probability for Alzheimer’s vs. control.
  - If using 2D slices: Decide on an aggregation strategy (e.g., majority voting, averaging predictions, or a simple RNN to handle slice sequences).
  - If using multi-view inputs: Consider combining orthogonal slices into multi-channel input.
- **Fine-Tuning Strategy:**
  - Freeze initial layers and train only the new layers for a few epochs.
  - Gradually unfreeze and fine-tune more layers with a lower learning rate.
  - Use loss functions like binary cross-entropy and optimizers like Adam.
- **Best Practices:**
  - Use data augmentation techniques (rotation, scaling, flipping) tailored for medical images.
  - Employ a validation set to monitor for overfitting and adjust learning rates (e.g., using learning rate schedulers).

### Experiments, Evaluation, and Validation

**Milestones:**

- Complete model training and conduct thorough performance evaluations.
- Analyze results both quantitatively and qualitatively.

**Key Tasks:**

- **Training & Testing:**
  - Split your dataset into training, validation, and testing sets (or consider k-fold cross-validation if data is limited).
  - Monitor training progress using loss curves and accuracy metrics.
- **Evaluation Metrics:**
  - Calculate accuracy, ROC-AUC, sensitivity, specificity, and confusion matrices.
  - Visualize performance using ROC curves, precision-recall curves, and training graphs.
- **Validation:**
  - Perform statistical validation where possible (e.g., bootstrapping or cross-validation statistics).
  - Analyze misclassified cases to gain insights and possibly refine preprocessing/modeling.
- **Tools:**
  - Utilize scikit-learn for metrics and evaluation.
  - Use matplotlib or seaborn for plotting and visualizing results.

# **Plan of Action for Next Work Day (Focused on Coding the AI)**

## **1. Setup & Preprocessing Pipeline (2-3 hours)**

**Objective:** Ensure the dataset is properly preprocessed and ready for model training.

### **Tasks:**

✅ **Dataset Preparation**

- Download a subset of the **ADNI** or **OASIS-3** dataset (or use preprocessed versions).
- Load fMRI scans using `NiBabel` and `nilearn`.
- Convert 3D fMRI scans into 2D slices or projections (mean/max intensity projections, axial/coronal slices).

✅ **Preprocessing Steps**

- Normalize intensity values.
- Apply skull stripping and motion correction (if necessary).
- Implement basic **data augmentation** (flipping, rotation, intensity shifts).

✅ **Code Deliverables:**

- A script (`preprocess.py`) to load, normalize, and convert fMRI scans into a format usable for a CNN.
- A function to visualize preprocessed slices to verify correctness.

---

## **2. Model Adaptation & Implementation (3-4 hours)**

**Objective:** Adapt a pretrained **ResNet/EfficientNet** model for 2D fMRI slice classification.

### **Tasks:**

✅ **Load Pretrained Model (Transfer Learning)**

- Load **ResNet-50 or EfficientNet-B0** from `torchvision.models`.
- Modify the first convolutional layer to handle **single-channel grayscale** inputs instead of 3-channel RGB.
- Adjust the final fully connected layer to match the number of output classes (e.g., Alzheimer’s vs. Healthy).

✅ **Define Training Pipeline**

- Implement **data loaders** for training/validation/testing using `torch.utils.data.Dataset`.
- Define **loss function** (`CrossEntropyLoss` or `FocalLoss` for imbalanced data).
- Use **Adam optimizer** with learning rate scheduling.

✅ **Code Deliverables:**

- `model.py`: Defines the adapted ResNet/EfficientNet model.
- `train.py`: Implements the training pipeline (data loading, training loop, validation).
- `dataloader.py`: Loads preprocessed data into PyTorch format.

---

## **3. Initial Training & Debugging (1-2 hours)**

**Objective:** Run a small-scale training experiment to ensure everything works.

### **Tasks:**

✅ Train for **one epoch** on a small dataset.  
✅ Debug potential issues (data format mismatches, loss not decreasing, model architecture errors).  
✅ Save and visualize early training results (loss curve, sample predictions).

✅ **Code Deliverables:**

- Log results (loss values, accuracy).
- Save the trained model checkpoint (`model_checkpoint.pth`).

---

## **4. Prepare for Supervisor Meeting (30 min - 1 hour)**

**Objective:** Summarize progress and outline next steps.

### **Tasks:**

✅ Prepare a short **Jupyter Notebook or Markdown summary**:

- **What was done?** Preprocessing, model setup, early training.
- **Challenges faced?** Data format issues, computational constraints, etc.
- **Next steps?** Train full model, evaluate results, implement Grad-CAM for interpretability.

---

## **Next Steps (After This Work Day)**

🔹 Scale up training on the full dataset.  
🔹 Implement model evaluation (confusion matrix, AUC-ROC, Grad-CAM visualization).  
🔹 Tune hyperparameters for better performance.

This plan ensures solid coding progress while giving you tangible results to discuss with your supervisor. 🚀
