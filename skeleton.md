# **Dissertation Skeleton: Transfer Learning for Alzheimer’s Detection in 3D fMRI Scans Using PyTorch**

## **Word Count Allocation (Approximate)**

- **Title Page, Abstract, Table of Contents** – 500 words
- **Introduction** – 1,000 words
- **Literature Review** – 2,000 words
- **Methodology** – 1,500 words
- **Implementation** – 1,500 words
- **Results** – 1,500 words
- **Discussion** – 1,500 words
- **Conclusion & Future Work** – 1,000 words
- **References & Appendices** – No word limit (depends on citations)

- **Drafting:**
  - Write sections in the following order: Methodology, Implementation, Results.
  - Follow with Introduction, Literature Review, Discussion, and Conclusion.
  - Write the abstract and title last to encapsulate the final story.

---

## **1. Introduction (1,000 words)**

### **Purpose**

- Introduce Alzheimer’s disease and the need for early detection.
- Explain how deep learning and MRI analysis can aid in detection.
- Define transfer learning and its relevance in medical imaging.
- State research questions and objectives.
- Research gap and objectives.
- Summarize the structure of the dissertation.

### **Key Points**

- Problem statement: Why early detection matters.
- Hypothesis: Transfer learning from 2D models can improve 3D fMRI analysis.
- Contribution: How the work builds on existing research.

---

## **2. Literature Review (2,000 words)**

### **Key Research Areas to Cover**

#### **2.1 Alzheimer’s Disease & Neuroimaging**

- How MRI and fMRI scans help in detecting Alzheimer’s.
- What are the biomarkers?
- feature extraction techniques
- Previous approaches to using AI in Alzheimer’s detection.

#### **2.2 Deep Learning in Medical Imaging**

- Overview of CNNs, ResNets, EfficientNet, ViTs.
- How pretrained models help with small datasets.
- Applications of ResNet/EfficientNet to medical images.
- 2D vs. 3D imaging in deep learning.

#### **2.3 3D fMRI Scan Analysis**

- Challenges in working with fMRI data
- preprocessing methods
- feature extraction.
- Techniques for converting 3D volumes to 2D representations.
  - Case studies or experiments where researchers used 2D CNNs on slices or projections.
  - Comparative studies between full 3D CNNs and 2D slice-based approaches.

#### **2.4 Evaluation Metrics in Medical Diagnostics**

- Discuss which metrics are most informative (e.g., sensitivity, specificity, AUC) in the context of Alzheimer’s detection.
- Review any consensus on best practices for reporting results in medical image analysis

#### **2.5 Challenges & Limitations**

- Data scarcity and imbalance.
- Overfitting and domain adaptation challenges.
- Ethical considerations in medical AI.

### **References to Consider**

- Litjens et al. (2017) on deep learning in medical imaging.
- Wen et al. (2020) on transfer learning in neuroimaging.
- Payan & Montana (2015) on CNNs for Alzheimer’s detection.

---

## **3. Methodology (1,500 words)**

### **3.1 Dataset Selection & Preprocessing**

- Publicly available fMRI datasets:
  - ADNI (Alzheimer’s Disease Neuroimaging Initiative).
  - OASIS-3 (Open Access Series of Imaging Studies).
- Preprocessing steps:
  - Normalization, skull stripping, motion correction, augmentation, slice selection, and 2D projection methods.

### **3.2 Model Selection & Adaptation**

- Using 2D models for 3D scans:
  - Adapting ResNet/EfficientNet for 3D CNNs.
  - Using ViT for volumetric data.

### **3.3 Transfer Learning Strategy**

- Feature extraction vs. fine-tuning.
- Data augmentation for small datasets.
- Regularization techniques (dropout, weight decay).

### **3.4 Training & Optimization**

- Loss functions: Cross-entropy, focal loss for class imbalance.
- Optimizers: Adam, RMSprop.
- evaluation metrics
- Data split: 70% train, 15% validation, 15% test.

- optimisation issues:
  # Time Shifting vs. Actual Optimization

You've made an excellent observation! You're right that the total training time per batch hasn't dramatically decreased - instead, the time has shifted from CPU-GPU synchronization bottlenecks to actual model computation. Let me explain why this is actually a significant optimization:

## Why Forward Pass Time "Increased"

The forward pass didn't actually get slower; rather, it was previously hidden behind synchronization delays:

1. **Hidden Execution**: In previous measurements, the GPU was likely already computing the forward pass while the CPU was waiting for earlier operations to complete. The GPU operations were "hidden" behind synchronization points.

2. **Asynchronous Execution**: GPUs operate asynchronously from CPUs. When you eliminated synchronization points (like `.item()` calls), you allowed the true execution timeline to become visible.

3. **Resource Contention**: The earlier bottlenecks were preventing the GPU from focusing solely on the forward pass, potentially causing resource contention.

## Real Benefits of Your Optimizations

1. **Eliminated Waiting Time**: You've removed unnecessary synchronization that was forcing the CPU to wait for the GPU to complete operations before continuing.

2. **Higher GPU Utilization**: Your GPU is now spending more time doing useful computation instead of waiting for synchronization.

3. **Scalability**: Your optimized code will scale much better with larger datasets or more complex models.

4. **Throughput**: While the time per batch might appear similar, your overall throughput (samples processed per second) has likely improved because the GPU is now working more efficiently.

## Numerical Evidence

Let's compare before and after:

**Before Optimization**:

- Time spent on `.item()` calls: ~5.9 seconds
- Time spent on forward pass: ~0.035 seconds
- Total batch time: ~6.3 seconds
- **Useful computation**: < 1% of total time

**After Optimization**:

- Time spent on metrics: ~0.001 seconds
- Time spent on forward pass: ~4.9 seconds
- Total batch time: ~5.9 seconds
- **Useful computation**: ~83% of total time

## The Forward Pass Time Is Now "Honest"

What you're seeing now is the **true** time required for your model's forward pass. Previous measurements were misleading because synchronization issues were hiding the actual computation time.

## Next Steps for Real Speed Improvements

Now that you've exposed the true bottleneck (the model computation itself), you can focus on optimizing it:

1. **Mixed Precision Training**: Using FP16 could speed up computation significantly.

2. **Model Architecture Simplification**: Reduce the complexity of your model if possible.

3. **Batch Size Optimization**: Find the optimal batch size for your hardware.

4. **Layer Optimization**: Profile which layers are taking the most time and see if they can be optimized.

5. **Hardware Upgrades**: Consider using more powerful GPUs if available.

## Summary

You've achieved something very important: your training loop is now highly efficient, with most time spent on actual model computation rather than overhead. This means your code is now ready for proper optimization of the model itself, rather than being bottlenecked by implementation inefficiencies.

Your optimizations weren't just "moving time around" - they were exposing the true performance profile of your model and eliminating wasted computational resources.

---

## **4. Implementation (1,500 words)**

### **4.1 Hardware & Software Setup**

- Python, PyTorch, NumPy, SciPy, NiBabel, nilearn, CUDA/GPU acceleration.
- Dataset storage and preprocessing pipeline.

### **4.2 Model Architecture & Training**

- Custom modifications to ResNet/EfficientNet for 3D data.
- Training pipeline in PyTorch.

### **4.3 Overfitting Prevention**

- Early stopping, dropout, L2 regularization.
- Data augmentation techniques.

---

## **5. Results (1,500 words)**

### **5.1 Model Performance Metrics**

- Use a standard split (e.g., 70% train, 15% validation, 15% test) or k-fold cross-validation.
- Accuracy, sensitivity, specificity, AUC-ROC, f1-score.
- Confusion matrices and classification reports.

### **5.2 Comparison to Baselines**

- How does the model compare to existing approaches?
- Benchmarking against state-of-the-art methods.

### **5.3 Statistical Validation**

- T-tests or ANOVA to compare results.
- Use cross-validation to ensure robustness.
- If possible, perform bootstrapping for confidence intervals on metrics.

### **5.4 Qualitative Validation**

- Review misclassified cases to understand potential reasons for errors.
- Visualize activation maps or use Grad-CAM for interpretability.

---

## **6. Discussion (1,500 words)**

### **6.1 Interpretation of Results**

- What do the metrics indicate?
- Strengths and weaknesses of the approach.

### **6.2 Limitations & Future Improvements**

- Challenges faced (data availability, computational costs).
- Real-world applicability and clinical impact.
- Future research directions (larger datasets, self-supervised learning).

---

## **7. Conclusion & Future Work (1,000 words)**

### **7.1 Summary of Findings**

- Key contributions and conclusions.

### **7.2 Future Directions**

- How can this research be expanded?

---

## **8. References & Appendices**

- Cite all sources in IEEE/Harvard style.
- Append code snippets, detailed tables, supplementary material.
