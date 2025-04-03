# Literature Review Structure for Alzheimer's Disease Detection Using Deep Learning

## 1. Alzheimer's Disease and Neuroimaging

### 1.1 Pathophysiology with emphasis on structural changes

- **Key points to cover:**
  - Progression of pathological changes in the brain (amyloid plaques, neurofibrillary tangles)
  - Timeline of structural changes vs. cognitive symptoms
  - Regional patterns of neurodegeneration
- **Suggested references:**
  - Teipel et al. (2013) covers relevance of MRI for early detection
  - **Gap:** Consider adding references on pathophysiological mechanisms (e.g., Jack et al. progression model)

### 1.2 Medical background of AD with focus on hippocampal atrophy

- **Key points to cover:**
  - Hippocampal volume reduction as earliest and most reliable imaging biomarker
  - Quantification methods and established thresholds
  - Correlation with cognitive decline and disease progression
- **Suggested references:**
  - Teipel et al. (2013) discusses hippocampal changes
  - Cuingnet et al. (2011) compares methods using hippocampal measurements
  - **Gap:** Add specialized references on hippocampal volumetry methods

### 1.3 Other neuroimaging markers

- **Key points to cover:**
  - Ventricular enlargement patterns and significance
  - Cortical thinning in temporal, parietal, and frontal regions
  - White matter changes and connectivity disruptions
- **Suggested references:**
  - Teipel et al. (2013) addresses multiple imaging markers
  - **Gap:** Need references on cortical thickness measurements and ventricular changes

### 1.4 Limitations of visual assessment by radiologists

- **Key points to cover:**
  - Inter-reader variability in subjective assessment
  - Challenges in detecting subtle early changes
  - Time constraints in clinical practice
- **Suggested references:**
  - **Gap:** Need references on radiologist performance and limitations in AD detection

### 1.5 Current clinical diagnostic practices and limitations

- **Key points to cover:**
  - Multi-modal diagnostic approach (clinical, cognitive, biomarkers)
  - Diagnostic accuracy in clinical settings
  - Delay between pathological changes and clinical diagnosis
- **Suggested references:**
  - Cuingnet et al. (2011) mentions clinical diagnostic challenges
  - **Gap:** Add references on clinical diagnostic criteria (e.g., NIA-AA guidelines)

### 1.6 Role of structural MRI in diagnosis

- **Key points to cover:**
  - Position in diagnostic algorithm
  - Complementary role to clinical assessment and other biomarkers
  - Accessibility compared to PET and CSF markers
- **Suggested references:**
  - Teipel et al. (2013) covers this topic comprehensively
  - **Gap:** Consider adding references on MRI in diagnostic guidelines

### 1.7 Advantages of T1-weighted imaging for AD detection

- **Key points to cover:**
  - Optimal tissue contrast for detecting atrophy
  - Standardized acquisition protocols (MPRAGE)
  - Wider availability compared to specialized sequences
- **Suggested references:**
  - Herrera et al. (2013) discusses MRI techniques for classification
  - **Gap:** Add technical references on T1-weighted sequence advantages

## 2. Deep Learning for Medical Image Analysis

### 2.1 Evolution from traditional ML to deep learning in medical imaging

- **Key points to cover:**
  - Historical progression from handcrafted features to learned representations
  - Key milestones in deep learning for medical imaging
  - Performance comparisons between traditional and deep approaches
- **Suggested references:**
  - Litjens et al. (2017) provides a survey on deep learning in medical imaging
  - Cuingnet et al. (2011) represents earlier machine learning approaches
  - **Gap:** Add references on traditional ML methods specifically for AD

### 2.2 2D vs. 3D approaches for volumetric data

- **Key points to cover:**
  - Trade-offs between slice-based and volumetric analysis
  - Memory and computational considerations
  - Information loss in 2D approaches vs. implementation complexity in 3D
- **Suggested references:**
  - Yang et al. (2021) discusses dimensionality considerations
  - Payan and Montana (2015) presents early 3D CNN work
  - Sarraf and Tofighi (2016) uses 2D approaches
  - Liang et al. (2021) uses 2D CNNs

### 2.3 Transfer learning in medical imaging context

- **Key points to cover:**
  - Domain shift challenges between natural and medical images
  - Pre-training strategies (natural images vs. medical domain)
  - Layer freezing approaches and fine-tuning strategies
- **Suggested references:**
  - Hon and Khan (2017), Ebrahimi-Ghahnavieh et al. (2019), Acharya et al. (2021) all address transfer learning for AD
  - Maqsood et al. (2019) and Wu et al. (2022) focus on 3D transfer learning
  - Mehmood et al. (2021) discusses early diagnosis with transfer learning
  - Francis et al. (2025) covers attention mechanisms with transfer learning

### 2.4 Challenges in deep learning for medical imaging

- **Key points to cover:**
  - Data scarcity and class imbalance
  - Interpretability requirements in clinical context
  - Validation challenges and overfitting risks
- **Suggested references:**
  - Litjens et al. (2017) discusses these challenges broadly
  - **Gap:** Add references specifically addressing challenges in neuroimaging

## 3. 3D Deep Learning Architectures

### 3.1 3D CNN architectures (ResNet and variants)

- **Key points to cover:**
  - Residual learning principles applied to 3D
  - Architecture details and implementation considerations
  - Resource requirements and optimization approaches
- **Suggested references:**
  - Payan and Montana (2015) for early 3D CNN implementations
  - Wu et al. (2022) for 3D transfer learning networks
  - **Gap:** Add references specifically on 3D ResNet implementations

### 3.2 Vision transformers for volumetric data

- **Key points to cover:**
  - Adaptation of transformer architectures to 3D medical data
  - Self-attention mechanisms for capturing long-range dependencies
  - Performance comparisons with CNN-based approaches
- **Suggested references:**
  - Lu et al. (2025) discusses efficient vision transformers for AD
  - Yan et al. (2025) proposes a hybrid ResNet-Vision Transformer approach
  - Mubonanyikuzo et al. (2025) provides a systematic review of vision transformers for AD

### 3.3 Video classification models and adaptation to medical data

- **Key points to cover:**
  - Parallels between video sequences and volumetric medical data
  - Temporal vs. spatial dimension modeling
  - Transfer learning strategies from video domains
- **Suggested references:**
  - **Gap:** Need references on video classification models adapted to medical imaging
  - **Gap:** Add papers on temporal/spatial modeling strategies

### 3.4 Performance comparisons from existing literature

- **Key points to cover:**
  - Benchmark results across architecture types
  - Standardized datasets and evaluation metrics
  - Trade-offs between performance and computational efficiency
- **Suggested references:**
  - Cuingnet et al. (2011) provides early benchmarking
  - More recent comparisons from Basaia et al. (2019)
  - **Gap:** Need updated comparative studies from 2020 onwards

## 4. MRI Preprocessing for Deep Learning

### 4.1 Skull stripping methodologies

- **Key points to cover:**
  - Comparison of traditional vs. learning-based approaches
  - Impact on downstream classification performance
  - Robustness considerations for pathological brains
- **Suggested references:**
  - **Gap:** Need specific references on skull stripping methods and their impact

### 4.2 Registration and normalization approaches

- **Key points to cover:**
  - Standard space registration (MNI152, etc.)
  - Impact of registration accuracy on classification
  - Trade-offs between native space and standardized space
- **Suggested references:**
  - **Gap:** Need references on registration methods and their effects on classification

### 4.3 Impact of preprocessing on model performance

- **Key points to cover:**
  - Empirical studies measuring preprocessing effects
  - Sensitivity analysis of different preprocessing steps
  - Preprocessing as a form of data augmentation
- **Suggested references:**
  - **Gap:** Need studies specifically measuring preprocessing effects on deep learning performance

### 4.4 Current best practices

- **Key points to cover:**
  - Consensus approaches in neuroimaging literature
  - Standardized pipelines (e.g., FreeSurfer, FSL)
  - Preprocessing considerations specific to deep learning
- **Suggested references:**
  - **Gap:** Need references establishing best practices for preprocessing in neuroimaging AI

## 5. Current State of the Art

### 5.1 Recent advances in automated AD detection

- **Key points to cover:**
  - Leading approaches from recent literature (2020-2025)
  - Performance breakthroughs and methodological innovations
  - Multi-modal integration approaches
- **Suggested references:**
  - Francis et al. (2025), Lu et al. (2025), and Yan et al. (2025) represent recent approaches
  - Mubonanyikuzo et al. (2025) provides a meta-analysis
  - Saikia and Kalita (2024) and Menagadevi et al. (2024) offer reviews

### 5.2 Performance benchmarks and limitations

- **Key points to cover:**
  - Current performance ceilings
  - Challenges in cross-dataset generalization
  - Limitations in clinical translation despite high reported accuracy
- **Suggested references:**
  - Basaia et al. (2019) for established benchmarks
  - Pradhan et al. (2024) for recent analysis
  - **Gap:** Need references discussing limitations in clinical translation

### 5.3 Gap addressed by this research

- **Key points to cover:**
  - Identified limitations in current approaches
  - Novelty of applying video model transfer learning
  - Expected advantages of proposed approach
- **Suggested references:**
  - This section should reference gaps identified throughout your review
  - Connect to your specific research questions and hypotheses

## Bibliography Gaps to Address:

1. **Pathophysiology**: Need references on AD pathophysiology and progression models
2. **Radiologist performance**: Add studies on human expert performance in MRI assessment
3. **Preprocessing**: Need references on skull stripping and registration methods specific to AD
4. **Video models**: Add papers on video classification models and their potential for medical volume analysis
5. **Clinical translation**: Include references discussing the gap between research results and clinical implementation
6. **Standardized pipelines**: Add references on established preprocessing pipelines for neuroimaging
7. **Ethical considerations**: Consider adding references on ethical aspects of AI in dementia diagnosis

This structure provides a comprehensive framework for your literature review, highlighting key areas to cover while identifying specific papers from your bibliography that address each topic. The identified gaps will help guide your additional literature search to ensure complete coverage of the field.
