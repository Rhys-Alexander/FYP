# Literature Review Structure for Alzheimer's Disease Detection Using Deep Learning

## 1. Alzheimer's Disease and Neuroimaging

### 1.1 Pathophysiology with Emphasis on Structural Changes

- **Key points to cover:**
  - Definition, prevalence, and burden of Alzheimer's Disease
  - Progression of pathological changes (amyloid plaques, neurofibrillary tangles)
  - Timeline of structural vs. cognitive symptoms
  - Regional patterns of neurodegeneration
- **Suggested references:**
  - Teipel et al. (2013) for relevance of MRI in early detection
  - **Gap:** Add references on AD pathophysiology progression models (e.g., Jack et al. progression model)

### 1.2 Hippocampal Atrophy as Primary Biomarker

- **Key points to cover:**
  - Hippocampal volume reduction patterns and progression
  - Quantification methods and established thresholds
  - Correlation with cognitive decline and disease progression
  - Sensitivity and specificity as a diagnostic marker
- **Suggested references:**
  - Teipel et al. (2013) for hippocampal changes
  - Cuingnet et al. (2011) for hippocampal measurement methods
  - **Gap:** Add specialized references on hippocampal volumetry

### 1.3 Additional Neuroimaging Markers

- **Key points to cover:**
  - Ventricular enlargement patterns and diagnostic value
  - Cortical thinning in temporal, parietal, and frontal regions
  - White matter changes and connectivity disruptions
  - Relative sensitivity of different markers
- **Suggested references:**
  - Teipel et al. (2013) for multiple imaging markers
  - **Gap:** Add references on cortical thickness and ventricular changes

### 1.4 Current Clinical Diagnostic Practices and Limitations

- **Key points to cover:**
  - Diagnostic criteria (NINCDS-ADRDA) and workflow
  - Multi-modal diagnostic approach (clinical, cognitive, biomarkers)
  - Inter-reader variability in visual assessment
  - Challenges in early and accurate detection
  - Delay between pathological changes and clinical diagnosis
- **Suggested references:**
  - Cuingnet et al. (2011) for diagnostic challenges
  - **Gap:** Add references on clinical diagnostic guidelines (e.g., NIA-AA guidelines) and radiologist performance

### 1.5 Role of Structural MRI in Diagnosis

- **Key points to cover:**
  - Position in diagnostic algorithm and clinical workflow
  - Complementary role to clinical assessment and other biomarkers
  - Accessibility advantages compared to PET and CSF markers
  - Limitations of visual/manual assessment
- **Suggested references:**
  - Teipel et al. (2013) covers comprehensive role of MRI
  - **Gap:** Add references on MRI in current diagnostic guidelines

### 1.6 Advantages of T1-weighted Imaging for AD Detection

- **Key points to cover:**
  - Optimal tissue contrast for detecting atrophy
  - Standardized acquisition protocols (MPRAGE)
  - Wider availability compared to specialized sequences
  - Trade-offs with other imaging modalities
- **Suggested references:**
  - Herrera et al. (2013) for MRI techniques in classification
  - **Gap:** Add technical references on T1-weighted sequence advantages

## 2. Deep Learning for Medical Image Analysis

### 2.1 Evolution from Traditional ML to Deep Learning

- **Key points to cover:**
  - Historical progression from handcrafted features to learned representations
  - Traditional machine learning techniques in neuroimaging
  - Key milestones in deep learning for medical imaging
  - Performance comparisons between approaches
- **Suggested references:**
  - Litjens et al. (2017) for survey on deep learning in medical imaging
  - Cuingnet et al. (2011) for earlier machine learning approaches
  - **Gap:** Add references on traditional ML methods specifically for AD

### 2.2 2D vs. 3D Approaches for Volumetric Data

- **Key points to cover:**
  - Fundamental differences in information extraction
  - Trade-offs between slice-based and volumetric analysis
  - Memory and computational considerations
  - Information loss in 2D vs. implementation complexity in 3D
- **Suggested references:**
  - Yang et al. (2021) for dimensionality considerations
  - Payan and Montana (2015) for early 3D CNN work
  - Sarraf and Tofighi (2016) and Liang et al. (2021) for 2D approaches

### 2.3 Transfer Learning in Medical Imaging

- **Key points to cover:**
  - Definition and rationale for transfer learning
  - Domain shift challenges between natural images and medical imaging
  - Pre-training strategies (natural images vs. medical domain) and layer freezing approaches
  - Previous successes in neuroimaging applications
- **Suggested references:**
  - Hon and Khan (2017), Ebrahimi-Ghahnavieh et al. (2019), Acharya et al. (2021) for transfer learning for AD
  - Maqsood et al. (2019) and Wu et al. (2022) for 3D transfer learning
  - Mehmood et al. (2021) for early diagnosis applications
  - Francis et al. (2025) for attention mechanisms with transfer learning

### 2.4 Challenges in Deep Learning for Medical Imaging

- **Key points to cover:**
  - Data scarcity and class imbalance issues
  - Interpretability requirements in clinical context
  - Validation challenges and risks of overfitting
  - Privacy and ethical considerations
- **Suggested references:**
  - Litjens et al. (2017) for broad challenges
  - **Gap:** Add references specifically addressing neuroimaging challenges

## 3. 3D Deep Learning Architectures

### 3.1 3D CNN Architectures (ResNet and Variants)

- **Key points to cover:**
  - Core principles of 3D CNNs
  - Residual learning principles applied to 3D volumes
  - Architecture details and implementation considerations
  - Parameter efficiency and resource requirements
- **Suggested references:**
  - Payan and Montana (2015) for early implementations
  - Wu et al. (2022) for 3D transfer learning networks
  - **Gap:** Add references on 3D ResNet implementations and optimizations

### 3.2 Vision Transformers for Volumetric Data

- **Key points to cover:**
  - Adaptation of transformer architectures to 3D medical data
  - Self-attention mechanisms for spatial context
  - Performance comparisons with CNN-based approaches
  - Advantages and limitations for neuroimaging
- **Suggested references:**
  - Lu et al. (2025) for efficient vision transformers
  - Yan et al. (2025) for hybrid ResNet-ViT approach
  - Mubonanyikuzo et al. (2025) for systematic review

### 3.3 Video Classification Models and Medical Adaptation

- **Key points to cover:**
  - Parallels between video sequences and volumetric medical data
  - Key video architectures (MC13, R(2+1)D, r3d_18, etc.)
  - Temporal vs. spatial dimension modeling
  - Transfer learning strategies from video domains
- **Suggested references:**
  - **Gap:** Add references on video classification models adapted to medical imaging
  - **Gap:** Add papers on temporal/spatial modeling strategies

### 3.4 Performance Comparisons from Existing Literature

- **Key points to cover:**
  - Benchmark results across architecture types
  - Standardized datasets and evaluation metrics
  - Computational efficiency vs. accuracy trade-offs
  - Memory requirements considerations
- **Suggested references:**
  - Cuingnet et al. (2011) for early benchmarking
  - Basaia et al. (2019) for more recent comparisons
  - **Gap:** Add updated comparative studies from 2020 onwards

## 4. MRI Preprocessing for Deep Learning

### 4.1 Skull Stripping Methodologies

- **Key points to cover:**
  - Importance for AD classification
  - Comparison of traditional vs. learning-based (synthstrip) approaches
  - Quality considerations and failure modes
  - Impact on downstream classification performance
- **Suggested references:**
  - **Gap:** Add references on skull stripping methods and their impact

### 4.2 Registration and Normalization Approaches

- **Key points to cover:**
  - Standard space registration (MNI152, etc.)
  - Intensity normalization techniques
  - Impact of registration accuracy on classification
  - Trade-offs between standardization and preserving pathology
- **Suggested references:**
  - **Gap:** Add references on registration methods and effects on classification

### 4.3 Impact of Preprocessing on Model Performance

- **Key points to cover:**
  - Empirical studies measuring preprocessing effects
  - Sensitivity analysis of different preprocessing steps
  - Relative importance of preprocessing pipeline components
  - Domain-specific considerations for AD
- **Suggested references:**
  - **Gap:** Add studies measuring preprocessing effects on deep learning performance

### 4.4 Current Best Practices

- **Key points to cover:**
  - Consensus approaches in neuroimaging literature
  - Standardized pipelines (e.g., FreeSurfer, FSL)
  - Preprocessing considerations specific to deep learning
  - Areas of ongoing research and debate
- **Suggested references:**
  - **Gap:** Add references establishing best practices for neuroimaging preprocessing

## 5. Current State of the Art

### 5.1 Recent Advances in Automated AD Detection

- **Key points to cover:**
  - Leading approaches from recent literature (2020-2025)
  - Performance breakthroughs and methodological innovations
  - Multi-modal integration approaches
  - Current performance benchmarks
- **Suggested references:**
  - Francis et al. (2025), Lu et al. (2025), and Yan et al. (2025) represent recent approaches
  - Mubonanyikuzo et al. (2025) for meta-analysis
  - Saikia and Kalita (2024), Menagadevi et al. (2024) for reviews

### 5.2 Performance Limitations and Challenges

- **Key points to cover:**
  - Current performance ceilings
  - Generalizability concerns and dataset biases
  - Challenges in cross-dataset validation
  - Clinical integration barriers despite high reported accuracy
- **Suggested references:**
  - Basaia et al. (2019) for established benchmarks
  - Pradhan et al. (2024) for recent analysis
  - **Gap:** Add references discussing clinical translation limitations

### 5.3 Research Gap Addressed by This Work

- **Key points to cover:**
  - Synthesis of limitations in existing approaches
  - Novelty of applying video model transfer learning
  - Expected advantages of proposed approach
  - Positioning your research in the current landscape
- **Suggested references:**
  - This section should reference gaps identified throughout your review
  - Connect to your specific research questions and hypotheses

## Bibliography Gaps to Address

1. **Pathophysiology**: Add references on AD progression models and cellular-to-structural changes
2. **Radiologist performance**: Include studies on expert performance and variability in MRI assessment
3. **Clinical guidelines**: Add papers on current diagnostic criteria and clinical workflows
4. **Preprocessing**: Find references on skull stripping and registration methods specific to AD
5. **Video models**: Seek papers on video classification models adapted to medical volumetric data
6. **Clinical translation**: Include references discussing challenges in translating research to practice
7. **Standardized pipelines**: Add references on established preprocessing approaches in neuroimaging
8. **3D ResNet**: Find papers specifically on 3D ResNet implementations for medical imaging

## Writing Guidelines

1. Begin each section with broader concepts before narrowing to specifics
2. Make explicit connections between sections to create a coherent narrative
3. Balance technical detail with accessibility for interdisciplinary readers
4. Maintain a critical perspective, noting limitations of existing approaches
5. Use transitions to highlight how each topic relates to your research focus
6. Conclude each major section with implications for your research approach
