# The Approach

"Most of the MRI studies demonstrated that atrophy of the medial temporal lobe structures (hippocampus, and entorhinal cortex) is common in AD. Structural MRI analysis has demonstrated that medial temporal atrophy is associated with increased risk of developing AD and can predict future memory decline in healthy adults."

- https://pmc.ncbi.nlm.nih.gov/articles/PMC4243931/#S5

"Longitudinal structural and functional imaging studies seem currently most robust to evaluate progressive impairment in MCI and AD. However, from the perspective of developing countries of the many technologies available, CT head scan and structural MRI imaging are the most useful, widely available and affordable imaging modalities."

- https://pmc.ncbi.nlm.nih.gov/articles/PMC4243931/#S20

## Biomarkers

![Biomarkers Image](imgs/biomarkers.jpg)

Biomarkers of Alzheimer's disease are neurochemical indicators that can help diagnose the disease early and monitor its progression. Some examples of biomarkers include:

### Amyloid beta and tau

These proteins build up in the brain in Alzheimer's disease, and can be detected in cerebrospinal fluid (CSF) or blood.

### Hippocampal atrophy

Measured by MRI, this can predict the conversion from mild cognitive impairment (MCI) to Alzheimer's disease.

### MicroRNAs (miRNAs)

These short RNAs can circulate in the blood and their levels can be detected using techniques like RNA sequencing or microarray analysis.

### Insulin pathway proteins

Levels of phospho-Ser312-insulin receptor substrate-1 (IRS-1) and phospho-panTyr-IRS-1 can be used as biomarkers.

### Lysosomal proteins

Levels of cathepsin D, alysosome-associated membrane protein, and ubiquitinated proteins can be used to discriminate Alzheimer's dementia.

### Repressor element 1-silencing transcription factor (REST)

Levels of REST are significantly lower in Alzheimer's patients and MCI compared to control subjects.

## Models

### 1. **ResNet (Residual Networks)**

- **Strengths**: Introduced residual connections to combat vanishing gradients, making it highly effective for training very deep networks. ResNet-50 and ResNet-101 are widely used for transfer learning.
- **Tradeoffs**: Can be computationally expensive due to depth, and shallower models might suffice for simpler datasets.

### 2. **VGG (Visual Geometry Group)**

- **Strengths**: Simplicity in architecture (stacked convolutional layers), which makes it easy to understand and modify. Works well with pre-trained weights like VGG16 or VGG19.
- **Tradeoffs**: Requires a large amount of memory and computational resources, making it inefficient compared to newer architectures.

### 3. **DenseNet (Dense Convolutional Networks)**

- **Strengths**: Each layer is connected to every other layer, improving feature reuse and efficiency. Typically smaller and faster to train compared to similarly deep models.
- **Tradeoffs**: High memory consumption due to dense connections, and may be overkill for simpler tasks.

### 4. **EfficientNet**

- **Strengths**: Balances accuracy and efficiency by scaling width, depth, and resolution uniformly. State-of-the-art performance on many benchmarks.
- **Tradeoffs**: Complex to implement manually due to compound scaling, and might require fine-tuning of scaling factors for specific tasks.

### 5. **Inception (GoogLeNet)**

- **Strengths**: Uses inception modules to combine filters of different sizes, capturing multi-scale features effectively. Good at reducing computational cost compared to models like VGG.
- **Tradeoffs**: More complex architecture, making it harder to customize or extend.

### 6. **MobileNet**

- **Strengths**: Optimized for mobile and embedded devices, with lightweight architecture and low resource requirements. Excellent for real-time applications.
- **Tradeoffs**: Sacrifices some accuracy compared to deeper models, particularly on complex datasets.

### 7. **Vision Transformers (ViT)**

- **Strengths**: Uses attention mechanisms to process image patches, excelling in capturing long-range dependencies and global features. Performs well on large datasets.
- **Tradeoffs**: Requires large amounts of data for effective training and may be less efficient for smaller-scale tasks compared to CNNs.

## 8. **ConvNeXt**

https://arxiv.org/abs/2201.03545

- **Strengths**: A modernized CNN architecture inspired by the design principles of Vision Transformers, improving on ResNet's backbone. It achieves competitive or superior accuracy while maintaining the computational efficiency of CNNs.
- **Tradeoffs**: While highly performant, ConvNeXt is relatively new, meaning it has less pre-trained support and community resources compared to older models like ResNet.

## Datasets

### Oasis

https://sites.wustl.edu/oasisbrains/

### ADNI

https://adni.loni.usc.edu/data-samples/adni-data/neuroimaging/mri/

## Plan

## The biomarker

atrophy, seen as that's what most commonly used as the marker in neuroimaging techniques

## The Scan

structural MRI, it details the shape and size of the brain needed to recognise atrophy. Also, more widely available.

types of struc imaging t1 t2 etc, look what kind sof patients, pool the data to decide on what kinf of scan I want to focus on one.

## the Models

Going to use pytorch https://pytorch.org/vision/main/models.html#table-of-all-available-classification-weights

To start off I want to learn more and test these:

- ConvNeXt

- ResNet

- DenseNet

- EfficientNet

- VGG

I want to undertand these morph/3d architectures better:

- 3D CNN https://paperswithcode.com/paper/detection-of-dementia-through-3d
- EfficientMorph https://paperswithcode.com/paper/efficientmorph-parameter-efficient
- TransMorph https://paperswithcode.com/paper/transmorph-transformer-for-unsupervised
- VoxelMorph https://paperswithcode.com/paper/voxelmorph-a-learning-framework-for

## Questions

- what can I do to account for the 3d nature of these scans
-

review the models and their relevenece to medical imaging transfer learning
look into unet
look at a reduction method to get to 2d
comparison between 2d with reductions and 3d

GET the data
data augmentation, perhaps especially for healthy patients, peak at the paper, play with contrast
