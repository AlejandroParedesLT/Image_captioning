# Image Captioning Evaluation: CNNs, RNNs, and Transformers

## Overview
This project evaluates image captioning models, focusing on the integration of Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) for automatic image description generation. We analyze various architectures, ranging from CNN-LSTM models to transformer-based models, using the Flickr30k dataset. Key objectives include understanding the efficacy of attention mechanisms and exploring state-of-the-art methods for image captioning.

## Dataset
The **Flickr30k** dataset is utilized, comprising 31,014 images, each annotated with four unique captions. This dataset provides a robust basis for training and evaluating image captioning systems.

## Architectures
### 1. **CNN-LSTM Model**
- **Encoder:** Pretrained VGG16 network extracts visual features from the last hidden layer.
- **Decoder:** LSTM network generates textual descriptions based on encoded image features.
- **Loss Function:** A combined loss function is used for optimizing the model.
- **Evaluation Metric:** BLEU score evaluates the quality of the generated captions.

### 2. **Attention-Enhanced CNN-LSTM Model**
- **Encoder:** ResNet101 network extracts image features.
- **Decoder:** LSTM with an attention mechanism focuses on spatial relationships, improving contextual understanding.
- **Enhancement:** Attention head allows the model to dynamically emphasize different regions of an image during caption generation.

### 3. **Transformer-Based Model**
- **Architecture:** Fine-tunes a pretrained Microsoft/git-base transformer model.
- **Strengths:** Demonstrates the potential advantages of transformers in capturing complex relationships between image features and textual representations.

## Results
- The **attention-enhanced LSTM model** significantly improves caption quality compared to the baseline CNN-LSTM architecture.
![image](https://github.com/user-attachments/assets/9e7fae8b-6bc5-4e2f-ac9c-232972cfdd82)

- The **transformer-based model** shows state-of-the-art performance, emphasizing its suitability for image captioning tasks.
![image](https://github.com/user-attachments/assets/8e704c08-e8ed-42ce-8796-ebc45eb82150)


## Implementation Details
### Environment
- **Libraries:** PyTorch, Hugging Face Transformers, NumPy, Matplotlib.
- **Tools:** Jupyter Notebook for experimentation and visualization.
- **Hardware:** Training conducted on an NVIDIA GeForce GTX 1660 Ti GPU (CUDA 12.5).

### Workflow
1. **Data Preprocessing:**
   - Resize and normalize images.
   - Tokenize captions and create vocabulary.
2. **Model Training:**
   - CNN-LSTM models trained using cross-entropy loss.
   - Attention-enhanced models leverage an additional attention loss.
   - Transformers fine-tuned on the Flickr30k dataset.
3. **Evaluation:**
   - BLEU metric used for consistent performance measurement.

## Challenges and Solutions
1. **Handling Large Data:** Addressed by batch processing and using data loaders with optimized collate functions.
2. **Complexity of Attention Mechanisms:** Simplified by implementing prebuilt modules from PyTorch.
3. **Debugging Runtime Errors:** Resolved dtype inference issues in data loaders by ensuring compatibility between data preprocessing and model input requirements.

## Future Work
- Extend evaluation to other datasets like COCO for generalization.
- Explore multimodal transformers (e.g., ViT-GPT2).
- Investigate reinforcement learning approaches for caption optimization.

## Conclusion
This study highlights the evolution of image captioning models from CNN-LSTM architectures to attention-enhanced and transformer-based models. The findings underscore the importance of attention mechanisms and the transformative potential of pretrained models for improving image captioning systems.

References:

:notebook: [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)

:notebook: [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)

:notebook: [Automated Image Captioning with ConvNets and Recurrent Nets](https://cs.stanford.edu/people/karpathy/sfmltalk.pdf)
