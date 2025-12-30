# AWS Certified AI Practitioner - Study Notes

## AWS AI/ML Services

### Amazon SageMaker AI
Build, train, and deploy machine learning models at scale.

**Core Components:**
- **SageMaker Studio** - Integrated IDE for machine learning development
- **SageMaker Canvas** - No-code ML model building interface
- **SageMaker JumpStart** - Pre-trained, open-source models for quick deployment

**Data Management:**
- **SageMaker Data Wrangler** - Visual interface to import, clean, transform, and analyze data; automatically surfaces data imbalance and drift issues
- **SageMaker Ground Truth** - Creates high-quality labeled datasets with human-in-the-loop workflows for images, text, video, and 3D point clouds
- **SageMaker Ground Truth Plus** - Managed data labeling service
- **SageMaker Feature Store** - Centralized repository for storing and sharing ML features across teams

**Model Development & Training:**
- **SageMaker Experiments** - Track, organize, view, analyze, and compare iterative ML experimentation
- **SageMaker Pipelines** - Orchestration and automation of end-to-end ML workflows
- **SageMaker Model Registry** - Catalog, version, and manage ML models throughout their lifecycle; supports model package versioning, approvals, and deployment lineage

**Model Evaluation & Monitoring:**
- **SageMaker Clarify** - Detect bias and explain model decisions; provides metrics for accuracy, robustness, and toxicity
- **SageMaker Model Cards** - Standardized documentation for model transparency and governance
- **SageMaker Model Dashboard** - Monitor deployed endpoints for data quality, model quality, bias, and drift
- **SageMaker Model Monitor** - Real-time performance monitoring for production models

**Deployment:**
- **SageMaker Inference Endpoint** - Allows clients to invoke deployed models for predictions; supports shadow testing and A/B testing via traffic shifting across production variants

### Amazon Bedrock
Fully managed service for foundation models with orchestration and customization capabilities.

- **Amazon Bedrock Guardrails** - Detect PII and unsafe content; enforce allowed/denied topics, word filters, and response boundaries
- **Amazon Bedrock Agents** - Orchestrate tool-calling and knowledge bases for complex tasks (customer service, data analysis); supports RAG via Knowledge Bases
- **Amazon Bedrock Model Evaluation** - Evaluate foundation models and custom models using automatic/offline metrics or human review

### Amazon Q
AI-powered assistant suite for different use cases.

- **Amazon Q Business** - Enterprise RAG assistant with data connectors, access controls, and built-in guardrails
- **Amazon Q Developer** - Conversational code assistant with security scanning, AWS integration, and agentic task automation
- **Amazon Q in Connect** - AI agent assist for contact center operations
- **Amazon Q in QuickSight** - Natural Language Query (NLQ) and AutoGraph for business intelligence and analytics

### AI Services - Language & Text

- **Amazon Lex** - Build conversational chatbots and voice assistants
- **Amazon Comprehend** - Natural language processing (NLP) for sentiment analysis, entity recognition, and topic modeling
- **Amazon Comprehend Medical** - Extract medical information from unstructured clinical text
- **Amazon Kendra** - Intelligent search service to query documents using natural language
- **Amazon Textract** - Extract text and structured data (forms, tables) from images and PDFs
- **Amazon Transcribe** - Speech-to-text (STT) conversion
- **Amazon Transcribe Medical** - Medical-specific speech-to-text transcription

### AI Services - Vision & Media

- **Amazon Rekognition** - Image and video analysis (faces, objects, scenes, activities); does not directly ingest PDFs
- **Amazon Polly** - Text-to-speech (TTS) with natural-sounding voices
- **AWS Panorama** - Run computer vision models on edge devices for on-premises cameras (low-latency, no cloud upload required)

### Additional AI Services

- **Amazon Personalize** - Build personalized recommendation systems
- **Amazon Forecast** - Fully managed time-series forecasting (demand planning, capacity planning)
- **Amazon Fraud Detector** - Managed fraud detection using machine learning with built-in rules and model hosting
- **Amazon Connect** - AI-powered cloud contact center platform
- **Amazon Augmented AI (A2I)** - Human review workflows for ML predictions
- **Amazon Mechanical Turk (MTurk)** - Crowdsourcing marketplace for distributing tasks to human workers

### AI Documentation & Resources

- **AWS AI Service Cards** - Official documentation describing AWS AI services, their capabilities, and use cases

### Specialized Hardware

- **AWS Trainium** - Custom ML training chip optimized for training models with high energy efficiency
- **AWS Inferentia** - High-performance, low-cost inference chip for deep learning and generative AI applications
- **AWS DeepRacer** - Wi-Fi-enabled autonomous vehicle for learning reinforcement learning through hands-on racing

### Security & Compliance Services

- **Amazon Macie** - Automatically discover and protect sensitive data in Amazon S3
- **Amazon Inspector** - Automated vulnerability scanning for EC2 instances, Lambda functions, and ECR container images (CVEs and runtime issues)
- **AWS Config** - Continuously assess, audit, and evaluate AWS resource configurations
- **AWS Artifact** - Self-service portal for security and compliance documentation
- **AWS Trusted Advisor** - Real-time recommendations for cost optimization, security, performance, fault tolerance, and service limits (full checks require Business/Enterprise support)
- **AWS Audit Manager** - Automate evidence collection for continuous compliance auditing

---

## Evaluation Metrics

### Classification Metrics

- **Classification accuracy** - Ratio of correct predictions to total predictions: `(TP + TN) / (TP + TN + FP + FN)`
- **Precision** - Ratio of true positives to all predicted positives: `TP / (TP + FP)` - Answers: "Of all positive predictions, how many were correct?"
- **Recall (Sensitivity)** - Ratio of true positives to all actual positives: `TP / (TP + FN)` - Answers: "Of all actual positives, how many did we catch?"
- **F1 Score** - Harmonic mean of precision and recall: `2 × (Precision × Recall) / (Precision + Recall)` - Useful when you need balance between precision and recall
- **Confusion Matrix** - Visual representation showing true positives, true negatives, false positives, and false negatives

### Regression Metrics

- **MAE (Mean Absolute Error)** - Average of absolute differences between predictions and actual values
- **MSE (Mean Squared Error)** - Average of squared differences: `(y - y_pred)²`
- **RMSE (Root Mean Squared Error)** - Square root of MSE; same units as target variable
- **R-squared (R²)** - Proportion of variance in the dependent variable predictable from independent variables

### Natural Language Processing Metrics

- **BLEU (Bilingual Evaluation Understudy)** - Evaluates machine translation quality by comparing n-gram overlap with reference translations
- **ROUGE-N (Recall-Oriented Understudy for Gisting Evaluation)** - Measures text summarization quality via n-gram overlap with reference summaries
- **BERTScore** - Evaluates text generation quality using contextual embeddings and semantic similarity
- **Perplexity** - Measures language model quality; lower values indicate better performance

### Other Metrics

- **Correlation Matrix** - Shows statistical relationships between variables in a dataset; useful for understanding feature dependencies

---

## Data Preparation & Feature Engineering

### Feature Engineering
The process of selecting, modifying, or creating features from raw data to improve ML model performance.

**For Structured Data:**
- Normalization and standardization
- Handling missing values
- Encoding categorical variables
- Creating interaction features

**For Unstructured Data:**
- **Text**: Tokenization, vectorization (Word2Vec, embeddings)
- **Images**: Feature extraction, dimensionality reduction
- **Audio**: Spectral features, mel-frequency cepstral coefficients (MFCCs)

**Techniques:**
- **Feature Extraction** - Transform data into a new feature space (e.g., PCA)
- **Feature Selection** - Reduce the number of features to most relevant ones
- **Feature Creation** - Engineer new features from existing data

### Training Approaches

- **Data Collection** - Label, ingest, and aggregate data from various sources
- **Instruction-Based Fine-Tuning** - Use labeled prompt-response pairs formatted as instructions for supervised fine-tuning
- **Dataset Splits** - Common rule of thumb: 80% training / 10% validation / 10% test; adjust based on data size and domain; use cross-validation when data is limited

---

## Core AI Concepts

### Model Performance Issues

**Overfitting:**
- Model performs well on training data but fails to generalize to new data
- Caused by high variance (model too complex)
- **Prevention techniques**: Early stopping, cross-validation, regularization, pruning, dropout

**Underfitting:**
- Model is too simple to capture underlying patterns in the data
- Caused by high bias (model not complex enough)
- **Solutions**: Increase model complexity, add more features, reduce regularization

**Variance vs. Bias:**
- **Low Variance** - Predictions are stable across different training sets; risk of underfitting if model is too simple
- **High Variance** - Model fits training data too closely; leads to overfitting and poor generalization

### Model Interpretability & Explainability

- **Explainability** - Ability to understand how a model arrives at specific predictions
- **Interpretability** - Understanding the internal mechanisms and logic of a model
- **Shapley Values** - Local interpretability method that assigns each feature a contribution score for individual predictions
- **Partial Dependence Plots (PDP)** - Global interpretability method showing how predictions change as a single feature varies while holding others constant

### Bias & Fairness

**Bias** - Unfair prejudice that favors or disfavors certain individuals or groups.

**Types of Bias:**
- **Sampling Bias** - Training data not representative of the target population
- **Measurement Bias** - Inaccurate or unreliable data collection methods
- **Observer Bias** - Researcher's expectations influence data collection
- **Confirmation Bias** - Tendency to favor information confirming existing beliefs

**Fairness** - Impartial and just treatment without discrimination based on protected attributes

### AI Risks & Security

- **Hallucination** - Model generates false or nonsensical information presented as fact
- **Toxicity** - Model produces harmful, offensive, or inappropriate content
- **Poisoning** - Malicious actors corrupt training data to compromise model behavior
- **Prompt Leaking** - Exposing system prompts or instructions through adversarial queries
- **Hijacking** - Manipulating AI systems for malicious purposes (phishing, malware distribution)
- **Jailbreaking** - Bypassing built-in safety measures to generate prohibited content or access restricted features

### Other Concepts

- **Model Inference** - Process of generating outputs (predictions/responses) from given inputs (features/prompts)
- **Transfer Learning** - Applying knowledge from one domain to improve performance in a related domain

---

## Machine Learning Algorithms

### Supervised Learning
Training on labeled data where correct outputs are known.

**Algorithms:**
- **Linear Regression** - Predict continuous values with linear relationships
- **Decision Trees** - Rule-based models; deterministic (same input → same output)
- **Neural Networks** - Multi-layer networks for complex pattern recognition (e.g., handwritten digit recognition)
- **K-Nearest Neighbors (KNN)** - Classify based on similarity to nearest training examples
- **Support Vector Machines (SVM)** - Find optimal hyperplane to separate classes; effective for high-dimensional data

**Use Cases:**
- Classification (spam detection, image recognition)
- Regression (price prediction, demand forecasting)
- Forecasting (DeepAR algorithm for time series)

### Unsupervised Learning
Finding patterns in unlabeled data without predefined outputs.

**Algorithms:**
- **K-Means Clustering** - Group similar data points into clusters (e.g., customer segmentation, network traffic analysis)
- **Random Cut Forest (RCF)** - Anomaly detection algorithm to identify outliers
- **Association Rule Learning** - Discover relationships between variables (market basket analysis)
- **Dimensionality Reduction** - Reduce feature space while preserving important information

**Use Cases:**
- Customer segmentation
- Anomaly detection for fraud or security incidents
- Data compression and visualization

### Semi-Supervised Learning
Combines small amounts of labeled data with large amounts of unlabeled data.

**Use Cases:**
- Document classification
- Fraud identification
- Sentiment analysis

### Reinforcement Learning
Agent learns by interacting with an environment, receiving rewards or penalties.

**Key Concepts:**
- Agent takes actions in an environment
- Receives rewards or penalties based on actions
- Learns optimal policy to maximize cumulative rewards over time

**Variants:**
- **RLHF (Reinforcement Learning from Human Feedback)** - Uses human feedback to optimize ML models for more efficient self-learning

**Use Cases:**
- Game playing (AWS DeepRacer)
- Robotics control
- Resource optimization

### Deep Learning
Neural networks with multiple layers that automatically learn hierarchical features.

**Architectures:**
- **Convolutional Neural Networks (CNN)** - Extract features from images; models include AlexNet, YOLO, Faster R-CNN
- **Recurrent Neural Networks (RNN)** - Process sequential data (text, time series, speech)
- **Transformers** - Attention-based architecture for NLP tasks (GPT, BERT)

**Use Cases:**
- Image classification and object detection
- Natural language processing
- Speech recognition

### Generative Models
Learn to generate new data that resembles training data.

**Model Types:**
- **Generative Adversarial Networks (GAN)** - Two competing neural networks (generator vs. discriminator)
- **Variational Autoencoders (VAE)** - Encoder-decoder architecture using latent space representation
- **Diffusion Models (DALL-E)** - Create data through iterative controlled random changes
- **Transformer-Based Models (GPT)** - Build on encoder-decoder concepts with attention mechanisms
- **WaveNet** - Generative model for realistic speech synthesis

**Discriminative vs. Generative:**
- **Discriminative Models** - Learn to distinguish between classes (classification)
- **Generative Models** - Learn to create new samples resembling training data

### Specialized Techniques

- **Incremental Training** - Continuously learn from new data without forgetting previous knowledge
- **Retrieval-Augmented Generation (RAG)** - Enhance LLM outputs by referencing external knowledge bases before generating responses
- **Named Entity Recognition (NER)** - Identify and classify entities (people, organizations, locations, dates) in text

---

## Foundation Models & LLMs

### Model Types

- **GPT (Generative Pre-trained Transformer)** - Transformer-based model for text generation, code generation, and reasoning tasks
- **BERT (Bidirectional Encoder Representations)** - Captures contextual word meaning by analyzing surrounding context in both directions
- **Claude** - Advanced reasoning, vision analysis, code generation, multilingual processing
- **Llama 3.1** - Open-source text generation model from Meta
- **Jurassic** - Question answering, summarization, information extraction, reasoning and logic tasks
- **Stable Diffusion** - Generates photorealistic images from text and image prompts

### Embedding Models

Transform data into dense vector representations in multi-dimensional space.

- **Word2Vec** - Creates word vectors based on co-occurrence patterns in text
- **BERT Embeddings** - Context-aware embeddings capturing semantic relationships
- **Principal Component Analysis (PCA)** - Dimensionality reduction while preserving variance
- **Singular Value Decomposition (SVD)** - Matrix decomposition for compression and noise reduction

---

## Model Customization

### Fine-Tuning (Foundation Models)
Adapt pre-trained models to specific tasks using labeled data.

**Requirements:**
- Labeled training data (prompt-response pairs or task-specific examples)
- Appropriate compute resources

**Hyperparameters:**
- **Epochs** - Number of complete passes through the training dataset
- **Learning Rate** - Step size for updating model weights between epochs
- **Batch Size** - Number of training examples processed together in each iteration

### Continued Pre-Training
Further train foundation models on domain-specific unlabeled data to adapt to specialized vocabulary and patterns.

**Requirements:**
- Large amounts of unlabeled domain-specific text
- Significant compute resources

---

## Prompt Engineering

### Prompting Techniques

- **Zero-Shot** - Model performs task without examples; relies solely on pre-training
- **Few-Shot** - Provide a few examples to guide model behavior
- **Chain-of-Thought** - Encourage step-by-step reasoning for complex problems
- **Negative** - Specify what to avoid in generated content

### Inference Parameters

- **Temperature (0.0 - 1.0)**
  - **Higher values** (0.7-1.0): More random, creative, diverse outputs
  - **Lower values** (0.0-0.3): More deterministic, focused, consistent outputs

- **Top K (integer, 0-500)**
  - Number of most likely next tokens to consider
  - Lower values (e.g., 10): More focused, coherent output
  - Higher values: More diverse possibilities

- **Top P (0.0 - 1.0, nucleus sampling)**
  - Cumulative probability threshold for token selection
  - Lower values (e.g., 0.25): Only top 25% of probability mass considered
  - Higher values (e.g., 0.9): More diverse token choices

---

## Computer Vision

### Architectures

- **Convolutional Neural Networks (CNN)** - Foundation of modern computer vision; extract hierarchical features from images
- **AlexNet** - Pioneering deep CNN for image classification
- **YOLO (You Only Look Once)** - Real-time object detection in single pass
- **Faster R-CNN** - High-accuracy region-based object detection

### Classification Types

- **Multi-Class Classification** - Each instance assigned to exactly one class from multiple options
- **Multi-Label Classification** - Each instance can be assigned to multiple classes simultaneously

---

## Probability & Statistical Models

- **Bayesian Networks** - Graphical models representing probabilistic relationships among variables; provide probability distributions for different outcomes
- **Decision Trees** - Deterministic models following rule-based paths; same input always produces same output

---

## Data Governance & Compliance

- **Data Residency** - Geographic location where data is physically stored and processed
- **Data Retention Policy** - Duration for which data is stored before deletion or archival
- **Data Security** - Protection from unauthorized access, use, disclosure, modification, or destruction
- **Data Integrity** - Ensuring data remains accurate, consistent, and unaltered throughout its lifecycle
- **Data Lineage** - Complete history of data transformations and usage over time
- **Benchmark Dataset** - Standardized dataset used to evaluate and compare model performance across different approaches

---

## Quick Reference

### When to Use Which Service

**Text Analysis**: Comprehend, Comprehend Medical, Kendra, Textract  
**Image/Video**: Rekognition, Textract (documents)  
**Speech**: Transcribe, Transcribe Medical, Polly  
**Chatbots**: Lex, Q Business  
**Custom ML**: SageMaker  
**Foundation Models**: Bedrock  
**Code Assistance**: Q Developer  
**Contact Centers**: Connect, Q in Connect

### Common Exam Scenarios

- **RAG Implementation**: Bedrock Knowledge Bases + Bedrock Agents
- **Data Labeling**: SageMaker Ground Truth / Ground Truth Plus
- **Bias Detection**: SageMaker Clarify
- **Model Monitoring**: SageMaker Model Monitor
- **No-Code ML**: SageMaker Canvas
- **Document Processing**: Textract (extract) → Comprehend (analyze)
- **Sensitive Data**: Macie (S3), Bedrock Guardrails (prompts/responses)