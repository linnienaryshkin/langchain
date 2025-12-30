# AWS Exam

## AWS GenAI and ML Platforms
- Amazon SageMaker | build, train, and deploy ML models
  - SageMaker Model Cards | model documentation
  - SageMaker JumpStart | pre-trained, open source models
  - SageMaker Data Wrangler | import, clean, transform, and analyze data; surfaces imbalance/drift
  - SageMaker Ground Truth | human-in-the-loop labeling for images, text, video, 3D point clouds; produces labels for downstream training
    - SageMaker GroundTruth Plus | data labeling service
  - SageMaker Canvas | build ML models without writing code
  - SageMaker Studio | IDE for machine learning
  - SageMaker Clarify | explainability and bias detection/mitigation
  - SageMaker Experiments | track and compare experiments
  - SageMaker Pipelines | orchestration and automation of ML workflows
  - SageMaker Model Registry | catalog, version, and manage ML models
  - SageMaker Feature Store | centralized repository for sharing ML features
  - SageMaker Model Dashboard | monitor endpoints for data/model quality, bias, and drift using Model Monitor jobs
    - SageMaker Model Monitor | monitor model performance in real-time
  - SageMaker Inference endpoint | invoke deployed models
- Amazon Bedrock | foundation models and orchestration
  - Amazon Bedrock Guardrails | detect PII/unsafe content; enforce allowed/denied topics, word filters, response boundaries
  - Amazon Bedrock Agents | tool-calling and knowledge bases for tasks; can use Knowledge Bases for RAG
  - Amazon Bedrock Model Evaluation | evaluate FMs/custom models (automatic/offline and human review options)
- Amazon Q
  - Amazon Q in Connect | agent assist for contact centers
  - Amazon Q in QuickSight | Natural Language Query (NLQ) and AutoGraph for BI dashboards
  - Amazon Q Developer | conversational code assistant with security scanning, AWS integration, agentic task automation
  - Amazon Q Business | enterprise RAG assistant with connectors, access controls, guardrails
- AWS Inferentia | inference accelerator for DL/GenAI
- AWS Trainium | training accelerator for ML/GenAI
- AWS DeepRacer | physical car for RL learning

## AI Application Services (by modality)
- Conversational and contact center: Amazon Lex; Amazon Connect
- Speech: Amazon Transcribe; Amazon Transcribe Medical; Amazon Polly
- Vision and documents: Amazon Rekognition (images/video; no PDFs); Amazon Textract (text/structured data from images/PDFs, forms/tables)
- Search and retrieval: Amazon Kendra
- Language/NLP: Amazon Comprehend; Amazon Comprehend Medical
- Recommendations: Amazon Personalize
- Human review: Amazon Mechanical Turk (MTurk); Amazon Augmented AI (A2I)
- Documentation: AWS AI Service Cards

## Security, Compliance, and Governance
- Amazon Macie | sensitive data protection in S3
- Amazon Inspector | automated vulnerability scanning for EC2, Lambda, ECR images (CVEs/runtime)
- AWS Config | assess/audit resource configurations
- AWS Artifact | compliance reports for AWS Cloud
- AWS Trusted Advisor | recommendations for cost, security, performance, fault tolerance, limits (full checks need Business/Enterprise support)
- AWS Audit Manager | automate evidence collection for audits

## Metrics
- Classification: Accuracy; Precision; Recall; F1; Confusion matrix
- Regression: MSE; RMSE; MAE; R-squared; Correlation matrix
- Generative/NLG: ROUGE-N; BERTScore; Perplexity; BLEU

## Data Preparation and Training
- Feature engineering: normalization, handling missing values, encoding categorical variables, tokenization/vectorization, feature extraction (e.g., PCA), feature selection
- Data collection: label, ingest, aggregate
- Instruction-based fine-tuning: prompt-response labeled pairs
- Dataset splits: training/validation/test (e.g., 80/10/10; adjust to data size/domain; cross-validation when limited data)

## Prompting, Parameters, and Customization
- Prompt parameters: Temperature; Top K; Top P (nucleus sampling)
- Prompt engineering techniques: Zero-shot; Few-shot; Chain-of-thought; Negative prompting
- Model customization: Fine-tuning (labeled data; epochs, learning rate, batch size); Continued pre-training (unlabeled data)

## AI and ML Concepts
- Overfitting (high variance); Underfitting (high bias); mitigation includes early stopping, cross-validation, regularization, pruning
- Explainability and interpretability
- Bias types: sampling, measurement, observer, confirmation; Fairness
- Model inference | generating outputs from inputs
- Risks: Hallucination, Toxicity, Poisoning, Prompt leaking
  - Hijacking | malicious manipulation
  - Jailbreaking | bypassing safety controls
- Bayesian Networks | probabilistic relationships with outcome probabilities
- GPT (Generative Pre-trained Transformer) | natural language understanding/generation
- Discriminative vs Generative models
- Stable Diffusion; Llama 3.1; Jurassic; Claude

## ML Models, Algorithms, and Tools
- Regression models | predict continuous values
- Variance: Low variance (stable predictions; underfitting if bias/complexity too low); High variance (overfitting)
- Shapley values | local interpretability
- Partial Dependence Plots (PDP) | global interpretability
- Retrieval-Augmented Generation (RAG) | ground LLM outputs with external knowledge
- WaveNet | generative speech model
- Multi-class vs multi-label classification
- Named Entity Recognition (NER)
- Deep learning | multiple layers of neurons to learn features
- Reinforcement learning | actions/rewards; RLHF uses human feedback
- Transfer learning | reuse knowledge across domains
- Supervised learning | examples include linear regression, neural networks, forecasting/DeepAR, classification, decision trees, KNN, SVM
- Semi-supervised learning | document classification, fraud ID, sentiment analysis
- Unsupervised learning | association rule learning; clustering/K-Means; anomaly detection (Random Cut Forest); dimensionality reduction
- Incremental training | adapt over time without forgetting
- Diffusion model (e.g., DALL-E) | iterative denoising generation
- Generative adversarial network (GAN) | two networks compete to generate realistic data
- Variational autoencoders (VAE) | encoder/decoder with latent Gaussian sampling
- Transformer-based generative AI (GPT-style) | stacked encoder/decoder concepts for text tasks
- Computer vision models: CNNs (AlexNet, YOLO, Faster R-CNN); RNNs for sequences

## Embedding and Transformer Models
Embedding models create dense representations for high-dimensional data.
- BERT | bidirectional contextual word representations
- PCA | dimensionality reduction
- Word2Vec | word vectors from co-occurrence
- SVD | matrix decomposition for compression/noise reduction

## Data Governance Concepts
- Data residency | where data is stored/processed
- Data retention policy | how long data is stored/processed
- Data security | protection from unauthorized access or change
- Data integrity | accuracy and consistency of data
- Data lineage | history of how data is transformed and used
- Benchmark dataset | used to evaluate and compare model performance