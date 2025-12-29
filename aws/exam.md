# AWS Exam

## AWS Services

- Amazon SageMaker AI | build, train, and deploy machine learning models
  - SageMaker Model Cards | documentation of the model
  - SageMaker JumpStart | pre-trained, open source models
  - SageMaker Ground Truth | creation of high-quality labeled datasets by incorporating human feedback in the labeling process, which can be used to improve reinforcement learning models
  - SageMaker Canvas | build machine learning models without writing code
  - SageMaker Studio | IDE for machine learning
  - SageMaker Clarify | explain the model's decisions and detect bias, identify and mitigate bias in machine learning models and datasets
  - SageMaker Pipelines | orchestration and automation of ML workflows
  - SageMaker Model Registry | catalog, version, and manage ML models
  - SageMaker Feature Store | centralized repository for storing and sharing ML features
  - SageMaker Data Wrangler | visual interface to import, clean, transform, and analyze data
  - SageMaker MLflow | track, organize, view, analyze, and compare iterative ML experimentation to gain comparative insights and register and deploy your best-performing models
  - SageMaker Model Dashboard | monitor and manage your models in real-time
  - SageMaker Inference endpoint | allows clients to invoke deployed models
  - SageMaker Model Monitor | monitor the performance of your models in real-time
- Amazon Bedrock | pre-trained models, orchestration, MCP
  - Amazon Bedrock Guardrails | detects sensitive information such as personally identifiable information (PIIs) in input prompts or model responses
  - Amazon Bedrock Agents | build agents that can perform tasks such as customer service, data analysis, and more
  - Amazon Bedrock Model Evaluation | preparing data, training models, selecting appropriate metrics, testing and analyzing results
- Amazon Q
  - Amazon Q in Connect | contact center service from AWS
  - Amazon Q in QuickSight | business intelligence (BI) service that allows users to easily create and share interactive dashboards and visualizations from various data sources
  - Amazon Q Developer | conversational assistant for developers that provides code assistance, security scanning, AWS integration, and agentic capabilities for automating multi-step tasks
  - Amazon Q Business | generative AI assistant for business users that connects to enterprise data sources, generates content, automates tasks, and enables building AI-driven applications
- Amazon Lex | build chatbots and voice assistants
- Amazon Rekognition | image/video analysis
- Amazon Polly | text-to-speech (TTS)
- Amazon Transcribe | speech-to-text (STT)
- Amazon Textract | image -> text/data
- Amazon Kendra | search/query documents
- Amazon Comprehend | natural language processing (NLP)
- Amazon Inspector | security and compliance of AWS resources
- AWS AI Service Cards | documentation of the AI services
- Amazon Personalize | build recommendation systems
- Amazon Mechanical Turk (MTurk) | marketplace for outsourcing various tasks to a distributed workforce
- Amazon Augmented AI (Amazon A2I) | human review workflows for machine learning predictions

- Amazon Macie | protect sensitive data in S3
- AWS Config | continuously assess, audit, and evaluate the configurations of your AWS resources
- AWS Artifact | security and compliance documentation for the AWS Cloud
- AWS Trusted Advisor | recommendations for cost optimization, security, and resilience
- AWS Audit Manager | automate the collection of evidence to continuously audit your AWS usage

## Metrics

- Classification accuracy | the ratio of correct predictions to all predictions (TP + TN) / (TP + TN + FP + FN)
- classification systems
  - Precision | the ratio of true positives to all predicted positives (TP) / (TP + FP)
  - Recall | the ratio of true positives to all actual positives (TP) / (TP + FN)
  - F1 score | harmonic mean of precision and recall: 2 * (precision * recall) / (precision + recall)
- regression models
  - MSE (Mean Squared Error) | the average of the squares of the errors (y - y_pred)^2
  - RMSE (Root Mean Squared Error) | the square root of the average of the squares of the errors
  - MAE (Mean Absolute Error) | the average of the absolute values of the errors
  - R-squared
- Confusion matrix | evaluate the performance of classification models by displaying the number of true positives, true negatives, false positives, and false negatives
- Correlation matrix | measures the statistical correlation between different variables or features in a dataset, typically used to understand the relationships between continuous variables
- ROUGE-N (Recall-Oriented Understudy for Gisting Evaluation) | evaluate the quality of the generated text by comparing it to the reference text for n-grams
- BERTScore | evaluate the quality of the generated text by comparing it to the reference text
- Perplexity | probability of a model to generate a given sequence of words
- BLEU (Bilingual Evaluation Understudy) | evaluate the quality of text that has been machine-translated by comparing it with one or more reference translations

## Data Preparation and Training

- Feature Engineering | selecting, modifying, or creating features from raw data to improve the performance of machine learning models
  - For structured data typically includes tasks like normalization, handling missing values, and encoding categorical variables
  - For unstructured data, such as text or images, feature engineering involves different tasks like tokenization (breaking down text into tokens), vectorization (converting text or images into numerical vectors), and extracting features that can represent the content meaningfully
  - Feature extraction (e.g. Principal Component Analysis (PCA)) | transforming the data into a new feature space
  - Feature selection | reduce the number of features
- Data collection | label, ingest, and aggregate data
- Instruction-based fine-tuning | labeled examples that are formatted as prompt-response pairs and that are phrased as instructions

## AI concepts

- Overfitting | when a model performs well on training data but fails to generalize to new data.
  - To prevent overfitting, techniques such as early stopping, cross-validation, regularization, and pruning can be used.
  - high variance can cause overfitting
- Underfitting | when a model is too simple to capture the underlying patterns in the data.
  - high bias can cause underfitting
- Explainability | the ability to understand how a model arrives at a prediction.
- Interpretability | understanding the internal mechanisms of a machine learning model
- Bias | unfair prejudice or preference that favors or disfavors a person or group.
  - Sampling bias | the selection of a sample that is not representative of the population
  - Measurement bias | the measurement of a variable that is not accurate or reliable
  - Observer bias | the observer's bias in the data collection process
  - Confirmation bias | the tendency to seek out information that confirms one's existing beliefs and ignore information that contradicts them
- Fairness | impartial and just treatment without discrimination.
- Model inference | the process of a model generating an output (response) from a given input (prompt).

## ML Models and Concepts

- Regression models | predict a continuous value (height of the person by their weight)
- Low variance | the model is too simple and does not fit the data well, leading to underfitting
- High variance | the model is too complex and fits the training data too closely, leading to overfitting
- Shapley values | a local interpretability method that explains individual predictions by assigning each feature a contribution score based on its marginal effect on the prediction. This method is useful for understanding the impact of each feature on a specific instance's prediction.
- Partial Dependence Plots (PDP) | a global interpretability method that provides a view of the model’s behavior by illustrating how the predicted outcome changes as a single feature is varied across its range, holding all other features constant. PDPs help understand the overall relationship between a feature and the model output across the entire dataset.
- Risks: Hallucination, Toxicity, Poisoning, Prompt Leaking
  - Hijacking | manipulating an AI system to serve malicious purposes or to misbehave in unintended ways (e.g. phishing, malware, etc.)
  - Jailbreaking | bypassing the built-in restrictions and safety measures of AI systems to unlock restricted functionalities or generate prohibited content (e.g. generating prohibited content, accessing restricted APIs, etc.)
- Bayesian Networks: These models represent probabilistic relationships among variables and provide probabilities for different outcomes
- Dataset
  - training set | 80% of the data | train the model
  - validation set | 10% of the data | tuning hyperparameters
  - test set | 10% of the data | evaluating the final performance on unseen data

## Prompt Parameters

- Temperature (0.0 to 1.0) | Higher temperature -> more random and creative outputs, lower temperature -> more deterministic and focused outputs
- Top K (0 to 500, integer) | Number of most likely candidates that the model considers for the next token. With a low top K setting, like 10, the model will only consider the 10 most probable words for the next word in the sequence. This helps the output be more focused and coherent.
- Top P (0.0 to 1.0, nucleus sampling) | Cumulative probability threshold that limits the token choices. With a low top P setting, like 0.25, the model will only consider words that make up the top 25 percent of the total probability distribution. This helps control the diversity of the model's responses.

## Prompt Engineering Techniques

- Zero-shot prompting | The model is asked to perform a task without any examples or guidance.
- Few-shot prompting | The model is asked to perform a task with a few examples or guidance.
- Chain-of-thought prompting | The model is asked to perform a task by thinking step by step.
- Negative prompting | avoid certain outputs or behaviors when generating content

## Model customization

- Fine-tuning (FM)
  - Labeled data
  - Hyperparameters
    - Epochs - One epoch is one cycle through the entire dataset
    - Learning rate - The amount that values should be changed between epochs.
    - Batch size - The number of records from the dataset that is to be selected for each interval to send to the GPUs for training
- Continued Pre-training (FM)
  - Unlabeled data

## Machine Learning Algorithms

- Reinforcement learning | agent interacting with an environment by taking actions and receiving rewards or penalties, learning a policy to maximize cumulative rewards over time
  - Reinforcement learning from human feedback (RLHF) | machine learning (ML) technique that uses human feedback to optimize ML models to self-learn more efficiently
- Transfer learning | applying knowledge gained from one domain to enhance performance in another related domain
- Supervised learning | using the latest datasets containing both positive and negative customer interactions to improve the chatbot's response quality
  - Linear regression
  - Neural network (e.g. predicting a digit from a handwritten image)
  - Forecasting, DeepAR | predict future values based on past values
  - Classification | classify data into a specific category
  - Decision Trees: Given the same input data, a decision tree will always follow the same path and produce the same output
  - K-Nearest Neighbors (KNN) | classify data based on the similarity of the input data to the nearest neighbors in the training data
- Semi-supervised learning
  - Document classification
  - Fraud identification
  - Sentiment analysis
- Unsupervised learning
  - Association rule learning
  - Clustering, K-Means | group data into clusters based on similarity (e.g. identifying different types of network traffic to predict potential security incidents)
  - Anomaly detection (unsupervised learning algorithm), Random Cut Forest (RCF) | identify outliers in the data
  - Dimensionality reduction | (e.g. it may blur out or crop background features in an image recognition application)
- Incremental training | allows the chatbot to adapt over time by learning from new data without forgetting previously learned information
- Diffusion Model (DALL-E) | create new data by iteratively making controlled random changes to an initial data sample
- Generative adversarial network (GAN) | work by training two neural networks in a competitive manner
- Variational autoencoders (VAE) | encoder and decoder. The encoder neural network maps the input data to a mean and variance for each dimension of the latent space. It generates a random sample from a Gaussian (normal) distribution
- Transformer-based generative AI model (GPT) | builds upon the encoder and decoder concepts of VAEs. Transformer-based models add more layers to the encoder to improve performance on text-based tasks like comprehension, translation, and creative writing.
- Computer Vision models
  - Deep learning | use multiple layers of neurons to learn features from the data
  - Convolutional Neural Networks (CNNs) | extract features from images
  - Recurrent Neural Networks (RNNs) | process sequential data
- Generative Adversarial Networks (GANs) | used for generating new data that resembles the training data, such as creating realistic images, but are not specifically designed for image classification
- Retrieval-Augmented Generation (RAG) | optimizing the output of a large language model, so it references an authoritative knowledge base outside of its training data sources before generating a response.

## Embedding/Transformer models

Embedding models are algorithms trained to encapsulate information into dense representations in a multi-dimensional space. Data scientists use embedding models to enable ML models to comprehend and reason with high-dimensional data.

- Bidirectional Encoder Representations from Transformers (BERT) | capture the contextual meaning of words by looking at both the words that come before and after them
- Principal Component Analysis (PCA) | reducing the dimensions of large datasets to simplify them while retaining most of the variance in the data
- Word2Vec | creates vector representations of words based on their co-occurrence in a given text
- Singular Value Decomposition (SVD) | matrix decomposition method used in various applications like data compression and noise reduction

---

- Data residency | the location where the data is stored and processed
- Data retention policy | the period for which the data is stored and processed
- Data security | the protection of data from unauthorized access, use, disclosure, disruption, modification, or destruction
- Data integrity | ensures the data is accurate, consistent, and unaltered