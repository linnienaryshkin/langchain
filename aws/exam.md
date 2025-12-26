# AWS Exam

## AWS Services

- Amazon SageMaker AI | build, train, and deploy machine learning models
  - SageMaker Model Cards | documentation of the model
  - SageMaker JumpStart | pre-trained, open source models
  - SageMaker Ground Truth | label data for machine learning models
  - SageMaker Canvas | build machine learning models without writing code
  - SageMaker Studio | IDE for machine learning
  - SageMaker Clarify | explain the model's decisions and detect bias
  - SageMaker Pipelines | orchestration and automation of ML workflows
  - SageMaker Model Registry | catalog, version, and manage ML models
  - SageMaker Feature Store | centralized repository for storing and sharing ML features
  - SageMaker Data Wrangler | visual interface to import, clean, transform, and analyze data
- Amazon Bedrock | pre-trained models, orchestration, MCP
  - Amazon Bedrock Guardrails | guardrails for the model
  - Amazon Bedrock Agents | build agents that can perform tasks such as customer service, data analysis, and more
- Amazon Q
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

- Amazon Macie | protect sensitive data in S3
- AWS Config | configuration of AWS resources
- AWS Artifact | security and compliance documentation for the AWS Cloud
- AWS Trusted Advisor | recommendations for cost optimization, security, and resilience

## Metrics

- Classification accuracy | the ratio of correct predictions to all predictions (TP + TN) / (TP + TN + FP + FN)
- Precision | the ratio of true positives to all predicted positives (TP) / (TP + FP)
- Recall | the ratio of true positives to all actual positives (TP) / (TP + FN)
- F1 score | harmonic mean of precision and recall: 2 * (precision * recall) / (precision + recall)
- MSE (Mean Squared Error) | the average of the squares of the errors (y - y_pred)^2
- Confusion matrix | evaluate the performance of classification models by displaying the number of true positives, true negatives, false positives, and false negatives
- Correlation matrix | measures the statistical correlation between different variables or features in a dataset, typically used to understand the relationships between continuous variables
- Root Mean Squared Error (RMSE) | measure the average error in regression models by calculating the square root of the average squared differences between predicted and actual values
- Mean Absolute Error (MAE) | measures the average magnitude of errors in a set of predictions without considering their direction
- ROUGE-N (Recall-Oriented Understudy for Gisting Evaluation) | evaluate the quality of the generated text by comparing it to the reference text for n-grams
- BERTScore | evaluate the quality of the generated text by comparing it to the reference text
- Perplexity | probability of a model to generate a given sequence of words
- BLEU (Bilingual Evaluation Understudy) | evaluate the quality of text that has been machine-translated by comparing it with one or more reference translations

## Data Preparation and Training

- Feature engineering | enhances the data by increasing the number of variables in the training dataset
- Data collection | label, ingest, and aggregate data
- Instruction-based fine-tuning | labeled examples that are formatted as prompt-response pairs and that are phrased as instructions

## AI concepts

- Overfitting | when a model performs well on training data but fails to generalize to new data.
- Underfitting | when a model is too simple to capture the underlying patterns in the data.
- Explainability | the ability to understand how a model arrives at a prediction.
- Bias | unfair prejudice or preference that favors or disfavors a person or group.
- Fairness | impartial and just treatment without discrimination.
- Model inference | the process of a model generating an output (response) from a given input (prompt).

## ML Models and Concepts

- Regression models | predict a continuous value (height of the person by their weight)
- Low variance | the model is too simple and does not fit the data well, leading to underfitting
- High variance | the model is too complex and fits the training data too closely, leading to overfitting

## Prompt Parameters

- Temperature (0.0 to 1.0) | Higher temperature -> more random and creative outputs, lower temperature -> more deterministic and focused outputs
- Top K (0 to 500, integer) | Number of most likely candidates that the model considers for the next token. With a low top K setting, like 10, the model will only consider the 10 most probable words for the next word in the sequence. This helps the output be more focused and coherent.
- Top P (0.0 to 1.0, nucleus sampling) | Cumulative probability threshold that limits the token choices. With a low top P setting, like 0.25, the model will only consider words that make up the top 25 percent of the total probability distribution. This helps control the diversity of the model's responses.

## Fine-tuning

- Hyperparameters
  - Epochs - One epoch is one cycle through the entire dataset
  - Learning rate - The amount that values should be changed between epochs.
  - Batch size - The number of records from the dataset that is to be selected for each interval to send to the GPUs for training

## Algorithms

- Clustering (unsupervised ML method), k-means | group data into clusters based on similarity
- Anomaly detection (unsupervised learning algorithm), Random Cut Forest (RCF) | identify outliers in the data
- Forecasting (supervised ML method), DeepAR | predict future values based on past values
- Classification (supervised ML method) | classify data into a specific category
- Reinforcement learning | agent interacting with an environment by taking actions and receiving rewards or penalties, learning a policy to maximize cumulative rewards over time
- Transfer learning | applying knowledge gained from one domain to enhance performance in another related domain
- Supervised learning | using the latest datasets containing both positive and negative customer interactions to improve the chatbot's response quality
- Incremental training | allows the chatbot to adapt over time by learning from new data without forgetting previously learned information
- Diffusion Model | create new data by iteratively making controlled random changes to an initial data sample
- Generative adversarial network (GAN) | work by training two neural networks in a competitive manner
- Variational autoencoders (VAE) | encoder and decoder. The encoder neural network maps the input data to a mean and variance for each dimension of the latent space. It generates a random sample from a Gaussian (normal) distribution

## Embedding models

Embedding models are algorithms trained to encapsulate information into dense representations in a multi-dimensional space. Data scientists use embedding models to enable ML models to comprehend and reason with high-dimensional data.

- Bidirectional Encoder Representations from Transformers (BERT) | capture the contextual meaning of words by looking at both the words that come before and after them
- Principal Component Analysis (PCA) | reducing the dimensions of large datasets to simplify them while retaining most of the variance in the data
- Word2Vec | creates vector representations of words based on their co-occurrence in a given text
- Singular Value Decomposition (SVD) | matrix decomposition method used in various applications like data compression and noise reduction
