# AWS Exam

## Metrics

- Classification accuracy | the ratio of correct predictions to all predictions (TP + TN) / (TP + TN + FP + FN)
- Recall | the ratio of true positives to all actual positives (TP) / (TP + FN)
- F1 score | precision and recall (TP) / (TP + FP + FN)
- MSE (Mean Squared Error) | the average of the squares of the errors (y - y_pred)^2
- ROUGE-N (Recall-Oriented Understudy for Gisting Evaluation) | evaluate the quality of the generated text by comparing it to the reference text for n-grams
- BERTScore | evaluate the quality of the generated text by comparing it to the reference text
- Perplexity | probability of a model to generate a given sequence of words

## Performance

- Feature engineering | enhances the data by increasing the number of variables in the training dataset
- Data collection | label, ingest, and aggregate data
- Instruction-based fine-tuning | labeled examples that are formatted as prompt-response pairs and that are phrased as instructions

## AWS Services

- Amazon SageMaker AI | build, train, and deploy machine learning models
  - SageMaker Model Cards | documentation of the model
  - SageMaker JumpStart | pre-trained, open source models
  - SageMaker Ground Truth | label data for machine learning models
  - SageMaker Canvas | build machine learning models without writing code
  - SageMaker Studio | IDE
  - SageMaker Clarify | explain the model's decisions
  - SageMaker Pipelines | orchestration
- Amazon Bedrock | pre-trained models, orchestration, MCP
  - Amazon Bedrock Guardrails | guardrails for the model
  - Amazon Bedrock Agents | build agents that can perform tasks such as customer service, data analysis, and more
- Amazon Lex | build chatbots and voice assistants
- Amazon Macie | protect sensitive data in S3
- Amazon Rekognition | image and video analysis
- AWS Config | configuration of AWS resources
- Amazon Polly | text-to-speech (TTS)
- Amazon Transcribe | speech-to-text (STT)
- Amazon Textract | images/documents -> text/data
- Amazon Kendra | search/query documents
- Amazon Comprehend | natural language processing (NLP)
- Amazon Inspector | security and compliance of AWS resources
- AWS Artifact | security and compliance documentation for the AWS Cloud
- Amazon Rekognition | image and video analysis
- AWS Trusted Advisor | recommendations for cost optimization, security, and resilience
- Amazon Q Business | a service that provides a way to ask questions and get answers from a model
- Amazon Personalize | build recommendation systems
- AWS AI Service Cards | documentation of the AI services

## AI concepts

- Overfitting | when a model performs well on training data but fails to generalize to new data.
- Underfitting | when a model is too simple to capture the underlying patterns in the data.
- Explainability | the ability to understand how a model arrives at a prediction.
- Bias | unfair prejudice or preference that favors or disfavors a person or group.
- Fairness | impartial and just treatment without discrimination.

## ML

- Regression models | predict a continuous value (height of the person by their weight)
- Low variance | the model is too simple and does not fit the data

## Promnt Parameters

- Temperature (0.0 to 1.0) | Higher temperature -> more random, lower temperature -> more deterministic
- Top K (0.0 to 1.0) | With a low top p setting, like 0.250, the model will only consider words that make up the top 25 percent of the total probability distribution. This can help the output be more focused and coherent, because the model is limited to choosing from the most probable words given the context.
- Top P (0 to unbounded) | With a low setting, like 10, the model will only consider the 10 most probable words for the next word in the sequence. This can help the output be more focused and coherent, because the model is limited to choosing from the most probable words given the context.

## Algorithms

- Clustering (unsupervised ML method), k-means | group data into clusters based on similarity
- Anomaly detection (unsupervised learning algorithm), Random Cut Forest (RCF) | identify outliers in the data
- Forecasting (supervised ML method), DeepAR | predict future values based on past values
- Classification (supervised ML method)| classify data into a specific category
