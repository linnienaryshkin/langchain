## AI SDLC

1. Design
2. Development
   1. There are several explainability frameworks available, such as SHapley Value Added (SHAP), Layout-Independent Matrix Factorization (LIME), and Counterfactual Explanations, that can help summarize and interpret the decisions made by AI systems. These frameworks can provide insights into the factors that influenced a particular decision and help assess the fairness and consistency of the AI system.
3. Deployment
4. Monitoring
5. Evaluation

### Training

- Underfitted & Overfitted & Balanced | capturing enough features of the data, without capturing noise
- To help overcome bias and variance errors
  - Cross-validation | split the data into training and validation sets
  - Increase data | more data is better, but not always possible
  - Regularization | penalizes extreme weight values
  - Simpler models | simpler model architectures to help with overfitting. If the model is underfitting, the model might be too simple.
  - Dimension reduction (Principal component analysis) | Dimension reduction is an unsupervised machine learning algorithm that attempts to reduce the dimensionality (number of features) within a dataset while still retaining as much information as possible.
  - Stop training early

### Gen AI

Issues:

- Toxicity | the model generates content that is harmful or offensive
- Hallucinations | the model generates content that is not based on the training data
- Intellectual property | the model generates content that is 侵犯他人的知识产权
- Plagiarism and cheating | the model generates content that is 抄袭他人的作品
- Disruption of the nature of work | the model generates content that is disruptive to the nature of work

### Evaluation

- Fairness
- Explainability
- Privacy and security
- Transparency
- Veracity and robustness
- Governance
- Safety
- Controllability

## Pick a model

- Define application use case narrowly
- Choosing a model based on performance (Level of customization, Model size, Inference options, Licensing agreements, Context windows, Latency)
  - Model that performs differently on different datasets
- Choosing a model based on sustainability concerns
  - Value alignment
  - Responsible reasoning skills
  - Appropriate level of autonomy
  - Transparency and accountability
  - Environmental impact - Energy consumption
  - Resources utilization
  - Environmental impact assessment

## Interpretability trade-offs

- Interpretability | the model's ability to be understood by humans

1. Linear regression — high interpretability, low performance
2. Decision tree
3. Logistic regression
4. Naive Bayes
5. K-Nearest Neighbors
6. Support Vector Machine
7. Ensemble Methods
8. Neural Networks — high performance, low interpretability
