1. **Artificial intelligence (AI)** - The ability of a machine to perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language understanding.
2. **Machine learning (ML)** - A subset of AI that uses statistical models to make predictions or decisions based on data.
3. **Deep learning (DL)** - A subset of machine learning that uses artificial neural networks to learn from data.
4. **Generative AI** - A type of AI that uses machine learning to generate new data, such as images, text, or audio.

## Machine Learning

### Training data

> garbage in, garbage out

- **Labeled data** is the most important thing
- **Unlabeled data** is used to train the model

- **Structured data** is data that is organized in a specific format, such as a database.
- **Unstructured data** is data that is not organized in a specific format, such as a photo or a video.

### Algorithms

- **Supervised learning** | The model is trained on labeled data.
  - Classification | the model is trained to classify data into a specific category (Fraud detection, Image classification, etc.)
  - Regression | the model is trained to predict a continuous value (Weather forecasting, Stock price prediction, etc.)
- **Unsupervised learning** | The model is trained on unlabeled data.
  - Clustering | the model is trained to group data into clusters (Customer segmentation, etc.)
  - Dimensionality reduction | the model is trained to reduce the number of features in the data (Meaningful compression, Big data visualization, etc.)
- **Semi-supervised learning** | The model is trained on a combination of labeled and unlabeled data.
- **Reinforcement learning** | The model is trained on a combination of labeled and unlabeled data.

### Inferencing

- Batch inferencing | the process of making predictions or decisions based on the data in a batch.
- Real-time inferencing | the process of making predictions or decisions based on the data in real-time.

## Deep Learning

- Neural networks | prediction and classification based on input
- Computer vision
- Natural language processing

## Generative AI

- Foundation models | large language models that can be fine-tuned for specific tasks

1. Data selection
2. Pre-training
3. Optimization
4. Evaluation
5. Deployment
6. Feedback and continuous improvement

- Large language models (LLMs), transformer architecture
  - Token | basic units of text processed by AI models
  - Embeddings and vector search | represent text as vectors and search for similar vectors
- Diffusion models | generate images from text, forward and backward diffusion
- Multi-modal models | models that can process multiple types of data, such as text, images, and audio
- Generative adversarial networks (GANs) | two competing models, generator and discriminator, that generate and discriminate between real and fake data
- Variational autoencoders (VAEs) | encode data into a latent space and decode it back to the original data

### Optimization

- Prompt engineering | design prompts to guide the model's behavior
- Fine-tuning | train a model on a specific task
- Retrieval-augmented generation (RAG) | use a model to retrieve relevant information from a database and use it to generate a response

## AWS Infrastructure and Technologies

- Amazon SageMaker AI | build, train, and deploy machine learning models

### AWS Services

- Amazon Comprehend | natural language processing (NLP)
- Amazon Translate | Neural machine translation is a form of language translation automation that uses deep learning models to deliver more accurate and more natural-sounding translation than traditional statistical and rule-based translation algorithms
- Amazon Textract | extract text and data from images and documents
- Amazon Lex | build chatbots and voice assistants
- Amazon Polly | text-to-speech (TTS)
- Amazon Transcribe | speech-to-text (STT)
- Amazon Rekognition | image and video analysis
- Amazon Kendra | search and query documents
- Amazon Personalize | build recommendation systems
- AWS DeepRacer | /18th scale race car that gives you an interesting and fun way to get started with reinforcement learning (RL)

### Generative AI

- SageMaker JumpStart | pre-trained models and tools for building and deploying machine learning models
- Amazon Bedrock | a service that provides a way to use pre-trained models from various providers
- Amazon Q | a service that provides a way to ask questions and get answers from a model
- Amazon Q Developer | provides ML–powered code recommendations to accelerate development of C#, Java, JavaScript, Python, and TypeScript applications

## Generative AI Development

1. Defining a Use Case
2. Selecting an FM (foundation model) | Cost, Modality, Latency, Multi-lingual support, Model size, Model complexity, Customization, Input/output length, Responsibility considerations, Deployment and integration
3. Improving the Performance of an FM | Prompt engineering, RAG, Fine-tuning, model from scratch
4. Evaluating an FM | Human evaluation, Benchmark datasets (GLUE, SuperGLUE, SQuAD, WMT), Automated metrics (ROUGE, BLEU, and BERTScore)
5. Deploying the Application
