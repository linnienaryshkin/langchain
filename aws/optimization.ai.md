## RAG (Retrieval-Augmented Generation)

- Vector embeddings | similarity search are k-nearest neighbors (k-NN) | OpenSearch
- Agents | Multi-task, Intermediary operations, Actions launch, Feedback integration
- Human evaluation, Benchmark datasets

## Fine-tuning

- Instruction tuning | retraining the model on a new dataset that consists of prompts followed by the desired outputs
- Reinforcement learning from human feedback (RLHF) | initially trained using supervised learning to predict human-like responses -> reinforcement learning process, where a reward model built from human feedback guides the model toward generating more preferable outputs
- Adapting models for specific domains || Transfer learning || Continuous pretraining
- Data curation, Labeling, Governance and compliance, Representativeness and bias checking, Feedback integration
- Evaluation
  - ROUGE (ROUGE-N, ROUGE-L) | Recall-Oriented Understudy for Gisting Evaluation | evaluate the quality of the generated text by comparing it to the reference text
  - BLEU (Bilingual Evaluation Understudy) | evaluate the quality of the generated text by comparing it to the reference text
  - BERTScore | evaluate the quality of the generated text by comparing it to the reference text

| Feature                            | BLEU                                         | ROUGE                                   | BERTScore                                     |
| ---------------------------------- | -------------------------------------------- | --------------------------------------- | --------------------------------------------- |
| **Primary Focus**                  | Precision                                    | Recall                                  | Semantic Similarity                           |
| **Matching Type**                  | Exact n-gram matching                        | Exact n-gram matching                   | Contextual embedding similarity               |
| **Output Range**                   | 0-100 (or 0-1)                               | 0-1                                     | -1 to 1 (typically 0-1)                       |
| **Common Variants**                | BLEU-1, BLEU-4                               | ROUGE-N, ROUGE-L, ROUGE-W               | BERTScore F1, Precision, Recall               |
| **Best Use Case**                  | Machine translation                          | Text summarization                      | General text generation, paraphrase detection |
| **Handles Synonyms**               | ❌ No                                        | ❌ No                                   | ✅ Yes                                        |
| **Handles Paraphrasing**           | ❌ No                                        | ❌ No                                   | ✅ Yes                                        |
| **Computation Speed**              | ⚡ Very fast                                 | ⚡ Very fast                            | 🐌 Slow (requires neural model)               |
| **Multiple References**            | ✅ Supported                                 | ✅ Supported                            | ✅ Supported                                  |
| **Brevity/Length Penalty**         | ✅ Yes (brevity penalty)                     | ⚠️ Varies by variant                    | ❌ No explicit penalty                        |
| **Correlates with Human Judgment** | Moderate                                     | Moderate                                | Strong                                        |
| **Year Introduced**                | 2002                                         | 2004                                    | 2019                                          |
| **Typical Score Interpretation**   | Higher = better overlap                      | Higher = better coverage                | Higher = more semantically similar            |
| **Main Weakness**                  | Misses semantics, rewards exact matches only | Misses semantics, can reward repetition | Computationally expensive, can be too lenient |
