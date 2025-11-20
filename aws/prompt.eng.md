# Essentials of Prompt Engineering

- Instructions
- Context
- Input data
- Output indicator

## Inference parameters

- Temperature (0.0 to 1.0) | Higher temperature -> more random, lower temperature -> more deterministic
- Top K (0.0 to 1.0) | With a low top p setting, like 0.250, the model will only consider words that make up the top 25 percent of the total probability distribution. This can help the output be more focused and coherent, because the model is limited to choosing from the most probable words given the context.
- Top P (0 to unbounded) | With a low setting, like 10, the model will only consider the 10 most probable words for the next word in the sequence. This can help the output be more focused and coherent, because the model is limited to choosing from the most probable words given the context.
- Length, maximum length and stop sequences | The model will stop generating text after it reaches the maximum length or after it generates a stop sequence.

## Prompt Engineering Techniques

- Zero-shot prompting | The model is asked to perform a task without any examples or guidance.
- Few-shot prompting | The model is asked to perform a task with a few examples or guidance.
- Chain-of-thought prompting | The model is asked to perform a task by thinking step by step.
