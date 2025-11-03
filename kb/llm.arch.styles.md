# [Levels of LLM Applications](https://www.vitalijneverkevic.com/six-levels-of-llm-applications/)

There are various flavours of the solutions with LLM, which can be distinguished by how many of the LLM capabilities are utilized. From bottom to the top in the pyramid, we have solution complexity as well as the number of problems that can be solved at any particular style.

## QUESTIONS ANSWERING

At its core, the question-answering architecture involves setting up an LLM to handle specific user inquiries. This setup treats each query as a separate, standalone interaction. The model receives a question, processes it, and returns the most accurate and relevant answer based on the knowledge it has learned during its training phase. There's no need for the model to remember past interactions; each question is processed fresh, ensuring that the responses are generated based solely on the information presented at that moment.

## CHAT BOTS

In the chatbot architecture, LLMs are employed to handle diverse user interactions ranging from simple queries to complex conversations. Each interaction with the chatbot is treated in the context of the previous dialogue. The LLM processes the user's input, applies its trained knowledge base, and generates a response that is both contextually relevant and informed by the data it has been trained on.

## [RAG (Retrieval-Augmented Generation)](https://github.com/NirDiamant/RAG_Techniques)

At its core, the RAG architecture involves a two-step process:

1. Retrieval: The first component, typically a dense vector search engine, quickly sifts through large datasets to find relevant information based on the query input. This part is crucial for ensuring that the generation model has access to the most pertinent and factual data.
2. Generation: The second component is a generative model, like a Transformer-based neural network. It takes the retrieved documents as additional context and generates a response that integrates this information. This process ensures that the output is not only relevant but also fluent and coherent.
3. This architecture effectively bridges the gap between traditional search engines and AI conversational models, leveraging the strengths of both to enhance the overall quality and relevance of the output.

## AGENTS

At its core, the Agentic approach involves deploying AI systems that possess a significant degree of agency, meaning they are not just passive tools but active decision-makers. These systems are designed to perceive their environments, interpret data in real-time, learn from interactions, and make autonomous decisions based on a combination of pre-trained knowledge and ongoing learning. Unlike traditional models that respond based on static datasets, Agentic AI systems continually adapt their behaviors and strategies to meet evolving conditions and requirements.

## LLM OS

The concept of a Large Language Model Operating System (LLM OS) is a forward-thinking integration of large language models (LLMs) like GPT or BERT into the core functionalities of digital systems, essentially making these AI models central to processing and operational capabilities. This approach treats AI not merely as an added feature but as an integral component of system architecture, which addresses various systemic and operational challenges.
