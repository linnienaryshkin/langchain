import { Document, Embeddings } from "langchain";

class SimpleMemoryVectorStore {
  private docs: Document[] = [];
  private vectors: number[][] = [];

  constructor(private embeddings: Embeddings) {}

  async addDocuments(documents: Document[]): Promise<string[]> {
    this.docs.push(...documents);
    // Embed all documents
    this.vectors = await Promise.all(
      documents.map((doc) => this.embeddings.embedQuery(doc.pageContent))
    );
    return documents.map((_, i) => `doc-${i}`);
  }

  async similaritySearch(query: string, k: number = 4): Promise<Document[]> {
    const queryVector = await this.embeddings.embedQuery(query);

    // Calculate cosine similarity with all stored vectors
    const similarities = this.vectors.map((vec, i) => ({
      index: i,
      score: this.cosineSimilarity(queryVector, vec),
    }));

    // Sort by similarity and return top k
    return similarities
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .map((item) => this.docs[item.index]);
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }
}

// Usage
import { OpenAIEmbeddings } from "@langchain/openai";

const embeddings = new OpenAIEmbeddings();
const vectorStore = new SimpleMemoryVectorStore(embeddings);

await vectorStore.addDocuments([
  new Document({ pageContent: "Hello world" }),
  new Document({ pageContent: "Goodbye world" }),
]);

const results = await vectorStore.similaritySearch("hello", 1);
console.log(results);

const tool = createRetrieverTool(retriever, {
  name: "retrieve_blog_posts",
  description:
    "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
});
const tools = [tool];

import { StateGraph, START, END } from "@langchain/langgraph";
