import "dotenv/config";
import * as z from "zod";
import { logger } from "logger";

/** To run the script, use the following command:

tsx src/rag/index.ts |& tee src/rag/index.json

 */

import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

const urls = [
  "https://lilianweng.github.io/posts/2023-06-23-agent/",
  "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
  "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
];

const docs = await Promise.all(
  urls.map((url) =>
    fetch(url)
      .then((res) => res.text())
      .then((text) => ({ url, text }))
  )
);

logger.info("Docs", { docs });

const docsList = docs.flat();

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 50,
});
const docSplits = await textSplitter.splitDocuments(docsList);

logger.info("Doc splits", { docSplits });

import { InMemoryVectorStore } from "@langchain/core/vectorstores";

import { OpenAIEmbeddings } from "@langchain/openai";

const vectorStore = await MemoryVectorStore.fromDocuments(
  docSplits,
  new OpenAIEmbeddings()
);

const retriever = vectorStore.asRetriever();
