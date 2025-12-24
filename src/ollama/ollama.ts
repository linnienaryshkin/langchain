import "dotenv/config";
import * as z from "zod";
import { logger } from "logger";

/** To run the script, use the following command:

tsx src/ollama/ollama.ts |& tee src/ollama/ollama.json

 */

import { tool } from "langchain";
import { ChatOllama } from "@langchain/ollama";

// const llm = new ChatOllama({
//   model: "gemma3:1b",
//   temperature: 0,
//   maxRetries: 2,
//   // other params...
// });
// const aiMsg = await llm.invoke([
//   [
//     "system",
//     "You are a helpful assistant that translates English to French. Translate the user sentence.",
//   ],
//   ["human", "I love programming."],
// ]);

const weatherTool = tool((_) => "Da weather is weatherin", {
  name: "get_current_weather",
  description: "Get the current weather in a given location",
  schema: z.object({
    location: z.string().describe("The city and state, e.g. San Francisco, CA"),
  }),
});

// Define the model
const llmForTool = new ChatOllama({
  model: "llama3-groq-tool-use",
});

// Bind the tool to the model
const llmWithTools = llmForTool.bindTools([weatherTool]);

const resultFromTool = await llmWithTools.invoke(
  "What's the weather like today in San Francisco? Ensure you use the 'get_current_weather' tool."
);

logger.info("Invoke Results", { resultFromTool });
