import "dotenv/config";
import * as z from "zod";
import { logger } from "logger";

/** To run the script, use the following command:

tsx src/models.ts |& tee src/models.json

 */

import {
  AIMessage,
  HumanMessage,
  initChatModel,
  SystemMessage,
  tool,
} from "langchain";

/**
 * @link https://docs.langchain.com/oss/javascript/integrations/chat#anthropic
 */
const model = await initChatModel("claude-haiku-4-5", {
  modelProvider: "anthropic",
  apiKey: process.env.ANTHROPIC_API_KEY,
  // The maximum time (in seconds) to wait for a response from the model before canceling the request.
  temperature: 0.7,
  // The maximum time (in seconds) to wait for a response from the model before canceling the request.
  timeout: 30,
  // Limits the total number of tokens in the response, effectively controlling how long the output can be.
  max_tokens: 1000,
});

// const conversation = [
//   new SystemMessage(
//     "You are a helpful assistant that translates English to French."
//   ),
//   new HumanMessage("Translate: I love programming."),
//   new AIMessage("J'adore la programmation."),
//   new HumanMessage("Translate: I love building applications."),
// ];
// const stream = await model.stream(conversation);
// for await (const chunk of stream) {
//   logger.info("Stream conversation chunk", { chunk });
// }

// ---

// const responses = await model.batch(
//   [
//     "Why do parrots have colorful feathers?",
//     "How do airplanes fly?",
//     "What is quantum computing?",
//     "Why do parrots have colorful feathers?",
//     "How do airplanes fly?",
//     "What is quantum computing?",
//   ],
//   {
//     tools: [],
//   },
//   {
//     maxConcurrency: 5, // Limit to 5 parallel calls
//   }
// );
// for (const response of responses) {
//   logger.info("Batch response", { response });
// }

// ---

// const getWeather = tool(
//   ({ location }: { location: string }) => {
//     logger.info("Get weather tool", { location });
//     return `It's sunny in ${location}.`;
//   },
//   {
//     name: "get_weather",
//     description: "Get the weather at a location.",
//     schema: z.object({
//       location: z.string().describe("The location to get the weather for"),
//     }),
//   }
// );
// const modelWithTools = model.bindTools([getWeather], { toolChoice: "any" });
// const response = await modelWithTools.invoke("What's the weather in Tokyo?");
// logger.info("Response with tools", { responseWithTools });

// ---

// const Movie = z.object({
//   title: z.string().describe("The title of the movie"),
//   year: z.number().describe("The year the movie was released"),
//   director: z.string().describe("The director of the movie"),
//   rating: z.number().describe("The movie's rating out of 10"),
//   cast: z.array(
//     z.object({
//       name: z.string(),
//       role: z.string(),
//     })
//   ),
// });

// const modelWithStructure = model.withStructuredOutput(Movie, {
//   includeRaw: true,
// });

// const response = await modelWithStructure.invoke(
//   "Provide details about the movie Inception"
// );
// logger.info("Response with structure", { response });

// ---

// Yeah, very poor TS support so far...
const response = await model.invoke("Tell me a joke", {
  runName: "joke_generation", // Custom name for this run
  tags: ["humor", "demo"], // Tags for categorization
  metadata: { user_id: "123" }, // Custom metadata
  callbacks: [
    (event: any) => {
      logger.info("Callback event", { event });
    },
  ], // Callback handlers
});

logger.info("Response", { response });
