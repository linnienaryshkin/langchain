import "dotenv/config";
import * as z from "zod";
import { logger } from "logger";

/** To run the script, use the following command:

tsx src/streaming.ts |& tee src/streaming.json

 */

import { createAgent, tool } from "langchain";
import { ChatAnthropic } from "@langchain/anthropic";
import { LangGraphRunnableConfig } from "@langchain/langgraph";

const getWeather = tool(
  async (_input: unknown, config: LangGraphRunnableConfig): Promise<string> => {
    config.writer?.("Looking up weather data...");
    // ... wait for 1 second
    await new Promise((resolve) => setTimeout(resolve, 1000));
    config.writer?.("Weather data found!");
    return `The weather is sunny!`;
  },
  { name: "get_weather" }
);

const agent = createAgent({
  model: new ChatAnthropic({
    apiKey: process.env.ANTHROPIC_API_KEY,
    model: "claude-haiku-4-5",
  }),
  tools: [getWeather],
});

for await (const chunk of await agent.stream(
  { messages: [{ role: "user", content: "what is the weather in sf" }] },
  {
    /**
     * `updates` - emits an event after every agent step
     * `messages` - stream tokens as they are produced by the LLM
     * `custom` - emit custom events (e.g. weather writer)
     */
    streamMode: ["updates", "messages", "custom"],
  }
)) {
  logger.info(`streaming chunk`, { chunk });
}
