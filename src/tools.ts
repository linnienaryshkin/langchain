import "dotenv/config";
import * as z from "zod";
import { logger } from "logger";

/** To run the script, use the following command:

tsx src/tools.ts |& tee src/tools.json

 */

import { ChatAnthropic } from "@langchain/anthropic";
import { createAgent, tool } from "langchain";
import { LangGraphRunnableConfig } from "@langchain/langgraph";

const contextSchema = z.object({
  user_name: z.string(),
});
type ContextSchema = z.infer<typeof contextSchema>;

// const getUserName = tool(
//   (_: unknown, config: { context: ContextSchema }) => {
//     return config.context.user_name;
//   },
//   {
//     name: "get_user_name",
//     description: "Get the user's name.",
//     schema: z.object({}),
//   }
// );

// const agent = createAgent({
//   model: new ChatAnthropic({ model: "claude-sonnet-4-5" }),
//   tools: [getUserName],
//   contextSchema,
// });

// const result = await agent.invoke(
//   {
//     messages: [{ role: "user", content: "What is my name?" }],
//   },
//   {
//     context: { user_name: "John Smith" },
//   }
// );

// logger.info("Tools result", { result });

// ---

const getWeather = tool(
  ({ city }: { city: string }, config: LangGraphRunnableConfig) => {
    config.writer?.(`Looking up data for city: ${city}`);
    config.writer?.(`Acquired data for city: ${city}`);

    return `It's always sunny in ${city}!`;
  },
  {
    name: "get_weather",
    description: "Get weather for a given city.",
    schema: z.object({
      city: z.string(),
    }),
  }
);

const agent = createAgent({
  model: new ChatAnthropic({ model: "claude-sonnet-4-5" }),
  tools: [getWeather],
});

const result = await agent.invoke({
  messages: [{ role: "user", content: "What's the weather in Tokyo?" }],
});

logger.info("Tools result", { result });
