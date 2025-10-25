import * as z from "zod";
import { createAgent, tool } from "langchain";
import "dotenv/config";
import { logger } from "logger";

/** To run the script, use the following command:

tsx src/quickstart.ts |& tee src/quickstart.json

 */

const getWeather = tool(
  ({ city }: { city: string }) => `It's always sunny in ${city}!`,
  {
    name: "get_weather",
    description: "Get the weather for a given city",
    schema: z.object({
      city: z.string(),
    }),
  }
);

const agent = createAgent({
  model: "anthropic:claude-sonnet-4-5",
  tools: [getWeather],
});

const result = await await agent.invoke({
  messages: [{ role: "user", content: "What's the weather in Tokyo?" }],
});

logger.info("Quickstart result", { result });
