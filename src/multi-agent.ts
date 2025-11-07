import "dotenv/config";
import * as z from "zod";
import { logger } from "logger";

/** To run the script, use the following command:

tsx src/multi-agent.ts |& tee src/multi-agent.json

 */

import { createAgent, tool } from "langchain";
import { LangGraphRunnableConfig } from "@langchain/langgraph";

const subagent1 = createAgent({
  model: "anthropic:claude-haiku-4-5",
  tools: [],
});

const callSubagent1 = tool(
  async ({ query }: { query: string }, config: LangGraphRunnableConfig) => {
    const result = await subagent1.invoke({
      messages: [{ role: "user", content: query }],
    });
    return result.messages.at(-1)?.text;
  },
  {
    name: "call_subagent1",
    description: "This tool calls subagent1 to get the weather in Tokyo.",
    schema: z.object({
      query: z.string().describe("The query to to send to subagent1."),
    }),
    response_format: z.object({
      weather: z.string().describe("The weather in Tokyo.").optional(),
    }),
  }
);

const agent = createAgent({
  model: "anthropic:claude-haiku-4-5",
  tools: [callSubagent1],
});

const result = await agent.invoke({
  messages: [{ role: "user", content: "What is the weather in Tokyo?" }],
});

logger.info("Multi-agent result", { result });
