import "dotenv/config";
import * as z from "zod";
import { logger } from "logger";

/** To run the script, use the following command:

tsx src/short-term-memory.ts |& tee src/short-term-memory.json

 */

import {
  BaseMessage,
  createAgent,
  createMiddleware,
  summarizationMiddleware,
  trimMessages,
} from "langchain";
import { MemorySaver, MessagesZodState } from "@langchain/langgraph";

const checkpointer = new MemorySaver();

const customStateSchema = z.object({
  messages: MessagesZodState.shape.messages,
  userId: z.string(),
  preferences: z.record(z.string(), z.any()),
});
/**
 * Save story in a specific format
 */
const stateExtensionMiddleware = createMiddleware({
  name: "StateExtension",
  stateSchema: customStateSchema,
});

/**
 * This function will be called every time before the node that calls LLM
 */
const trimMiddleware = createMiddleware({
  name: "TrimMessages",
  beforeModel: async (request) => {
    const trimmed = await trimMessages(request.messages, {
      strategy: "last",
      maxTokens: 10,
      startOn: "human",
      endOn: ["human", "tool"],
      tokenCounter: (msgs: BaseMessage[]) => msgs.length,
    });
    return request.replace(trimmed);
  },
});

const sumMiddleware = summarizationMiddleware({
  name: "Summarization",
  model: "anthropic:claude-sonnet-4-5",
  maxTokensBeforeSummary: 4000,
  messagesToKeep: 20,
});

const agent = createAgent({
  model: "anthropic:claude-sonnet-4-5",
  tools: [],
  middleware: [
    //
    stateExtensionMiddleware,
    trimMiddleware,
    summarizationMiddleware,
  ] as const,
  checkpointer,
  stateSchema: customStateSchema,
});

const result = await agent.invoke(
  {
    messages: [{ role: "user", content: "hi! i am Bob" }],
    userId: "user_123",
    preferences: { theme: "dark" },
  },
  { configurable: { thread_id: "1" } }
);

logger.info("Short-term memory result", { result });
