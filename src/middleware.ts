import "dotenv/config";
import z from "zod";
import { logger } from "logger";

/** To run the script, use the following command:

tsx src/middleware.ts |& tee src/middleware.json

 */

import {
  createAgent,
  humanInTheLoopMiddleware,
  HumanMessage,
  summarizationMiddleware,
  AIMessage,
  SystemMessage,
  tool,
  anthropicPromptCachingMiddleware,
  modelCallLimitMiddleware,
  toolCallLimitMiddleware,
  modelFallbackMiddleware,
  piiRedactionMiddleware,
  createMiddleware,
} from "langchain";
import {
  Command,
  LangGraphRunnableConfig,
  MemorySaver,
} from "@langchain/langgraph";

const deleteOldRecords = tool(
  async ({ input }: { input: string }, config: LangGraphRunnableConfig) => {
    config.writer?.(`Deleting old records... ${input}`);
    return `Old records deleted`;
  },
  {
    name: "delete_old_records",
    description: "Delete old records from the database",
  }
);

const deleteOldRecordsLimit = toolCallLimitMiddleware({
  toolName: "delete_old_records",
  threadLimit: 5,
  runLimit: 2,
});

const kittySummarization = summarizationMiddleware({
  model: "anthropic:claude-haiku-4-5",
  maxTokensBeforeSummary: 4000, // Trigger summarization at 4000 tokens
  messagesToKeep: 20, // Keep last 20 messages after summary
  summaryPrompt:
    "Summarize the conversation, so a kitty could understand it (use kitty language)", // Optional
});

const sqlHumanInTheLoop = humanInTheLoopMiddleware({
  interruptOn: {
    write_file: true, // All decisions (approve, edit, reject) allowed
    execute_sql: {
      allowedDecisions: ["approve", "reject"],
      // No editing allowed
      description: "🚨 SQL execution requires DBA approval",
    },
    // Safe operation, no approval needed
    read_data: false,
  },
  // Prefix for interrupt messages - combined with tool name and args to form the full message
  // e.g., "Tool execution pending approval: execute_sql with query='DELETE FROM...'"
  // Individual tools can override this by specifying a "description" in their interrupt config
  descriptionPrefix: "Tool execution pending approval",
});

const fiveMinutesCache = anthropicPromptCachingMiddleware({ ttl: "5m" });

const modelCallLimit = modelCallLimitMiddleware({
  threadLimit: 10, // Max 10 calls per thread (across runs)
  runLimit: 5, // Max 5 calls per run (single invocation)
  exitBehavior: "end", // Or "error" to throw exception
});

const modelFallback = modelFallbackMiddleware(
  "anthropic:claude-haiku-4-5", // Try first on error
  "anthropic:claude-sonnet-4-5" // Then this
);

const piiRedaction = piiRedactionMiddleware({
  piiType: "email",
  strategy: "redact",
  applyToInput: true,
});

const createRetryMiddleware = (maxRetries: number = 3) => {
  return createMiddleware({
    name: "RetryMiddleware",
    wrapModelCall: (request, handler) => {
      for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
          return handler(request);
        } catch (e) {
          if (attempt === maxRetries - 1) {
            throw e;
          }
          console.log(`Retry ${attempt + 1}/${maxRetries} after error: ${e}`);
        }
      }
      throw new Error("Unreachable");
    },
  });
};

const earlyExitMiddleware = createMiddleware({
  name: "EarlyExitMiddleware",
  beforeModel: (state) => {
    // Check some condition
    if (state.messages.length > 10) {
      return {
        messages: [new AIMessage("Exiting early due to condition.")],
        jumpTo: "end",
      };
    }
    return;
  },
});

// ------------------------------------------------------------

const agent = createAgent({
  model: "anthropic:claude-haiku-4-5",
  tools: [deleteOldRecords],
  middleware: [
    kittySummarization,
    fiveMinutesCache,
    modelCallLimit,
    modelFallback,
    piiRedaction,
    //
    sqlHumanInTheLoop,
    deleteOldRecordsLimit,
    createRetryMiddleware(3),
    earlyExitMiddleware,
  ],
  // Human-in-the-loop requires checkpointing to handle interrupts.
  // In production, use a persistent checkpointer like AsyncPostgresSaver.
  checkpointer: new MemorySaver(),
});

const result = await agent.invoke(
  {
    messages: [
      new SystemMessage("You are a weather expert"),
      new HumanMessage("What is the weather in Tokyo?"),
      new AIMessage(
        "I don't have access to real-time weather data, so I can't tell you the current weather in Tokyo."
      ),
      new HumanMessage("My email is linnie@example.com"),
      new HumanMessage(
        "Delete weather data older than 30 days from the database table 'weather_data'"
      ),
    ],
  },
  { configurable: { thread_id: "1" } }
);

logger.info("Middleware result", { result });

// await agent.invoke(
//   new Command({
//     resume: { decisions: [{ type: "reject" }] }, // or "edit", "reject"
//   }),
//   // Same thread ID to resume the paused conversation
//   { configurable: { thread_id: "1" } }
// );

// logger.info("Middleware result after rejection", { result });
