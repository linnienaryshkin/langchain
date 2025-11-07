import "dotenv/config";
import z from "zod";
import { logger } from "logger";

/** To run the script, use the following command:

tsx src/human-in-the-loop.ts |& tee src/human-in-the-loop.json

 */

import {
  createAgent,
  humanInTheLoopMiddleware,
  HumanMessage,
  tool,
} from "langchain";
import {
  Command,
  LangGraphRunnableConfig,
  MemorySaver,
} from "@langchain/langgraph";

const executeSQLTool = tool(
  async (input: string, config: LangGraphRunnableConfig) => {
    config.writer?.(`Executing SQL query: ${input}`);
    return `SQL query executed: ${input}`;
  },
  {
    name: "execute_sql",
    description: "Execute a SQL query",
  }
);

const agent = createAgent({
  model: "anthropic:claude-haiku-4-5",
  tools: [executeSQLTool],
  middleware: [
    humanInTheLoopMiddleware({
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
    }),
  ],
  // Human-in-the-loop requires checkpointing to handle interrupts.
  // In production, use a persistent checkpointer like AsyncPostgresSaver.
  checkpointer: new MemorySaver(),
});

// You must provide a thread ID to associate the execution with a conversation thread,
// so the conversation can be paused and resumed (as is needed for human review).
const config = { configurable: { thread_id: "some_id" } };

// Run the graph until the interrupt is hit.
const result = await agent.invoke(
  {
    messages: [
      new HumanMessage(
        "Delete old records from the database table 'weather_data' older than 30 days"
      ),
    ],
  },
  config
);

logger.info("Human in the loop result 1", { result });

// Resume with approval decision
const result2 = await agent.invoke(
  new Command({
    resume: { decisions: [{ type: "approve" }] }, // or "edit", "reject"
  }),
  config // Same thread ID to resume the paused conversation
);

logger.info("Human in the loop result 2", { result2 });
