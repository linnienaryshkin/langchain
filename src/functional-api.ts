import "dotenv/config";
import * as z from "zod";
import { logger } from "logger";

/** To run the script, use the following command:

tsx src/functional-api.ts |& tee src/functional-api.json

 */

import {
  MemorySaver,
  entrypoint,
  task,
  interrupt,
  Command,
  LangGraphRunnableConfig,
} from "@langchain/langgraph";

const writeEssay = task("writeEssay", async (topic: string) => {
  // A placeholder for a long-running task.
  await new Promise((resolve) => setTimeout(resolve, 1000));
  return `An essay about topic: ${topic}`;
});

const workflow = entrypoint(
  { checkpointer: new MemorySaver(), name: "workflow" },
  async (topic: string) => {
    logger.info("Topic", { topic });

    const essay = await writeEssay(topic);
    logger.info("Essay", { essay });

    const isApproved = interrupt({
      // Any json-serializable payload provided to interrupt as argument.
      // It will be surfaced on the client side as an Interrupt when streaming data
      // from the workflow.
      essay, // The essay we want reviewed.
      // We can add any additional information that we need.
      // For example, introduce a key called "action" with some instructions.
      action: "Please approve/reject the essay",
    });
    logger.info("Is approved", { isApproved });

    return {
      essay, // The essay that was generated
      isApproved, // Response from HIL
    };
  }
);

const config = {
  configurable: { thread_id: crypto.randomUUID() },
} as Partial<LangGraphRunnableConfig>;

const result1 = await workflow.invoke("cat", config);
logger.info("Result 1", { result1 });

// ------------------------------------------------------------

// Get review from a user (e.g., via a UI)
// In this case, we're using a bool, but this can be any json-serializable value.
const humanReview = true;

const result2 = await workflow.invoke(
  new Command({ resume: humanReview }),
  config
);
logger.info("Result 2", { result2 });
