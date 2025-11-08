import "dotenv/config";
import * as z from "zod";
import { logger } from "logger";

/** To run the script, use the following command:

tsx src/langgraph/index.ts |& tee src/langgraph/index.json

 */

import {
  MessagesAnnotation,
  StateGraph,
  START,
  END,
} from "@langchain/langgraph";

const mockLlm = (state: typeof MessagesAnnotation.State) => {
  return { messages: [{ role: "ai", content: "hello world" }] };
};

const graph = new StateGraph(MessagesAnnotation)
  .addNode("mock_llm", mockLlm)
  .addEdge(START, "mock_llm")
  .addEdge("mock_llm", END)
  .compile();

const result = await graph.invoke({
  messages: [{ role: "user", content: "hi!" }],
});

logger.info("Result", { result });
