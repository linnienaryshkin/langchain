import "dotenv/config";
import * as z from "zod";
import { logger } from "logger";

/** To run the script, use the following command:

tsx src/messages.ts |& tee src/messages.json

 */

import {
  initChatModel,
  HumanMessage,
  SystemMessage,
  AIMessage,
  ToolMessage,
  AIMessageChunk,
} from "langchain";

const agent = await initChatModel("claude-sonnet-4-5", {
  modelProvider: "anthropic",
  apiKey: process.env.ANTHROPIC_API_KEY,
  /**
   * If an application outside of LangChain needs access to the standard content block representation,
   * you can opt-in to storing content blocks in message content
   */
  outputVersion: "v1",
});

const messages = [
  /** Tells the model how to behave and provide context for interactions */
  new SystemMessage("You are a weather expert"),
  /** Represents user input and interactions with the model */
  new HumanMessage({
    content: [
      { type: "text", text: "What's the weather in San Francisco?" },
      // {
      //   type: "image",
      //   url: "https://png.pngtree.com/png-clipart/20200225/original/pngtree-image-of-cute-radish-vector-or-color-illustration-png-image_5274337.jpg",
      // },
    ],
  }),
  /** Represents the outputs of the model, including text content, tool calls, and metadata */
  new AIMessage({
    content: "Calling tool: get_weather",
    tool_calls: [
      {
        name: "get_weather",
        args: { location: "San Francisco" },
        id: "call_tool_014K3VnnZWkLFc3n59gq3d9y",
      },
    ],
  }),
  /** Represents the outputs of tool calls */
  new ToolMessage({
    content: "Sunny, 72°F",
    tool_call_id: "call_tool_014K3VnnZWkLFc3n59gq3d9y",
  }),
];
const stream = await agent.stream(messages);

let finalChunk: AIMessageChunk | undefined;
for await (const chunk of stream) {
  finalChunk = finalChunk ? finalChunk.concat(chunk) : chunk;
}

logger.info("Response", { finalChunk });

// ---

// const response = await agent.invoke([
//   new HumanMessage({
//     content: [
//       { type: "text", text: "Describe the content of this image." },
//       {
//         type: "image",
//         source_type: "url",
//         url: "https://png.pngtree.com/png-clipart/20200225/original/pngtree-image-of-cute-radish-vector-or-color-illustration-png-image_5274337.jpg",
//       },
//     ],
//   }),
// ]);

// logger.info("Response", { response });
