import "dotenv/config";
import * as z from "zod";
import { logger } from "logger";

/** To run the script, use the following command:

tsx src/structured-output.ts |& tee src/structured-output.json

 */

import {
  createAgent,
  StructuredOutputParsingError,
  toolStrategy,
} from "langchain";
import { ChatAnthropic } from "@langchain/anthropic";

const ProductReview = z.object({
  rating: z.number().min(1).max(5).optional(),
  sentiment: z.enum(["positive", "negative"]),
  keyPoints: z
    .array(z.string())
    .describe("The key points of the review. Lowercase, 1-3 words each."),
});

const agent = createAgent({
  model: new ChatAnthropic({
    model: "claude-haiku-4-5",
    apiKey: process.env.ANTHROPIC_API_KEY,
  }),
  tools: [],
  responseFormat: toolStrategy(ProductReview, {
    toolMessageContent: "Product review captured and added to database!",
    handleError: (error) => {
      if (
        error instanceof StructuredOutputParsingError &&
        error.errors.includes("rating")
      ) {
        return "Please provide a valid rating between 1-5 and include a comment.";
      }

      return error.message;
    },
  }),
});

const result = await agent.invoke({
  messages: [
    {
      role: "user",
      content:
        "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'",
    },
  ],
});

logger.info("Structured output result", { result });
