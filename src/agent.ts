import "dotenv/config";
import * as z from "zod";
import { logger } from "logger";

/** To run the script, use the following command:

tsx src/agent.ts |& tee src/agent.json

 */

import { ChatAnthropic } from "@langchain/anthropic";
import {
  createAgent,
  createMiddleware,
  dynamicSystemPromptMiddleware,
  tool,
  ToolCall,
  ToolMessage,
} from "langchain";
// import { MessagesZodState } from "@langchain/langgraph";

/**
 * Smartest model for complex agents and coding tasks
 * @link https://docs.claude.com/en/docs/about-claude/models/overview
 */
const claudeSonnet45 = new ChatAnthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
  model: "claude-sonnet-4-5",
});

/**
 * Fastest model with near-frontier intelligence
 */
const claudeHaiku45 = new ChatAnthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
  model: "claude-haiku-4-5",
});

const dynamicModelSelection = createMiddleware({
  name: "DynamicModelSelection",
  wrapModelCall: (request, handler) => {
    // Choose model based on conversation complexity
    const messageCount = request.messages.length;

    const model = messageCount > 10 ? claudeSonnet45 : claudeHaiku45;

    logger.debug("Dynamic model selection", { messageCount, model });

    return handler({ ...request, model });
  },
});

const handleToolErrors = createMiddleware({
  name: "HandleToolErrors",
  wrapToolCall: (request, handler) => {
    try {
      return handler(request);
    } catch (error) {
      logger.error("Tool error", { error });

      // Return a custom error message to the model
      return new ToolMessage({
        content: `Tool error: Please check your input and try again. (${error})`,
        tool_call_id: request.toolCall.id!,
      });
    }
  },
});

const contextSchema = z.object({
  userRole: z.enum(["expert", "beginner"]),
});

const dynamicSystemPrompt = dynamicSystemPromptMiddleware<
  z.infer<typeof contextSchema>
>((_state, runtime) => {
  const userRole = runtime?.context?.userRole ?? "user";
  const basePrompt = "You are a helpful assistant.";

  logger.debug("Dynamic system prompt", { userRole });

  if (userRole === "expert") {
    return `${basePrompt} Provide detailed technical responses.`;
  } else if (userRole === "beginner") {
    return `${basePrompt} Explain concepts simply and avoid jargon.`;
  }
  return basePrompt;
});

// ---

const search = tool(
  ({ query }: { query: string }) => {
    logger.debug("Search tool", { query });
    return `Results for: ${query}`;
  },
  {
    name: "search",
    description: "Search for information",
    schema: z.object({
      query: z.string().describe("The query to search for"),
    }),
  }
);

const getWeather = tool(
  ({ location }: { location: string }) => {
    logger.debug("Get weather tool", { location });
    return `Weather in ${location}: Sunny, 72°F`;
  },
  {
    name: "get_weather",
    description: "Get weather information for a location",
    schema: z.object({
      location: z.string().describe("The location to get weather for"),
    }),
  }
);

// ---

const agent = createAgent({
  model: claudeHaiku45, // Base model (used when messageCount ≤ 10)
  tools: [search, getWeather],
  contextSchema,
  middleware: [
    dynamicModelSelection,
    handleToolErrors,
    dynamicSystemPrompt,
  ] as const,
  //   responseFormat: z.object({
  //     result: z.string().describe("The result of the agent's task"),
  //     temperature: z.number().describe("The temperature of the agent's response"),
  //   }),
  //   stateSchema: z.object({
  //     messages: MessagesZodState.shape.messages,
  //     userPreferences: z.record(z.string(), z.string()),
  //   }),
});

// The system prompt will be set dynamically based on context
const stream = await agent.stream(
  {
    messages: [{ role: "user", content: "Give me a weather report for Tokyo" }],
  },
  { context: { userRole: "expert" } }
);

for await (const chunk of stream) {
  if (chunk?.content) {
    logger.info(`Agent: ${chunk.content}`);
  } else if (chunk?.tool_calls) {
    const toolCallNames = chunk.tool_calls.map((tc: ToolCall) => tc.name);

    logger.info(`Calling tools: ${toolCallNames.join(", ")}`);
  }

  logger.debug("Agent stream chunk", { chunk });
}
