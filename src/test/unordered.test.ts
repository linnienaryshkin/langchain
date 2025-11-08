import "dotenv/config";
import * as z from "zod";
import { logger } from "logger";

/** To run the script, use the following command:

tsx src/test/unordered.test.ts |& tee src/test/unordered.test.json

 */

import {
  createAgent,
  tool,
  HumanMessage,
  AIMessage,
  ToolMessage,
} from "langchain";
import { createTrajectoryMatchEvaluator } from "agentevals";
import { assert } from "console";
import { ChatAnthropic } from "@langchain/anthropic";

const getWeather = tool(
  async ({ city }: { city: string }) => {
    return `It's 75 degrees and sunny in ${city}.`;
  },
  {
    name: "get_weather",
    description: "Get weather information for a city.",
    schema: z.object({ city: z.string() }),
  }
);

const getEvents = tool(
  async ({ city }: { city: string }) => {
    return `Concert at the park in ${city} tonight.`;
  },
  {
    name: "get_events",
    description: "Get events happening in a city.",
    schema: z.object({ city: z.string() }),
  }
);

const agent = createAgent({
  model: new ChatAnthropic({ model: "claude-haiku-4-5" }),
  tools: [getWeather, getEvents],
});

const evaluator = createTrajectoryMatchEvaluator({
  trajectoryMatchMode: "unordered",
});

async function testMultipleToolsAnyOrder() {
  const result = await agent.invoke({
    messages: [
      new HumanMessage(
        "What's happening in SF today? What's the weather in SF?"
      ),
    ],
  });

  // Reference shows tools called in different order than actual execution
  const referenceTrajectory = [
    new HumanMessage("What's happening in SF today? What's the weather in SF?"),
    new AIMessage({
      content: "",
      tool_calls: [
        { id: "call_1", name: "get_events", args: { city: "SF" } },
        { id: "call_2", name: "get_weather", args: { city: "SF" } },
      ],
    }),
    new ToolMessage({
      content: "Concert at the park in SF tonight.",
      tool_call_id: "call_1",
    }),
    new ToolMessage({
      content: "It's 75 degrees and sunny in SF.",
      tool_call_id: "call_2",
    }),
    new AIMessage(
      "Today in SF: 75 degrees and sunny with a concert at the park tonight."
    ),
  ];

  const evaluation = await evaluator({
    outputs: result.messages,
    referenceOutputs: referenceTrajectory,
  });

  logger.info("Evaluation", { evaluation });
  assert(evaluation.score === true, "Evaluation score should be true");
}

testMultipleToolsAnyOrder();
