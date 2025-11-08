import "dotenv/config";
import * as z from "zod";
import { logger } from "logger";

/** To run the script, use the following command:

tsx src/test/strict.test.ts |& tee src/test/strict.test.json

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
    schema: z.object({
      city: z.string(),
    }),
  }
);

const agent = createAgent({
  model: new ChatAnthropic({ model: "claude-haiku-4-5" }),
  tools: [getWeather],
});

const evaluator = createTrajectoryMatchEvaluator({
  trajectoryMatchMode: "strict",
});

async function testWeatherToolCalledStrict() {
  const result = await agent.invoke({
    messages: [new HumanMessage("What's the weather in San Francisco?")],
  });

  const referenceTrajectory = [
    new HumanMessage("What's the weather in San Francisco?"),
    new AIMessage({
      content: "",
      tool_calls: [
        { id: "call_1", name: "get_weather", args: { city: "San Francisco" } },
      ],
    }),
    new ToolMessage({
      content: "It's 75 degrees and sunny in San Francisco.",
      tool_call_id: "call_1",
    }),
    new AIMessage("The weather in San Francisco is 75 degrees and sunny."),
  ];

  const evaluation = await evaluator({
    outputs: result.messages,
    referenceOutputs: referenceTrajectory,
  });

  logger.info("Evaluation", { evaluation });
  assert(evaluation.score === true, "Evaluation score should be true");
}

testWeatherToolCalledStrict();
