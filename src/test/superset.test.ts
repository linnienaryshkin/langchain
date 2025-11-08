import "dotenv/config";
import * as z from "zod";
import { logger } from "logger";

/** To run the script, use the following command:

tsx src/test/superset.test.ts |& tee src/test/superset.test.json

 */

import {
  tool,
  createAgent,
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

const getDetailedForecast = tool(
  async ({ city }: { city: string }) => {
    return `Detailed forecast for ${city}: sunny all week.`;
  },
  {
    name: "get_detailed_forecast",
    description: "Get detailed weather forecast for a city.",
    schema: z.object({ city: z.string() }),
  }
);

const agent = createAgent({
  model: new ChatAnthropic({ model: "claude-haiku-4-5" }),
  tools: [getWeather, getDetailedForecast],
});

const evaluator = createTrajectoryMatchEvaluator({
  trajectoryMatchMode: "superset",
});

async function testAgentCallsRequiredToolsPlusExtra() {
  const result = await agent.invoke({
    messages: [new HumanMessage("What's the weather in Boston?")],
  });

  // Reference only requires getWeather, but agent may call additional tools
  const referenceTrajectory = [
    new HumanMessage("What's the weather in Boston?"),
    new AIMessage({
      content: "",
      tool_calls: [
        { id: "call_1", name: "get_weather", args: { city: "Boston" } },
      ],
    }),
    new ToolMessage({
      content: "It's 75 degrees and sunny in Boston.",
      tool_call_id: "call_1",
    }),
    new AIMessage("The weather in Boston is 75 degrees and sunny."),
  ];

  const evaluation = await evaluator({
    outputs: result.messages,
    referenceOutputs: referenceTrajectory,
  });

  logger.info("Evaluation", { evaluation });
  assert(evaluation.score === true, "Evaluation score should be true");
}

testAgentCallsRequiredToolsPlusExtra();
