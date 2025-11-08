import "dotenv/config";
import * as z from "zod";
import { logger } from "logger";

/** To run the script, use the following command:

tsx src/test/llm.test.ts |& tee src/test/llm.test.json

 */

import { createAgent } from "langchain";
import { tool, HumanMessage } from "langchain";
import {
  createTrajectoryLLMAsJudge,
  TRAJECTORY_ACCURACY_PROMPT,
} from "agentevals";
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

const agent = createAgent({
  model: new ChatAnthropic({ model: "claude-haiku-4-5" }),
  tools: [getWeather],
});

const evaluator = createTrajectoryLLMAsJudge({
  model: "claude-sonnet-4-5",
  prompt: TRAJECTORY_ACCURACY_PROMPT,
});

async function testTrajectoryQuality() {
  const result = await agent.invoke({
    messages: [new HumanMessage("What's the weather in Seattle?")],
  });

  const evaluation = await evaluator({
    outputs: result.messages,
  });

  logger.info("Evaluation", { evaluation });
  assert(evaluation.score === true, "Evaluation score should be true");
}

testTrajectoryQuality();
