import "dotenv/config";
import * as z from "zod";
import { logger } from "logger";

/**
 * MCP Multi-Server Example
 *
 * This script demonstrates how to use MultiServerMCPClient to connect to
 * multiple MCP servers (math and weather) and use their tools with a LangChain agent.
 *
 * PREREQUISITES:
 * -------------
 * 1. Install dependencies:
 *    npm install express cors
 *
 * 2. Start the weather server (in a separate terminal):
 *    tsx src/mcp/weather-server.ts
 *
 *    The weather server must be running on http://localhost:8000/mcp
 *    before running this script.
 *
 * 3. The math server will be automatically launched by MultiServerMCPClient
 *    using stdio transport. No manual startup needed.
 *
 * USAGE:
 * ------
 * To run this script:

tsx src/mcp/mcp.ts |& tee src/mcp/mcp.json

 *
 * The script will:
 * - Connect to the math server via stdio (auto-launched)
 * - Connect to the weather server via SSE (must be running separately)
 * - Use both servers' tools with a LangChain agent
 * - Execute math and weather queries
 */

import { MultiServerMCPClient } from "@langchain/mcp-adapters";
import { ChatAnthropic } from "@langchain/anthropic";
import { createAgent } from "langchain";

const client = new MultiServerMCPClient({
  math: {
    transport: "stdio", // Local subprocess communication via stdin/stdout
    command: "tsx", // Use tsx to run TypeScript directly
    args: ["src/mcp/math-server.ts"],
  },
  weather: {
    transport: "sse", // Server-Sent Events for streaming
    // IMPORTANT: Start the weather server first with: tsx src/mcp/weather-server.ts
    url: "http://localhost:8000/mcp",
  },
});

// Get tools from MCP servers
let tools;
try {
  tools = await client.getTools();
} catch (error: any) {
  logger.error("MCP connection error", {
    error: error?.message,
    serverName: error?.serverName,
    stack: error?.stack,
  });
  if (
    error?.serverName === "weather" ||
    error?.message?.includes("ECONNREFUSED") ||
    error?.message?.includes("404") ||
    error?.message?.includes("Failed to create SSE")
  ) {
    logger.error(
      "❌ Weather server connection failed!\n" +
        "   Please ensure the weather server is running in a separate terminal:\n" +
        "   tsx src/mcp/weather-server.ts\n" +
        "   Then run this script again."
    );
    process.exit(1);
  }
  throw error;
}
const agent = createAgent({
  model: "anthropic:claude-haiku-4-5",
  tools,
});

const mathResponse = await agent.invoke({
  messages: [{ role: "user", content: "what's (3 + 5) x 12?" }],
});

logger.info("Math response", { mathResponse });

const weatherResponse = await agent.invoke({
  messages: [{ role: "user", content: "what is the weather in nyc?" }],
});

logger.info("Weather response", { weatherResponse });
