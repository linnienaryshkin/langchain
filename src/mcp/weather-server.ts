#!/usr/bin/env node

/**
 * Weather MCP Server
 *
 * This server provides weather information via the Model Context Protocol (MCP)
 * using Server-Sent Events (SSE) transport over HTTP.
 *
 * USAGE:
 * ------
 * To launch the weather server:

tsx src/mcp/weather-server.ts |& tee src/mcp/weather-server.json

 *
 * The server will start on http://localhost:8000/mcp
 *
 * Make sure this server is running before executing mcp.ts, as it expects
 * the weather server to be available at http://localhost:8000/mcp
 *
 * The server provides the following tools:
 * - get_weather: Gets weather information for a given location
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import express, { Request, Response } from "express";
import cors from "cors";
import { logger } from "logger";

const app = express();
app.use(cors());
app.use(express.json());

const server = new Server(
  {
    name: "weather-server",
    version: "0.1.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

server.setRequestHandler(ListToolsRequestSchema, async () => {
  logger.info("Weather MCP server: ListToolsRequest");

  return {
    tools: [
      {
        name: "get_weather",
        description: "Get weather for location",
        inputSchema: {
          type: "object",
          properties: {
            location: {
              type: "string",
              description: "Location to get weather for",
            },
          },
          required: ["location"],
        },
      },
    ],
  };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  logger.info("Weather MCP server: CallToolRequest", { request });

  switch (request.params.name) {
    case "get_weather": {
      const { location } = request.params.arguments as { location: string };
      return {
        content: [
          {
            type: "text",
            text: `It's always sunny in ${location}`,
          },
        ],
      };
    }
    default:
      throw new Error(`Unknown tool: ${request.params.name}`);
  }
});

app.get("/mcp", async (req: Request, res: Response) => {
  const transport = new SSEServerTransport("/mcp", res);
  await server.connect(transport);
});

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
  logger.info("Weather MCP server running", { PORT });
});
