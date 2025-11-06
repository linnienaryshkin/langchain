#!/usr/bin/env node

/**
 * Math MCP Server
 *
 * This server provides mathematical operations via the Model Context Protocol (MCP)
 * using stdio transport.
 *
 * USAGE:
 * ------
 * This server is designed to be launched by the MultiServerMCPClient in mcp.ts.
 * It communicates via stdio (standard input/output).
 *
 * To test it manually:
 *   tsx src/mcp/math-server.ts |& tee src/mcp/math-server.json
 *
 * The server will read MCP protocol messages from stdin and write responses to stdout.
 *
 * The server provides the following tools:
 * - calculate: Evaluates a mathematical expression
 * - add: Adds two numbers
 * - multiply: Multiplies two numbers
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { logger } from "logger";

const server = new Server(
  {
    name: "math-server",
    version: "0.1.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "add",
        description: "Add two numbers",
        inputSchema: {
          type: "object",
          properties: {
            a: {
              type: "number",
              description: "First number",
            },
            b: {
              type: "number",
              description: "Second number",
            },
          },
          required: ["a", "b"],
        },
      },
      {
        name: "multiply",
        description: "Multiply two numbers",
        inputSchema: {
          type: "object",
          properties: {
            a: {
              type: "number",
              description: "First number",
            },
            b: {
              type: "number",
              description: "Second number",
            },
          },
          required: ["a", "b"],
        },
      },
    ],
  };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  logger.info("Math MCP server: CallToolRequest", { request });

  switch (request.params.name) {
    case "add": {
      const { a, b } = request.params.arguments as { a: number; b: number };
      return {
        content: [
          {
            type: "text",
            text: String(a + b),
          },
        ],
      };
    }
    case "multiply": {
      const { a, b } = request.params.arguments as { a: number; b: number };
      return {
        content: [
          {
            type: "text",
            text: String(a * b),
          },
        ],
      };
    }
    default:
      throw new Error(`Unknown tool: ${request.params.name}`);
  }
});

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  logger.info("Math MCP server running on stdio");
}

main();
