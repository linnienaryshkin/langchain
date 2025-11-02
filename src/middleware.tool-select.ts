import "dotenv/config";
import z from "zod";
import { logger } from "logger";

/** To run the script, use the following command:

tsx src/middleware.tool-select.ts |& tee src/middleware.tool-select.json

 */

import { createAgent, createMiddleware, tool, HumanMessage } from "langchain";
import { ChatAnthropic } from "@langchain/anthropic";

const githubCreateIssue = tool(
  async ({ repo, title }: { repo: string; title: string }) => ({
    url: `https://github.com/${repo}/issues/1`,
    title,
  }),
  {
    name: "github_create_issue",
    description: "Create an issue in a GitHub repository",
    schema: z.object({ repo: z.string(), title: z.string() }),
  }
);

const gitlabCreateIssue = tool(
  async ({ project, title }: { project: string; title: string }) => ({
    url: `https://gitlab.com/${project}/-/issues/1`,
    title,
  }),
  {
    name: "gitlab_create_issue",
    description: "Create an issue in a GitLab project",
    schema: z.object({ project: z.string(), title: z.string() }),
  }
);

const allTools = [githubCreateIssue, gitlabCreateIssue];

const toolSelector = createMiddleware({
  name: "toolSelector",
  contextSchema: z.object({ provider: z.enum(["github", "gitlab"]) }),

  wrapModelCall: (request, handler) => {
    const provider = request.runtime.context.provider;
    const toolName =
      provider === "gitlab" ? "gitlab_create_issue" : "github_create_issue";

    const selectedTools = request.tools.filter((t) => t.name === toolName);
    const modifiedRequest = { ...request, tools: selectedTools };

    return handler(modifiedRequest);
  },
});

const agent = createAgent({
  model: new ChatAnthropic({
    model: "claude-haiku-4-5",
    apiKey: process.env.ANTHROPIC_API_KEY,
  }),
  tools: allTools,
  middleware: [toolSelector],
});

// Invoke with GitHub context
const result = await agent.invoke(
  {
    messages: [
      new HumanMessage(
        "Open an issue titled 'Bug: where are the cats' in the repository `its-a-cats-game`"
      ),
    ],
  },
  {
    context: { provider: "github" },
  }
);

logger.info("Tool select result", { result });
