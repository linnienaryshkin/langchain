import "dotenv/config";
import * as z from "zod";
import { logger } from "logger";

/** To run the script, use the following command:

tsx src/long-term-memory.ts |& tee src/long-term-memory.json

 */

import { createAgent, Runtime, tool } from "langchain";
import { InMemoryStore, LangGraphRunnableConfig } from "@langchain/langgraph";

// InMemoryStore saves data to an in-memory dictionary. Use a DB-backed store in production.
const store = new InMemoryStore();
const contextSchema = z.object({
  userId: z.string(),
});

// Write sample data to the store using the put method
await store.put(
  ["users"], // Namespace to group related data together (users namespace for user data)
  "user_123", // Key within the namespace (user ID as key)
  {
    name: "John Smith",
    language: "English",
  } // Data to store for the given user
);

const getUserInfo = tool(
  // Look up user info.
  async (
    _: unknown,
    config: LangGraphRunnableConfig<z.infer<typeof contextSchema>>
  ) => {
    // Access the store - same as that provided to `createAgent`
    const userId = config.context?.userId;
    if (!userId) {
      throw new Error("userId is required");
    }
    // Retrieve data from store - returns StoreValue object with value and metadata
    const userInfo = await config.store?.get(["users"], userId);
    return userInfo?.value ? JSON.stringify(userInfo.value) : "Unknown user";
  },
  {
    name: "getUserInfo",
    description: "Look up user info by userId from the store.",
    schema: z.object({}),
  }
);

// Schema defines the structure of user information for the LLM
const UserInfo = z.object({
  name: z.string(),
  language: z.string(),
});

// Tool that allows agent to update user information (useful for chat applications)
const saveUserInfo = tool(
  async (
    userInfo: z.infer<typeof UserInfo>,
    config: LangGraphRunnableConfig<z.infer<typeof contextSchema>>
  ) => {
    const userId = config.context?.userId;
    if (!userId) {
      throw new Error("userId is required");
    }
    // Store data in the store (namespace, key, data)
    await config.store?.put(["users"], userId, userInfo);
    return "Successfully saved user info.";
  },
  {
    name: "save_user_info",
    description: "Save user info",
    schema: UserInfo,
  }
);

const agent = createAgent({
  model: "anthropic:claude-haiku-4-5",
  tools: [getUserInfo, saveUserInfo],
  contextSchema,
  // Pass store to agent - enables agent to access store when running tools
  store,
});

// Run the agent
const result = await agent.invoke(
  {
    messages: [
      { role: "user", content: "I am an authenticated user" },
      { role: "user", content: "Get my user info" },
      { role: "user", content: "My name is Linnie and I speak Russia" },
      { role: "user", content: "Save my user info" },
    ],
  },
  { context: { userId: "user_123" } }
);

logger.info("Long-term memory result", { result });
