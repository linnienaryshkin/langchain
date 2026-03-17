# [GitHub Copilot (GH-300)](https://learn.microsoft.com/en-us/credentials/certifications/github-copilot) ceriticatoin

## Microsoft Responsible AI and GitHub Copilot Principles

* Fairness - avoid bias and ensure equitable treatment for all users
* Reliability and safety - ensure the system performs as intended and does not cause harm
* Privacy and security - protect user data and ensure secure interactions
* Inclusiveness - design for a diverse range of users and use cases
* Transparency - provide clear information about how the system works and makes decisions
* Accountability - ensure responsibility for the system's outcomes and impacts

## GitHub Copilot features and the subscription plans

* Copilot Free: Free access for individuals with limited usage (approx. 2,000 completions/month).
* Copilot Pro: $10/month for individuals, offering unlimited code suggestions and higher Chat limits.
* Copilot Pro+: $39/month for individual power users with maximum premium request limits.
* Copilot Business: $19/user/month for teams, adding security and policy management.
* Copilot Enterprise: $39/user/month for large organizations requiring custom AI and deeper GitHub integration.

## Principles of Prompt Engineering 4s

* Single: Always focus your prompt on a single, well-defined task or question. This clarity is crucial for eliciting accurate and useful responses from Copilot.
* Specific: Ensure that your instructions are explicit and detailed. Specificity leads to more applicable and precise code suggestions.
* Short: While being specific, keep prompts concise and to the point. This balance ensures clarity without overloading Copilot or complicating the interaction.
* Surround: Utilize descriptive filenames and keep related files open. This provides Copilot with rich context, leading to more tailored code suggestions.

## Prompt flow

1. Code Editor context: Code before and after the cursor, Filename, project structure, programming languages - Fill-in-the-Middle
2. Proxy Server: filters traffic, blocking attempts to hack the prompt or manipulate the system into revealing details about how the model generates code suggestions
3. Toxicity filter: Hate speech, Personal data
4. LLM: GitHub Copilot utilizes LLMs to provide context-aware code suggestions
5. Proxy Server
6. Toxicity filter
7. Code Editor

### LoRA fine-tuning

* LoRA adds smaller trainable parts to each layer of the pretrained model, instead of changing everything.
* The original model remains the same, which saves time and resources.

## Chat features

You can significantly improve the quality and relevance of GitHub Copilot Chat's responses with certain key features

* `#` File references
* `@` Environment References
  * `@workspace` reference the entire solution or workspace
* `/` allow you to quickly specify the intent of your query
  * `/doc` Adds comments to the specified or selected code
  * `/explain`
  * `/fix` Proposes fixes for problems in the selected code
  * `/generate` generating new code based on your requirements
  * `/optimize` suggests improvements to the running time or efficiency of the selected code
  * `/tests` creates unit tests for the selected code

## GitHub Copilot coding agent

* GitHub Copilot coding agent is an AI-powered assistant that helps developers write code more efficiently and effectively.
* It runs on the server side, performing assigned tasks in repository issues.
