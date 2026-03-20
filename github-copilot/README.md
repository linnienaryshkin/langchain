# [GitHub Copilot (GH-300)](https://learn.microsoft.com/en-us/credentials/certifications/github-copilot) certification

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

## Advanced Copilot Features

### Copilot Spaces
* Collaborative workspaces for teams to study and build together
* Centralized context for team projects
* Shared knowledge base and documentation

### Copilot Extensions
* Extend Copilot with custom capabilities
* Integrate with third-party tools and services
* Create specialized workflows for specific domains

### Copilot in CLI
* GitHub Copilot integration in command-line interfaces
* Assists with writing shell commands and scripts
* Available in GitHub CLI (gh copilot)

### IDE Support
* VS Code (with Copilot extension)
* JetBrains IDEs (IntelliJ, PyCharm, WebStorm, etc.)
* Visual Studio
* Neovim and other editors

## Security & Best Practices

### Code Review
* Always review suggested code before accepting
* Ensure security best practices are followed
* Validate code quality and correctness

### Sensitive Data Handling
* Be cautious when working with code handling sensitive data
* Implement proper data protection measures
* Consider data retention and privacy implications

### Public Code Index
* GitHub Copilot may use public code for training (opt-out available)
* Option to exclude repositories from the public code index
* Configure organization policies for data usage

### Enterprise Security
* Data encryption in transit and at rest
* Compliance with security standards (SOC 2, ISO 27001)
* Audit logs for enterprise deployments
* Custom deployment options for sensitive environments

## GitHub Copilot for Business/Enterprise

### Organization Settings & Policies
* Manage Copilot access at the organization level
* Set usage policies and guidelines
* Control which users can use Copilot

### User Management & Licensing
* Assign Copilot licenses to team members
* Track usage and seat allocation
* Manage license renewals

### Audit Logs & Monitoring
* View audit logs for Copilot usage
* Monitor team productivity metrics
* Track adoption and usage patterns

### Custom Models & Fine-tuning
* Enterprise customers can request custom models
* Fine-tune models on proprietary codebases
* Deploy custom instances for specialized workflows

## Hands-on Skills & Best Practices

### Code Completion Techniques
* Provide clear context before the cursor
* Use descriptive variable and function names
* Structure code for optimal Copilot suggestions

### Context Optimization
* Keep related files open in the editor
* Use meaningful filenames and folder structures
* Leverage @workspace for broader context

### Chat Interaction Best Practices
* Start with clear, specific questions
* Use file references (#) and environment references (@) effectively
* Follow up with clarifying questions for better results

### Testing & Validation
* Always test generated code thoroughly
* Run unit tests and integration tests
* Use code review processes to validate suggestions

## Limitations & Ethical Considerations

### Context Awareness
* Copilot may lack awareness of very large codebases
* Complex domain-specific logic may require manual intervention
* Edge cases and corner cases need developer validation

### Occasional Inaccuracies
* Generated code may contain bugs or inefficiencies
* Security vulnerabilities may be present in suggestions
* Always review and test code before production use

### When NOT to Use Copilot
* For highly sensitive or proprietary algorithms
* When dealing with critical security infrastructure
* For compliance-sensitive code requiring explicit documentation

### Copyright & Licensing
* Be aware of licensing implications of suggested code
* Consider GPL and other copyleft licenses
* Understand usage rights and attribution requirements

### Performance Considerations
* Large context windows may impact performance
* Processing time for complex prompts varies
* Network latency affects suggestion latency

## GitHub Integration Points

### Issues & Pull Requests
* Copilot can assist with issue descriptions
* Help draft pull request descriptions
* Suggest code changes and improvements

### GitHub Actions Automation
* Generate CI/CD workflow files
* Automate testing and deployment
* Create custom actions and scripts

### Repository Context
* Optimize repository structure for better suggestions
* Add descriptive README and documentation
* Use meaningful commit messages and branches

## Pricing & Licensing Deep Dive

### Comparison of Plans
| Plan | Cost | Features | Best For |
|------|------|----------|----------|
| Free | $0 | ~2,000 completions/month, basic Chat | Individual hobbyists |
| Pro | $10/month | Unlimited completions, priority Chat | Individual developers |
| Pro+ | $39/month | Maximum priority, advanced features | Power users, professionals |
| Business | $19/user/month | Team management, policies, audit logs | Teams & organizations |
| Enterprise | $39/user/month | Custom models, deep integration, SLA | Large enterprises |

### Cost Optimization Strategies
* Evaluate team size and usage patterns
* Consider seat sharing for part-time users
* Leverage organization discounts
* Monitor ROI and productivity gains
