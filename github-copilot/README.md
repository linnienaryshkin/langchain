# GitHub Copilot (GH-300) Certification Study Guide

> Preparation materials for the [GitHub Copilot (GH-300)](https://learn.microsoft.com/en-us/credentials/certifications/github-copilot) certification exam.

Read the [Study guide for Exam GH-300: GitHub Copilot](https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/gh-300) first to understand the exam objectives and structure. Then use this guide as a reference to prepare.

I also prepared a [GitHub Copilot (GH-300) Certification space](https://github.com/copilot/spaces/linnienaryshkin/2) to help you study and practice with AI chat.

## Exam Domains at a Glance

| Domain                                                | Weight |
| ----------------------------------------------------- | ------ |
| Use GitHub Copilot responsibly                        | 15–20% |
| Use GitHub Copilot features                           | 25–30% |
| GitHub Copilot features                               | 25–30% |
| Understand GitHub Copilot data and architecture       | 10–15% |
| Apply prompt engineering and context crafting         | 10–15% |
| Improve developer productivity with GitHub Copilot    | 10–15% |
| Configure privacy, content exclusions, and safeguards | 10–15% |

## Table of Contents

1. [Responsible AI Principles](#1-responsible-ai-principles)
2. [Subscription Plans](#2-subscription-plans)
3. [How Copilot Works](#3-how-copilot-works)
4. [Using Copilot in the IDE](#4-using-copilot-in-the-ide)
5. [Prompt Engineering](#5-prompt-engineering)
6. [Advanced Features](#6-advanced-features)
7. [Individual User Settings](#7-individual-user-settings)
8. [Administration: Policies & Hierarchy](#8-administration-policies--hierarchy)
9. [Content Exclusion](#9-content-exclusion)
10. [Audit Logs & Usage Metrics](#10-audit-logs--usage-metrics)
11. [Security & Data Handling](#11-security--data-handling)
12. [Best Practices & Limitations](#12-best-practices--limitations)
13. [Quick Reference](#13-quick-reference)

---

## 1. Responsible AI Principles

### Microsoft Responsible AI

GitHub Copilot is built on six principles from the [Microsoft Responsible AI framework](https://www.microsoft.com/en-us/ai/responsible-ai):

| Principle                | Definition                                                      |
| ------------------------ | --------------------------------------------------------------- |
| **Fairness**             | Avoid bias and ensure equitable treatment for all users         |
| **Reliability & Safety** | Ensure the system performs as intended and prevents harm        |
| **Privacy & Security**   | Protect user data and ensure secure interactions                |
| **Inclusiveness**        | Design for diverse users and use cases                          |
| **Transparency**         | Provide clear information about system operations and decisions |
| **Accountability**       | Ensure responsibility for system outcomes and impacts           |

### Responsible Use in Practice

- Always review and validate AI-generated code before accepting
- Understand that Copilot is a tool — you are accountable for the code you ship
- Be cautious with sensitive data, PII, and security-critical code
- Use content exclusion and policy controls to protect confidential content
- Run security scanning (SAST/DAST) on generated code

---

## 2. Subscription Plans

### Plan Comparison

| Plan           | Cost           | Code Completions | Premium Requests | Best For              |
| -------------- | -------------- | ---------------- | ---------------- | --------------------- |
| **Free**       | $0             | 2,000/mo         | 50/mo            | Hobbyists, learners   |
| **Student**    | $0 (verified)  | Unlimited        | Included         | Students              |
| **Pro**        | $10/month      | Unlimited        | 300/mo           | Individual developers |
| **Pro+**       | $39/month      | Unlimited        | 1,500/mo         | Power users           |
| **Business**   | $19/user/month | Unlimited        | 300/user/mo      | Teams & organizations |
| **Enterprise** | $39/user/month | Unlimited        | 1,000/user/mo    | Large organizations   |

> **Free for students/educators:** Verified students, teachers, and popular OSS maintainers get Pro-level access at no cost.

### Feature Availability by Plan

| Feature              | Free    | Pro      | Pro+  | Business | Enterprise     |
| -------------------- | ------- | -------- | ----- | -------- | -------------- |
| Code completions     | 2K/mo   | ∞        | ∞     | ∞        | ∞              |
| Copilot Chat         | Limited | ∞        | ∞     | ∞        | ∞              |
| Premium requests     | 50      | 300      | 1,500 | 300/user | 1,000/user     |
| Model selection      | Basic   | Extended | All   | All      | Custom/private |
| Copilot coding agent | ❌       | ✅        | ✅     | ✅        | ✅              |
| Agent mode           | ✅       | ✅        | ✅     | ✅        | ✅              |
| MCP support          | ✅       | ✅        | ✅     | ✅        | ✅              |
| PR summaries         | ❌       | ✅        | ✅     | ✅        | ✅              |
| Copilot CLI          | ✅       | ✅        | ✅     | ✅        | ✅              |
| Content exclusion    | ❌       | ❌        | ❌     | ✅        | ✅              |
| Audit logs           | ❌       | ❌        | ❌     | ✅        | ✅              |
| Policy management    | ❌       | ❌        | ❌     | ✅        | ✅              |
| SAML SSO             | ❌       | ❌        | ❌     | ✅        | ✅              |
| IP indemnity         | ❌       | ❌        | ❌     | ✅        | ✅              |
| Knowledge bases      | ❌       | ❌        | ❌     | ❌        | ✅              |
| Custom models        | ❌       | ❌        | ❌     | ❌        | ✅              |

> Premium requests are used for Copilot Chat, agent mode, code review, and advanced model selection. Additional requests cost **$0.04/request**.

> 📎 [Plans for GitHub Copilot](https://docs.github.com/en/copilot/get-started/plans)

---

## 3. How Copilot Works

### Prompt Flow Architecture

```text
┌─────────────────────────────────────────────────────────┐
│ 1. CODE EDITOR CONTEXT                                  │
│    • Code before/after cursor                           │
│    • Filename & project structure                       │
│    • Programming language detection                     │
│    • Fill-in-the-Middle (FIM) approach                  │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│ 2. PROXY SERVER                                         │
│    • Filters malicious traffic                          │
│    • Blocks prompt injection attempts                   │
│    • Protects model internals                           │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│ 3. TOXICITY FILTER                                      │
│    • Detects hate speech                                │
│    • Identifies personal data                           │
│    • Prevents harmful content                           │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│ 4. LLM ENGINE                                           │
│    • Processes context                                  │
│    • Generates code suggestions                         │
│    • Context-aware recommendations                      │
└─────────────────┬───────────────────────────────────────┘
                  │
        ┌─────────┴──────────┐
        │                    │
┌───────▼──────┐  ┌──────────▼───────┐
│ PROXY SERVER │  │ TOXICITY FILTER  │
│ (Return Path)│  │ (Return Path)    │
└───────┬──────┘  └──────────┬───────┘
        │                    │
        └─────────┬──────────┘
                  │
        ┌─────────▼──────────┐
        │    CODE EDITOR     │
        │   Display Results  │
        └────────────────────┘
```

### Code Referencing (Duplicate Detection)

Copilot checks suggestions against **~150 characters** of surrounding context in the **public code index** on GitHub:

- **Allow mode** — Suggestion is shown with references (repository URLs, license info)
- **Block mode** — Suggestion is suppressed entirely
- Matches occur in **less than 1%** of suggestions (more common in empty files or for generic patterns like FizzBuzz)
- Configurable per individual (Pro/Pro+) or enforced by organization policy (Business/Enterprise)

> 📎 [Finding public code that matches GitHub Copilot suggestions](https://docs.github.com/en/copilot/how-tos/get-code-suggestions/find-matching-code)
> 📎 [Introducing code referencing for GitHub Copilot](https://github.blog/news-insights/product-news/introducing-code-referencing-for-github-copilot/)

### Model Training & Data Usage

- By default, GitHub, its affiliates, and third parties will **NOT** use your data (prompts, suggestions, code snippets) for AI model training — this **cannot be enabled** by the user
- Opt-in telemetry: You can choose to allow prompt & suggestion collection for **product improvements** (separate from model training)
- **LoRA (Low-Rank Adaptation)** is used for model fine-tuning — adds smaller trainable components to each layer while preserving original weights, enabling efficient customization

> 📎 [Managing GitHub Copilot policies as an individual subscriber](https://docs.github.com/en/copilot/how-tos/manage-your-account/manage-policies)

---

## 4. Using Copilot in the IDE

### Supported Environments

| Category         | Tools                                                          |
| ---------------- | -------------------------------------------------------------- |
| **Code Editors** | VS Code, Neovim, Vim, Xcode, Eclipse                           |
| **IDEs**         | JetBrains (IntelliJ, PyCharm, WebStorm), Visual Studio         |
| **CLI**          | GitHub CLI (`gh copilot`) — explain commands, suggest commands |
| **Web/Mobile**   | GitHub.com, GitHub Mobile                                      |

### Inline Suggestions

- Real-time code completions as you type
- Accept with **Tab** (default, customizable)
- Cycle through alternatives with keyboard shortcuts
- **Next Edit Suggestions** — Copilot predicts your next edit location

### Chat Features & Commands

**References:**

| Syntax       | Purpose                              |
| ------------ | ------------------------------------ |
| `#filename`  | Attach a specific file to your query |
| `@workspace` | Reference entire project/workspace   |

**Slash commands:**

| Command     | Purpose                | Use Case               |
| ----------- | ---------------------- | ---------------------- |
| `/doc`      | Add comments to code   | Document existing code |
| `/explain`  | Clarify how code works | Understand algorithms  |
| `/fix`      | Propose bug fixes      | Resolve errors         |
| `/generate` | Create new code        | Build features         |
| `/optimize` | Improve efficiency     | Enhance performance    |
| `/tests`    | Generate unit tests    | Ensure code quality    |

### Chat Modes

| Mode      | Description                                                                     |
| --------- | ------------------------------------------------------------------------------- |
| **Ask**   | Ask questions and get answers in the chat panel                                 |
| **Edit**  | Copilot makes targeted edits across files based on instructions                 |
| **Agent** | Autonomous mode — Copilot plans, executes, iterates, and runs terminal commands |

### Copilot CLI

GitHub Copilot in the CLI helps developers directly from the terminal:

- `gh copilot explain` — Explain a command in natural language
- `gh copilot suggest` — Suggest a command for a task
- Interactive sessions for multi-step exploration
- Generate scripts and manage files

> 📎 [Best practices for using GitHub Copilot](https://docs.github.com/en/copilot/using-github-copilot/best-practices-for-using-github-copilot)
> 📎 [Configuring GitHub Copilot in your environment](https://docs.github.com/en/copilot/how-tos/configure-personal-settings/configure-in-ide)

---

## 5. Prompt Engineering

### The 4 S's of Effective Prompts

| Principle    | Definition                                       | Why It Matters                               |
| ------------ | ------------------------------------------------ | -------------------------------------------- |
| **Single**   | Focus on one well-defined task/question          | Clarity elicits accurate responses           |
| **Specific** | Use explicit and detailed instructions           | Specificity yields precise suggestions       |
| **Short**    | Keep prompts concise                             | Balance clarity without overwhelming Copilot |
| **Surround** | Use descriptive filenames and open related files | Rich context enables tailored suggestions    |

### Zero-Shot Prompting

Provide a description with **no examples** — Copilot infers from its training:

```python
# Function to calculate the factorial of a number
def factorial(n):
    # Copilot generates code with just this prompt
```

### Few-Shot Prompting

Provide **2–3 input/output examples** in comments so Copilot learns the pattern:

```python
# Convert temperature from Celsius to Fahrenheit
# celsius_to_fahrenheit(0) -> 32
# celsius_to_fahrenheit(100) -> 212
def celsius_to_fahrenheit(c):
    # Copilot infers the formula from the examples above
```

> **Exam tip:** Few-shot prompting dramatically improves accuracy for complex or custom logic. Providing examples guides Copilot to understand the exact pattern, format, or behavior you expect.

### Context Optimization

- **Open related files** in the editor — Copilot uses them as context
- **Close irrelevant files** to reduce noise
- Use **meaningful filenames** and folder structures
- Use `@workspace` for broader project understanding
- Maintain clean git history with descriptive commit messages
- Start new chat threads when the conversation context becomes stale

### Prompt Files

- Create reusable prompt templates as `.prompt.md` files
- Enable consistent responses across team members
- Reference with `#` in chat

> 📎 [Best practices for using GitHub Copilot](https://docs.github.com/en/copilot/using-github-copilot/best-practices-for-using-github-copilot)

---

## 6. Advanced Features

### Copilot Coding Agent

- **Server-side AI assistant** that works autonomously from GitHub issues
- Creates branches, writes code, opens pull requests
- Integrates with GitHub Actions for execution
- Can be enabled/disabled per repository in user settings
- Third-party coding agents (Anthropic Claude, OpenAI Codex) also available

### Copilot Code Review

- Automated code review on pull requests
- Suggests improvements and identifies issues
- Generates PR summaries automatically
- "Review selection" available on Free plan (VS Code only); full review on paid plans

### Custom Instructions

- Persistent instructions Copilot follows across all interactions
- Set coding standards, preferred patterns, and project-specific rules
- Configured via `.github/copilot-instructions.md` in your repository
- Organization-level custom instructions available (Business/Enterprise, preview)

### Model Context Protocol (MCP)

- Connect Copilot to external tools and data sources via MCP servers
- Available across all plans
- Enterprise policy can control MCP server access

### Copilot Extensions

- Extend Copilot with custom capabilities
- Integration with third-party tools and services
- Build domain-specific assistants

### Copilot Spaces

- Collaborative workspaces for team projects
- Centralized context and shared knowledge base

### Knowledge Bases (Enterprise Only)

- Curated collections of Markdown documentation for Copilot Chat context
- Enables organization-specific domain knowledge in responses

### GitHub Integration Points

| Area             | Capabilities                                                          |
| ---------------- | --------------------------------------------------------------------- |
| **Issues & PRs** | Draft issue descriptions, generate PR summaries, Copilot code review  |
| **CI/CD**        | Generate GitHub Actions workflows, automate testing/deployment        |
| **Repository**   | Improve structure for better suggestions, descriptive README and docs |

---

## 7. Individual User Settings

### Settings on GitHub.com (Free/Pro/Pro+)

Configure at **GitHub.com → Profile → Copilot Settings**:

| Setting                              | Options                          | Default   |
| ------------------------------------ | -------------------------------- | --------- |
| **Suggestions matching public code** | Allow / Block                    | Allow     |
| **Prompt & suggestion collection**   | Allow / Block                    | Block     |
| **Copilot coding agent**             | All repos / Selected / None      | All repos |
| **Third-party coding agents**        | Toggle per agent (Claude, Codex) | Off       |
| **Web search (Bing)**                | Enabled / Disabled               | Disabled  |

> ⚠️ **Business/Enterprise users**: These settings are **inherited from org/enterprise policy** and cannot be overridden individually.

> **Model training**: By default, GitHub/affiliates/third parties will **NOT** use your data for AI model training. This **cannot** be enabled.

### IDE-Level Settings

| Setting                     | How to Configure                                        |
| --------------------------- | ------------------------------------------------------- |
| Enable/disable Copilot      | Command palette or IDE settings                         |
| Per-language enable/disable | `settings.json` (VS Code) or IDE preferences            |
| Next edit suggestions       | `github.copilot.nextEditSuggestions.enabled` in VS Code |
| Accept keybinding           | Tab (default), customizable                             |
| Inline suggest toggle       | Toggle in editor settings                               |

> 📎 [Managing GitHub Copilot policies as an individual subscriber](https://docs.github.com/en/copilot/how-tos/manage-your-account/manage-policies)
> 📎 [Configuring GitHub Copilot in your environment](https://docs.github.com/en/copilot/how-tos/configure-personal-settings/configure-in-ide)

---

## 8. Administration: Policies & Hierarchy

### Policy Hierarchy

```text
┌────────────────────────────────────────────────────────┐
│ ENTERPRISE (Highest Authority)                         │
│  • Enterprise owner sets global policies               │
│  • Can enforce or delegate ("No policy") to orgs       │
│  • Enterprise-defined = overrides all lower levels     │
├────────────────────────────────────────────────────────┤
│ ORGANIZATION                                           │
│  • Org owner manages within enterprise boundaries      │
│  • Sets feature, privacy, and model policies           │
│  • Assigns Copilot seats to members/teams              │
│  • Cannot override enterprise-enforced policies        │
├────────────────────────────────────────────────────────┤
│ TEAM                                                   │
│  • Inherits policies from organization                 │
│  • Org admin can grant/deny Copilot per team           │
│  • Cannot override org or enterprise restrictions      │
├────────────────────────────────────────────────────────┤
│ REPOSITORY                                             │
│  • Repo admin can set content exclusion rules          │
│  • Cannot override org or enterprise restrictions      │
│  • Content exclusion applies to all users in that repo │
└────────────────────────────────────────────────────────┘
```

### Policy Types

| Type               | Controls                            | Examples                                                |
| ------------------ | ----------------------------------- | ------------------------------------------------------- |
| **Feature policy** | Availability of Copilot features    | Copilot in IDE, Copilot Chat, code review, coding agent |
| **Privacy policy** | Sensitive actions (Allowed/Blocked) | Suggestions matching public code, telemetry             |
| **Models policy**  | Availability of AI models           | Advanced models beyond basic                            |

### Enforcement Options

| Level            | Options                                                              |
| ---------------- | -------------------------------------------------------------------- |
| **Enterprise**   | Enabled / Disabled / No policy (delegate to orgs)                    |
| **Organization** | Enabled / Disabled / Unconfigured (placeholder, treated as disabled) |

### Conflict Resolution (Multi-Org Users)

When a user belongs to **multiple organizations** with conflicting policies:

| Resolution                                                    | Applies To                                                                                        |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **Most restrictive** (any org disables → disabled everywhere) | Copilot Metrics API, suggestions matching public code, Copilot code review for unlicensed members |
| **Least restrictive** (any org enables → enabled everywhere)  | Copilot Chat, coding agent, Copilot CLI, MCP servers, most other features                         |

> 📎 [Feature availability when policies conflict](https://docs.github.com/en/copilot/reference/policy-conflicts)

### ⏱️ 30-Minute Propagation Delay

After changing content exclusion or policy settings:

> **It can take up to 30 minutes for changes to take effect in IDEs where the settings are already loaded.**

**Force immediate propagation:**

| IDE                           | How to Reload                                  |
| ----------------------------- | ---------------------------------------------- |
| **VS Code**                   | Command Palette → `Developer: Reload Window`   |
| **JetBrains / Visual Studio** | Close and reopen the application               |
| **Vim/Neovim**                | Automatic — settings fetched on each file open |

> 📎 [Managing policies for Copilot in your enterprise](https://docs.github.com/en/copilot/how-tos/administer-copilot/manage-for-enterprise/manage-enterprise-policies)
> 📎 [GitHub Copilot policies](https://docs.github.com/en/copilot/concepts/policies)

---

## 9. Content Exclusion

Content exclusion is the mechanism for preventing Copilot from accessing certain files. Available on **Business** and **Enterprise** plans only.

### Who Can Configure

| Level            | Controller       | Scope                                             |
| ---------------- | ---------------- | ------------------------------------------------- |
| **Repository**   | Repo admin       | Affects Copilot users working in that repo        |
| **Organization** | Org owner        | Affects users assigned a Copilot seat by that org |
| **Enterprise**   | Enterprise owner | Affects ALL Copilot users in the enterprise       |

### What Happens When Content Is Excluded

- ❌ No inline suggestions in excluded files
- ❌ Excluded content will NOT inform suggestions in other files
- ❌ Excluded content will NOT inform Copilot Chat responses
- ❌ Excluded files will NOT be reviewed by Copilot code review

### How to Configure

1. **Repository:** Settings → Code & automation → Copilot → Content exclusion
2. **Organization:** Settings → Copilot → Content exclusion
3. **Programmatically:** Via the [REST API for content exclusion management](https://docs.github.com/en/rest/copilot/copilot-content-exclusion-management)

### Limitations

- Copilot may still use **semantic information** provided by the IDE indirectly (e.g., type info, hover definitions, build config)
- Does not apply to **symbolic links** or **remote filesystems**
- **Not supported** in Copilot CLI, Copilot coding agent, and Edit/Agent modes of Copilot Chat in IDEs

> ⚠️ **Important:** Having a file in `.gitignore` does **NOT** prevent Copilot from reading it. You must configure content exclusion separately.

> 📎 [Content exclusion for GitHub Copilot](https://docs.github.com/en/copilot/concepts/context/content-exclusion)
> 📎 [Excluding content from GitHub Copilot](https://docs.github.com/en/copilot/how-tos/configure-content-exclusion/exclude-content-from-copilot)

---

## 10. Audit Logs & Usage Metrics

### Audit Logs

Available for **Business** and **Enterprise** plans. Retained for **180 days**.

| Event                                 | Description                         |
| ------------------------------------- | ----------------------------------- |
| `copilot.cfb_seat_assignment_created` | Copilot seat assigned to a user     |
| `copilot.cfb_seat_assignment_deleted` | Copilot seat removed from a user    |
| `copilot.content_exclusion_changed`   | Content exclusion settings modified |
| `copilot.policy.updated`              | Copilot policy changed              |

Each event includes: **Timestamp**, **Actor**, **Action**, **Affected user**, **Metadata** (IP, user agent).

**How to access:**

1. Navigate to **Organization Settings → Archive → Logs → Audit log**
2. Search using `action:copilot` to filter all Copilot events
3. Example: `action:copilot.cfb_seat_assignment_created` for seat assignments

**Content exclusion auditing:**

1. Go to **Repository Settings → Copilot** or **Organization Settings → Copilot → Content exclusion**
2. Scroll to the bottom to see who last changed settings and when
3. Click the timestamp to view the full audit log with `copilot.content_exclusion_changed` entries

### Usage Metrics (Enterprise)

Enterprise owners have access to **Copilot usage metrics** via dashboard and APIs:

- Adoption rates and active users (daily/weekly/monthly)
- Code completions suggested vs. accepted
- Agent and chat usage breakdowns
- Lines of code changed with AI
- Model and language distribution
- Pull request activity by Copilot

> 📎 [Reviewing audit logs for GitHub Copilot Business](https://docs.github.com/en/copilot/how-tos/administer-copilot/manage-for-organization/review-activity/review-audit-logs)
> 📎 [Reviewing changes to content exclusions](https://docs.github.com/en/copilot/how-tos/configure-content-exclusion/review-changes)
> 📎 [Audit log events for your organization – Copilot](https://docs.github.com/en/organizations/keeping-your-organization-secure/managing-security-settings-for-your-organization/audit-log-events-for-your-organization#copilot)
> 📎 [Data available in Copilot usage metrics](https://docs.github.com/en/copilot/reference/copilot-usage-metrics/copilot-usage-metrics)

---

## 11. Security & Data Handling

### Data Protection

| Measure                   | Details                                                                     |
| ------------------------- | --------------------------------------------------------------------------- |
| **Encryption in Transit** | All data encrypted during transmission                                      |
| **Encryption at Rest**    | Data encrypted in storage                                                   |
| **Compliance**            | SOC 2, ISO 27001                                                            |
| **Audit Logs**            | Track usage and actions (180 days, Business/Enterprise)                     |
| **IP Indemnity**          | Intellectual property indemnification (Business/Enterprise)                 |
| **Data Training**         | Your code is **NOT** used for model training by default (cannot be enabled) |

### Sensitive Data Handling

- Be cautious with code handling sensitive data or PII
- Use content exclusion to prevent Copilot from accessing secrets and credentials
- Review generated code for hardcoded secrets or insecure patterns
- Test in non-production environments first

### Code Review & Validation Checklist

1. ✅ Always review suggested code before accepting
2. ✅ Validate correctness, security, and readability
3. ✅ Run security scanning tools (SAST/DAST)
4. ✅ Write tests for generated code (unit + integration)
5. ✅ Peer review all AI-generated suggestions
6. ✅ Check licensing implications

---

## 12. Best Practices & Limitations

### ✅ When to Use Copilot

| Scenario                          | Benefits                                |
| --------------------------------- | --------------------------------------- |
| Boilerplate and repetitive code   | Saves time on patterns you already know |
| Learning new languages/frameworks | Quick syntax and pattern examples       |
| Test generation                   | Rapid unit test creation                |
| Documentation and comments        | Auto-documentation features             |
| Debugging and fixing syntax       | Quick error resolution                  |
| CLI commands and scripts          | Shell command generation                |
| Refactoring and modernization     | Modernize legacy code                   |
| Regex and complex expressions     | Pattern generation                      |

### ❌ When NOT to Use Copilot

- Highly sensitive or proprietary algorithms
- Critical security infrastructure code
- Compliance-sensitive code requiring explicit documentation
- Cryptographic implementations (use audited libraries instead)
- Safety-critical systems (medical devices, transportation)

### Known Limitations

| Area                  | Details                                                                                                 |
| --------------------- | ------------------------------------------------------------------------------------------------------- |
| **Context awareness** | May lack awareness of very large codebases; limited understanding of custom business logic              |
| **Accuracy**          | Generated code may contain bugs, inefficiencies, or security vulnerabilities                            |
| **Edge cases**        | Complex domain-specific logic and corner cases need developer validation                                |
| **Performance**       | Large context windows and complex prompts may impact response time; network latency affects suggestions |
| **Licensing**         | Review GPL and copyleft licenses carefully; IP indemnity is included with Business and Enterprise plans |

---

## 13. Quick Reference

### Key Numbers to Remember

| Value              | Meaning                                                    |
| ------------------ | ---------------------------------------------------------- |
| **30 minutes**     | Max propagation delay for policy/content exclusion changes |
| **150 characters** | Context window checked for duplicate/public code matching  |
| **< 1%**           | Frequency of suggestions matching public code              |
| **180 days**       | Audit log retention period                                 |
| **$0.04**          | Cost per additional premium request                        |

### The 4 S's

```text
Single   → One clear task
Specific → Detailed instructions
Short    → Concise phrasing
Surround → Rich file context
```

### Chat Quick Reference

```text
#filename      → Reference specific files
@workspace     → Reference entire project
/doc           → Add documentation
/explain       → Explain code
/fix           → Fix problems
/generate      → Create new code
/optimize      → Improve efficiency
/tests         → Generate tests
```

### Policy Hierarchy

```text
Enterprise  →  Defines or delegates ("No policy")
  └─ Organization  →  Configures within enterprise boundaries
       └─ Team  →  Inherits from org
            └─ Repository  →  Content exclusion only
```

---

## Related Resources

- [GitHub Copilot Official Documentation](https://docs.github.com/en/copilot)
- [GitHub Copilot Trust Center](https://copilot.github.trust.page)
- [Microsoft Responsible AI Principles](https://www.microsoft.com/en-us/ai/responsible-ai)
- [Study guide for Exam GH-300](https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/gh-300)
- [Plans for GitHub Copilot](https://docs.github.com/en/copilot/get-started/plans)
- [Best practices for using GitHub Copilot](https://docs.github.com/en/copilot/using-github-copilot/best-practices-for-using-github-copilot)
- [GitHub Copilot policies](https://docs.github.com/en/copilot/concepts/policies)
- [Content exclusion for GitHub Copilot](https://docs.github.com/en/copilot/concepts/context/content-exclusion)
- [Managing policies as an individual subscriber](https://docs.github.com/en/copilot/how-tos/manage-your-account/manage-policies)
- [Reviewing audit logs for Copilot Business](https://docs.github.com/en/copilot/how-tos/administer-copilot/manage-for-organization/review-activity/review-audit-logs)
- [Finding public code that matches Copilot suggestions](https://docs.github.com/en/copilot/how-tos/get-code-suggestions/find-matching-code)
- [Feature availability when policies conflict](https://docs.github.com/en/copilot/reference/policy-conflicts)
