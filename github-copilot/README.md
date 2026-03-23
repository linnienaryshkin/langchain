# GitHub Copilot (GH-300) Certification Study Guide

> Preparation materials for the [GitHub Copilot (GH-300)](https://learn.microsoft.com/en-us/credentials/certifications/github-copilot) certification exam.

First, I encourage you to read [Study guide for Exam GH-300: GitHub Copilot](https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/gh-300?utm_source=openai) so you understand the exam objectives and structure. Then, use this guide as reference to prepare for the exam.

Also, I prepared a [GitHub Copilot (GH-300) Certification space](https://github.com/copilot/spaces/linnienaryshkin/2) to help you study and practice with AI chat.

## Table of Contents

1. [Core Principles](#core-principles)
2. [Subscription Plans](#subscription-plans)
3. [How Copilot Works](#how-copilot-works)
4. [Using Copilot Effectively](#using-copilot-effectively)
5. [Advanced Features](#advanced-features)
6. [Enterprise & Security](#enterprise--security)
7. [Content Exclusion & Ignoring Files](#content-exclusion--ignoring-files)
8. [Policy Hierarchy & Propagation](#policy-hierarchy--propagation)
9. [Audit Logs](#audit-logs)
10. [Individual User Settings](#individual-user-settings)
11. [Best Practices & Limitations](#best-practices--limitations)

---

## Core Principles

### Microsoft Responsible AI & GitHub Copilot Principles

- **Fairness** - Avoid bias and ensure equitable treatment for all users
- **Reliability & Safety** - Ensure the system performs as intended and prevents harm
- **Privacy & Security** - Protect user data and ensure secure interactions
- **Inclusiveness** - Design for diverse users and use cases
- **Transparency** - Provide clear information about system operations and decisions
- **Accountability** - Ensure responsibility for system outcomes and impacts

> 📎 [Microsoft Responsible AI Principles](https://www.microsoft.com/en-us/ai/responsible-ai)

### Prompt Engineering: The 4 S's

| Principle    | Definition                                       | Why It Matters                               |
| ------------ | ------------------------------------------------ | -------------------------------------------- |
| **Single**   | Focus on one well-defined task/question          | Clarity elicits accurate responses           |
| **Specific** | Use explicit and detailed instructions           | Specificity yields precise suggestions       |
| **Short**    | Keep prompts concise                             | Balance clarity without overwhelming Copilot |
| **Surround** | Use descriptive filenames and open related files | Rich context enables tailored suggestions    |

### Prompt Techniques

#### Zero-Shot Prompting

Provide a description with **no examples** — Copilot infers from its training:

```python
# Function to calculate the factorial of a number
def factorial(n):
    # Copilot generates code with just this prompt
```

#### Few-Shot Prompting

Provide **2–3 input/output examples** in comments so Copilot learns the pattern:

```python
# Convert temperature from Celsius to Fahrenheit
# celsius_to_fahrenheit(0) -> 32
# celsius_to_fahrenheit(100) -> 212
def celsius_to_fahrenheit(c):
    # Copilot infers the formula from the examples above
```

**Why it matters for the exam:** Few-shot prompting dramatically improves accuracy for complex or custom logic. Providing examples guides Copilot to understand the exact pattern, format, or behavior you expect.

> 📎 [Best practices for using GitHub Copilot](https://docs.github.com/en/copilot/using-github-copilot/best-practices-for-using-github-copilot)

---

## Subscription Plans

### Plan Comparison

| Plan           | Cost           | Code Completions | Premium Requests   | Key Features                                            | Best For                   |
| -------------- | -------------- | ---------------- | ------------------ | ------------------------------------------------------- | -------------------------- |
| **Free**       | $0             | 2,000/mo         | 50/mo              | Basic Chat, limited suggestions                         | Hobbyists, learners        |
| **Pro**        | $10/month      | Unlimited        | 300/mo             | Unlimited completions, extended model selection          | Individual developers      |
| **Pro+**       | $39/month      | Unlimited        | 1,500/mo           | All models (incl. advanced), maximum priority            | Power users, professionals |
| **Business**   | $19/user/month | Unlimited        | 300/user/mo        | Policy management, content exclusion, audit logs, SSO   | Teams & organizations      |
| **Enterprise** | $39/user/month | Unlimited        | 1,000/user/mo      | Custom models, codebase indexing, knowledge bases, SLA  | Large organizations        |

### What's Included by Plan

| Feature                        | Free | Pro | Pro+ | Business | Enterprise |
| ------------------------------ | ---- | --- | ---- | -------- | ---------- |
| Code completions               | 2K/mo | ∞  | ∞    | ∞        | ∞          |
| Copilot Chat                   | Limited | ∞ | ∞   | ∞        | ∞          |
| Premium requests               | 50   | 300 | 1,500 | 300/user | 1,000/user |
| Model selection                | Basic | Extended | All | All   | Custom/private |
| Content exclusion              | ❌   | ❌  | ❌   | ✅       | ✅         |
| Audit logs                     | ❌   | ❌  | ❌   | ✅       | ✅         |
| Policy management              | ❌   | ❌  | ❌   | ✅       | ✅         |
| SAML SSO                       | ❌   | ❌  | ❌   | ✅       | ✅         |
| IP indemnity                   | ❌   | ❌  | ❌   | ✅       | ✅         |
| Codebase indexing              | ❌   | ❌  | ❌   | ❌       | ✅         |
| Knowledge bases                | ❌   | ❌  | ❌   | ❌       | ✅         |
| Custom models                  | ❌   | ❌  | ❌   | ❌       | ✅         |

> **Note:** Premium requests are used for advanced features (Copilot Chat, agent mode, code review, advanced model selection). Additional requests cost $0.04/request.

> **Free for students/educators:** Verified students, teachers, and popular OSS maintainers get Pro-level access at no cost.

### Cost Optimization Strategies

- Evaluate team size and actual usage patterns
- Consider seat sharing for part-time users
- Leverage organization discounts
- Monitor ROI and measure productivity gains
- Regular license audits to eliminate waste

> 📎 [Plans for GitHub Copilot](https://docs.github.com/en/copilot/get-started/plans)

---

## How Copilot Works

### The Prompt Flow Architecture

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
┌───────▼──────┐  ┌──────────▼────────┐
│ PROXY SERVER │  │ TOXICITY FILTER   │
│ (Return Path)│  │ (Return Path)     │
└───────┬──────┘  └──────────┬────────┘
        │                    │
        └─────────┬──────────┘
                  │
        ┌─────────▼──────────┐
        │  CODE EDITOR       │
        │ Display Results    │
        └────────────────────┘
```

### Duplicate Detection / Code Referencing

When Copilot generates a suggestion, it checks against ~150 characters of surrounding context in the **public code index on GitHub**:

- **Allow mode** - Suggestion is shown with references (repository URLs, license info)
- **Block mode** - Suggestion is suppressed entirely ("matched public code so it was blocked")
- Occurs in **less than 1%** of suggestions (more common for generic patterns like FizzBuzz)
- Can be configured per individual (Pro/Pro+) or enforced by organization policy (Business/Enterprise)

> 📎 [Finding public code that matches GitHub Copilot suggestions](https://docs.github.com/en/copilot/how-tos/get-code-suggestions/find-matching-code)
> 📎 [Introducing code referencing for GitHub Copilot](https://github.blog/news-insights/product-news/introducing-code-referencing-for-github-copilot/)

### Model Fine-tuning: LoRA Approach

**LoRA (Low-Rank Adaptation):**

- Adds smaller trainable components to each model layer
- Preserves original model weights (memory efficient)
- Reduces training time and resource requirements
- Enables quick customization without full retraining

### Model Training & Data Usage

- By default, GitHub, its affiliates, and third parties will **NOT** use your data (prompts, suggestions, code snippets) for AI model training
- This is reflected in personal settings and **cannot be enabled** by the user
- Opt-in telemetry: You can choose to allow prompt & suggestion collection for **product improvements** (separate from model training)

> 📎 [Managing GitHub Copilot policies as an individual subscriber](https://docs.github.com/en/copilot/how-tos/manage-your-account/manage-policies)

---

## Using Copilot Effectively

### Chat Features & Commands

#### References

- **`#` File References** - Link specific files to your query
- **`@` Environment References**
  - `@workspace` - Reference entire solution/workspace

#### Intent Modifiers (`/` commands)

| Command     | Purpose                | Use Case               |
| ----------- | ---------------------- | ---------------------- |
| `/doc`      | Add comments to code   | Document existing code |
| `/explain`  | Clarify how code works | Understand algorithms  |
| `/fix`      | Propose bug fixes      | Resolve errors         |
| `/generate` | Create new code        | Build features         |
| `/optimize` | Improve efficiency     | Enhance performance    |
| `/tests`    | Generate unit tests    | Ensure code quality    |

### Code Completion Techniques

#### Before You Ask

1. **Provide Clear Context** - Code before the cursor sets expectations
2. **Use Descriptive Names** - Variable and function names guide suggestions
3. **Structure for Success** - Organize code logically for optimal parsing

#### Context Optimization

- Keep related files open in the editor
- Use meaningful filenames and folder structures
- Leverage `@workspace` for broader understanding
- Maintain clean git history with descriptive commit messages

#### Chat Interaction Best Practices

1. Start with clear, specific questions
2. Use file (`#`) and environment (`@`) references effectively
3. Follow up with clarifying questions for refinement
4. Review suggestions critically before acceptance

---

## Advanced Features

### Copilot Agents & Automation

#### GitHub Copilot Coding Agent

- **Server-side AI assistant** that performs assigned tasks
- Executes operations from repository issues automatically
- Handles complex multi-step code generation tasks
- Integrates with GitHub Actions and workflows
- Can be enabled/disabled per repository in user settings
- Third-party coding agents (Anthropic Claude, OpenAI Codex) also available

### Copilot Code Review

- Automated code review on pull requests
- Suggests improvements and identifies issues
- Can generate PR summaries automatically
- Controlled by enterprise/org policy
- Available on Copilot Business and Enterprise plans

### IDE & Environment Support

| Category         | Tools                                                  |
| ---------------- | ------------------------------------------------------ |
| **Code Editors** | VS Code, Neovim, GitHub Web Editor, Xcode, Eclipse     |
| **IDEs**         | JetBrains (IntelliJ, PyCharm, WebStorm), Visual Studio |
| **CLI**          | GitHub CLI (`gh copilot`), Terminal integration         |
| **Web/Mobile**   | GitHub.com, GitHub Mobile                              |

### Copilot Spaces

- Collaborative workspaces for team study and development
- Centralized context repository for team projects
- Shared knowledge base and documentation
- Enables consistent team workflows

### Copilot Extensions & MCP

- Extend Copilot with custom capabilities
- Integration with third-party tools and services
- Create specialized workflows for specific domains
- Build domain-specific assistants
- **Model Context Protocol (MCP)** - Connect Copilot to external tools and data sources via MCP servers (controlled by enterprise policy)

### Knowledge Bases (Enterprise Only)

- Create collections of documentation for Copilot Chat context
- Curated repositories of Markdown docs that Copilot can reference
- Enables organization-specific domain knowledge in responses

### Custom Instructions

- Provide persistent instructions that Copilot follows across all interactions
- Set coding standards, preferred patterns, and project-specific rules
- Configured via `.github/copilot-instructions.md` in your repository

### GitHub Integration Points

#### Issues & Pull Requests

- Draft issue descriptions
- Generate PR descriptions and summaries automatically
- Suggest code improvements and changes
- Copilot code review on PRs

#### CI/CD & Automation

- Generate GitHub Actions workflow files
- Automate testing and deployment pipelines
- Create custom actions and shell scripts

#### Repository Optimization

- Improve repo structure for better suggestions
- Write descriptive README and documentation
- Use meaningful commit messages and branch names
- Enable better code discovery and context

---

## Enterprise & Security

### Security Architecture

#### Data Protection Measures

- **Encryption in Transit** - All data encrypted during transmission
- **Encryption at Rest** - Data encrypted in storage
- **Compliance Standards** - SOC 2, ISO 27001
- **Audit Logs** - Track all usage and actions (last 180 days)
- **IP Indemnity** - Business and Enterprise plans include intellectual property indemnification
- **Custom Deployments** - Sensitive environment support

#### Code Review & Validation

⚠️ **Critical Security Steps:**

1. Always review suggested code before accepting
2. Ensure security best practices are followed
3. Validate code quality and correctness
4. Run security scanning tools
5. Test in non-production environments first

#### Sensitive Data Handling

- ⚠️ Be cautious with code handling sensitive data
- Implement proper data protection measures
- Consider data retention and privacy implications
- Use encryption for secrets and credentials
- Exclude sensitive files from Copilot context using content exclusion

### Public Code Index & Training

- GitHub Copilot checks suggestions against a **public code index** on GitHub
- **Duplicate detection**: Matches ~150 chars of context; can **Allow** (show references) or **Block** (suppress suggestion)
- **Opt-out available** - Exclude repositories from public index
- Configure organization policies for data usage
- Review and adjust personal privacy settings
- By default, your data is **NOT used for model training**

---

## Content Exclusion & Ignoring Files

### Content Exclusion (Repository/Organization Settings)

Content exclusion is the **official mechanism** for preventing Copilot from accessing certain files. Available on **Business** and **Enterprise** plans only.

#### Who Can Configure

| Level | Controller | Scope |
|---|---|---|
| **Repository** | Repo admin | Affects Copilot users working in that repo |
| **Organization** | Org owner | Affects users assigned a Copilot seat by that org |
| **Enterprise** | Enterprise owner | Affects ALL Copilot users in the enterprise |

#### What Happens When Content Is Excluded

- ❌ No inline suggestions in excluded files
- ❌ Excluded content will NOT inform suggestions in other files
- ��� Excluded content will NOT inform Copilot Chat responses
- ❌ Excluded files will NOT be reviewed by Copilot code review

#### How to Configure

1. Repository: **Settings → Code & automation → Copilot → Content exclusion**
2. Organization: **Settings → Copilot → Content exclusion**
3. Programmatically: Via the [REST API for content exclusion management](https://docs.github.com/en/rest/copilot/copilot-content-exclusion-management)

#### Limitations

- Copilot may still use **semantic information** provided by the IDE indirectly (e.g., type info, hover definitions, build config)
- Does not apply to symbolic links or remote filesystems
- Not supported in Edit and Agent modes of Copilot Chat (preview)

⚠️ **Important:** Having a file in `.gitignore` does **NOT** prevent Copilot from reading it. You must configure content exclusion separately.

> 📎 [Content exclusion for GitHub Copilot](https://docs.github.com/en/copilot/concepts/context/content-exclusion)
> 📎 [Excluding content from GitHub Copilot](https://docs.github.com/en/copilot/how-tos/configure-content-exclusion/exclude-content-from-copilot)

---

## Policy Hierarchy & Propagation

### Enterprise → Organization → Team → Repository

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

| Type | Controls | Examples |
|---|---|---|
| **Feature policy** | Availability of Copilot features | Copilot in IDE, Copilot Chat, code review |
| **Privacy policy** | Sensitive actions (Allowed/Blocked) | Suggestions matching public code, telemetry |
| **Models policy** | Availability of AI models | Advanced models beyond basic |

### Policy Enforcement Options

| Level | Options |
|---|---|
| **Enterprise** | Enabled / Disabled / No policy (delegate to orgs) |
| **Organization** | Enabled / Disabled / Unconfigured (placeholder, treated as disabled) |

### Conflict Resolution

- When a user belongs to **multiple organizations** with conflicting policies, either the **least or most permissive** policy applies depending on the specific policy type
- See: [Feature availability when policies conflict](https://docs.github.com/en/copilot/reference/policy-conflicts)

### ⏱️ 30-Minute Propagation Delay

After changing content exclusion or policy settings:

> **It can take up to 30 minutes for changes to take effect in IDEs where the settings are already loaded.**

**Force immediate propagation:**

| IDE | How to Reload |
|---|---|
| **VS Code** | Command Palette → `Developer: Reload Window` |
| **JetBrains / Visual Studio** | Close and reopen the application |
| **Vim/Neovim** | Automatic — settings fetched on each file open |

> 📎 [Excluding content from GitHub Copilot – Propagation](https://docs.github.com/en/copilot/how-tos/configure-content-exclusion/exclude-content-from-copilot)
> 📎 [Managing policies for Copilot in your enterprise](https://docs.github.com/en/copilot/how-tos/administer-copilot/manage-for-enterprise/manage-enterprise-policies)
> 📎 [GitHub Copilot policies](https://docs.github.com/en/copilot/concepts/policies)

---

## Audit Logs

### Overview

Audit logs for GitHub Copilot are available for **Business** and **Enterprise** plans. They are retained for **180 days** and track actions taken by users in the organization.

### What Is Tracked

| Event | Description |
|---|---|
| `copilot.cfb_seat_assignment_created` | Copilot seat assigned to a user |
| `copilot.cfb_seat_assignment_deleted` | Copilot seat removed from a user |
| `copilot.content_exclusion_changed` | Content exclusion settings modified |
| `copilot.policy.updated` | Copilot policy changed |

Each event includes: **Timestamp**, **Actor** (who made the change), **Action**, **Affected user**, **Metadata** (IP, user agent).

### How to Access

1. Navigate to **Organization Settings → Archive → Logs → Audit log**
2. Search using `action:copilot` to filter all Copilot events
3. Example: `action:copilot.cfb_seat_assignment_created` for seat assignments

### Content Exclusion Auditing

Changes to content exclusion settings are specifically tracked:

1. Go to **Repository Settings → Copilot** or **Organization Settings → Copilot → Content exclusion**
2. Scroll to the bottom to see who last changed settings and when
3. Click the timestamp to view the full audit log with `copilot.content_exclusion_changed` entries

### Usage Metrics (Enterprise)

Enterprise owners also have access to **Copilot usage metrics** via dashboard and APIs, covering adoption, usage, and code generation activity.

> 📎 [Reviewing audit logs for GitHub Copilot Business](https://docs.github.com/en/copilot/how-tos/administer-copilot/manage-for-organization/review-activity/review-audit-logs)
> 📎 [Reviewing changes to content exclusions](https://docs.github.com/en/copilot/how-tos/configure-content-exclusion/review-changes)
> 📎 [Audit log events for your organization – Copilot](https://docs.github.com/en/organizations/keeping-your-organization-secure/managing-security-settings-for-your-organization/audit-log-events-for-your-organization#copilot)
> 📎 [Data available in Copilot usage metrics](https://docs.github.com/en/copilot/reference/copilot-usage-metrics/copilot-usage-metrics)

---

## Individual User Settings

### Settings on GitHub.com (Pro/Pro+/Free)

Individual subscribers can configure the following at **GitHub.com → Profile → Copilot Settings**:

| Setting | Options | Default |
|---|---|---|
| **Suggestions matching public code** | Allow / Block | Allow |
| **Prompt & suggestion collection** | Allow / Block | Block |
| **Copilot coding agent** | All repos / Selected / None | All repos |
| **Third-party coding agents** | Toggle per agent (Claude, Codex) | Off |
| **Web search (Bing)** | Enabled / Disabled | Disabled |

> ⚠️ **Business/Enterprise users**: These settings are **inherited from org/enterprise policy** and cannot be overridden individually.

> **Model training**: By default, GitHub/affiliates/third parties will **NOT** use your data for AI model training. This cannot be enabled.

### IDE-Level Settings

| Setting | How to Configure |
|---|---|
| Enable/disable Copilot | Command palette or IDE settings |
| Per-language enable | `settings.json` (VS Code) or IDE preferences |
| Suggestion delay | Configure in extension settings |
| Accept keybinding | Tab (default), customizable |
| Inline suggest toggle | Toggle in editor settings |

> 📎 [Managing GitHub Copilot policies as an individual subscriber](https://docs.github.com/en/copilot/how-tos/manage-your-account/manage-policies)
> 📎 [Configuring GitHub Copilot in your environment](https://docs.github.com/en/copilot/how-tos/configure-personal-settings/configure-in-ide)

---

## Best Practices & Limitations

### ✅ When to Use Copilot

| Scenario                          | Benefits                          |
| --------------------------------- | --------------------------------- |
| Boilerplate code                  | Saves time on repetitive patterns |
| Learning new languages/frameworks | Quick syntax and pattern examples |
| Test generation                   | Rapid unit test creation          |
| Documentation & comments          | Auto-documentation features       |
| Code optimization suggestions     | Performance improvement ideas     |
| CLI commands & scripts            | Quick shell command generation    |

### ❌ When NOT to Use Copilot

- Highly sensitive or proprietary algorithms
- Critical security infrastructure code
- Compliance-sensitive code requiring explicit documentation
- Code handling personally identifiable information (PII)
- Cryptographic implementations (use audited libraries instead)
- Safety-critical systems (medical devices, transportation)

### Known Limitations & Considerations

#### Context Awareness

- May lack awareness of very large codebases
- Complex domain-specific logic requires manual intervention
- Edge cases and corner cases need developer validation
- Limited understanding of custom business logic

#### Occasional Inaccuracies

- Generated code may contain bugs or inefficiencies
- Security vulnerabilities may appear in suggestions
- Always review and test code before production deployment
- Performance implications may not be obvious

#### Performance Factors

- Large context windows may impact response time
- Processing time varies with prompt complexity
- Network latency affects suggestion latency
- LLM reasoning time increases with complex requests

#### Licensing & Copyright

⚠️ **Legal Considerations:**

- Be aware of licensing implications of suggested code
- Review GPL and other copyleft licenses carefully
- Understand usage rights and attribution requirements
- Risk of license compliance issues in enterprise environments
- Consult legal team for sensitive projects
- **IP indemnity** is included with Business and Enterprise plans

### Testing & Validation Framework

1. **Unit Testing** - Create tests for generated code
2. **Integration Testing** - Verify with existing systems
3. **Code Review** - Peer review all suggestions
4. **Security Scanning** - Run SAST/DAST tools
5. **Performance Testing** - Validate efficiency metrics
6. **Documentation Review** - Ensure accuracy and completeness

---

## Quick Reference: Principles & Commands

### The 4 S's of Effective Prompts

```text
Single   → One clear task
Specific → Detailed instructions  
Short    → Concise phrasing
Surround → Rich file context
```

### Chat Command Quick Guide

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

### Policy Hierarchy Quick Reference

```text
Enterprise  →  Defines or delegates ("No policy")
  └─ Organization  →  Configures within enterprise boundaries
       └─ Team  →  Inherits from org
            └─ Repository  →  Content exclusion only
```

### Key Numbers to Remember

```text
30 minutes    → Max propagation delay for policy/content exclusion changes
150 characters → Context window checked for duplicate/public code matching
< 1%          → Frequency of suggestions matching public code
180 days      → Audit log retention period
$0.04         → Cost per additional premium request
```

---

## Related Resources

- [GitHub Copilot Official Documentation](https://docs.github.com/en/copilot)
- [Microsoft Responsible AI Principles](https://www.microsoft.com/en-us/ai/responsible-ai)
- [GitHub Copilot Exam Preparation Guide](https://assets.ctfassets.net/wfutmusr1t3h/3i7ISEUsTLBgOGrWrML07y/dd586e2b2b607988e2679ed8cce36a76/github-copilot-exam-preparation-study-guide.pdf)
- [Plans for GitHub Copilot](https://docs.github.com/en/copilot/get-started/plans)
- [Content exclusion for GitHub Copilot](https://docs.github.com/en/copilot/concepts/context/content-exclusion)
- [GitHub Copilot policies](https://docs.github.com/en/copilot/concepts/policies)
- [Managing policies as an individual subscriber](https://docs.github.com/en/copilot/how-tos/manage-your-account/manage-policies)
- [Reviewing audit logs for Copilot Business](https://docs.github.com/en/copilot/how-tos/administer-copilot/manage-for-organization/review-activity/review-audit-logs)
- [Finding public code that matches Copilot suggestions](https://docs.github.com/en/copilot/how-tos/get-code-suggestions/find-matching-code)
- [GitHub Copilot Trust Center](https://copilot.github.trust.page)
