# GitHub Copilot Practice Workbook

This guide is organized to align with the **official GH-300 “Skills measured” outline (updated significantly in January 2026; Microsoft Learn page last updated 2026-02-19)**.  
Use it as a checklist + practice workbook, not as a “read once” document. ([learn.microsoft.com](https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/gh-300?utm_source=openai))

---

## Table of Contents

1. [Use GitHub Copilot Responsibly (15–20%)](#use-github-copilot-responsibly-1520)
2. [Use GitHub Copilot Features (25–30%)](#use-github-copilot-features-2530)
3. [Understand GitHub Copilot Data and Architecture (10–15%)](#understand-github-copilot-data-and-architecture-1015)
4. [Apply Prompt Engineering and Context Crafting (10–15%)](#apply-prompt-engineering-and-context-crafting-1015)
5. [Improve Developer Productivity with GitHub Copilot (10–15%)](#improve-developer-productivity-with-github-copilot-1015)
6. [Configure Privacy, Content Exclusions, and Safeguards (10–15%)](#configure-privacy-content-exclusions-and-safeguards-1015)
7. [Plans & Licensing (Exam-Relevant Summary)](#plans--licensing-exam-relevant-summary)
8. [Scenario Practice (Exam-Style)](#scenario-practice-exam-style)
9. [Quick Reference](#quick-reference)
10. [References](#references)

---

## Use GitHub Copilot Responsibly (15–20%)

This domain is about *responsible use*, not “Copilot is cool.”

### Microsoft Responsible AI principles (high level)

- Fairness
- Reliability & Safety
- Privacy & Security
- Inclusiveness
- Transparency
- Accountability

### Practical responsibilities (what the exam tends to probe)

- **Human-in-the-loop**: you are accountable for correctness, security, and license compliance.
- **Don’t paste secrets/PII** into prompts. Treat prompts as potentially sensitive.
- **Verify**: run tests, linters, security scans; review diffs carefully.

### Quick drills

- For a suggested snippet, identify:
  1) the likely bug/edge case  
  2) a security concern  
  3) a license risk (if any)

---

## Use GitHub Copilot Features (25–30%)

You should recognize “where Copilot lives” and the kinds of tasks it can do on each surface.

### Common surfaces

- IDE chat + inline completions (VS Code / JetBrains / Visual Studio, etc.)
- GitHub.com experiences (where enabled by org policy)
- CLI experiences (e.g., via GitHub CLI integrations)
- Issues/PR assistance (drafting descriptions, summaries, suggestions)

> Tip: For exam questions, “which feature is available where” is often more important than memorizing exact UI labels.

### Chat features & common patterns

#### References (bring context)

- `#` file references (attach specific file(s) to the question)
- `@workspace` (use broader solution context where supported)

#### Intent modifiers (`/` commands)

| Command     | Purpose                | Example use             |
| ----------- | ---------------------- | ----------------------- |
| `/explain`  | Clarify what code does | Understand legacy logic |
| `/fix`      | Suggest a fix          | Error, failing test     |
| `/tests`    | Generate tests         | Unit test scaffolding   |
| `/doc`      | Add docs/comments      | Document public APIs    |
| `/optimize` | Improve efficiency     | Hot path performance    |

---

## Understand GitHub Copilot Data and Architecture (10–15%)

The exam expects you to understand, at a practical level:

- what context is sent,
- what systems enforce safety controls,
- and what governance tools exist to review activity.

### Mental model: request/response flow (conceptual)

1. Context is assembled (open files, selection, surrounding code, instructions)
2. Requests are transmitted to Copilot services
3. Safety and policy controls apply
4. Suggestions are generated and returned to the client

(Implementation details vary across products and can change; focus on the governance and risk implications.)

### What to be able to explain in one minute

- Why “surrounding context” improves answers but increases data exposure risk
- Why filtering/policy enforcement matters (harmful content, data handling)
- Why auditability matters (org governance)

---

## Apply Prompt Engineering and Context Crafting (10–15%)

### The 4 S’s (keep, but make it exam-usable)

| Principle    | Definition                                    | Exam takeaway              |
| ------------ | --------------------------------------------- | -------------------------- |
| **Single**   | One task/question                             | Avoid multi-goal prompts   |
| **Specific** | Constraints + acceptance criteria             | “Done means…”              |
| **Short**    | Minimal but sufficient detail                 | Less noise, better answers |
| **Surround** | Provide real context (files, names, examples) | Attach relevant files      |

### Add what’s missing: acceptance criteria + constraints

A strong exam-ready prompt includes:

- **Goal** (what to build/change)
- **Constraints** (versions, libraries, style, security)
- **Inputs/outputs** (API shape, examples)
- **Acceptance criteria** (“must have tests”, “no breaking changes”, etc.)
- **Context** (files, errors, logs)

#### Template prompt

```text
Task:
- ...

Constraints:
- ...

Context:
- Relevant files: #fileA #fileB
- Error/logs: ...

Acceptance criteria:
- ...
```

---

## Improve Developer Productivity with GitHub Copilot (10–15%)

This domain is “how to use Copilot to move faster *safely*.”

### Common workflows

- **Refactor** with guardrails: ask for small diffs; verify with tests
- **Debug**: ask for hypotheses, then ask for targeted experiments
- **Modernize**: convert patterns (callbacks → async/await, etc.) with incremental PRs
- **Docs**: generate API docs, then validate against code behavior
- **Tests**: generate tests from observed behavior and bug reports

### Testing with Copilot (make this explicit)

When generating tests, always ask for:

- happy path + edge cases
- negative cases (invalid input)
- mocks/stubs only where needed
- deterministic tests (avoid time/race randomness)

---

## Configure Privacy, Content Exclusions, and Safeguards (10–15%)

This is one of the most commonly under-studied sections.

### Organization/Enterprise policies (what to know)

GitHub provides **Copilot policies** that control availability of features/models and (for privacy policies) “Allowed/Blocked” style enforcement. Policies can be set at **enterprise level** and may override organization-level settings. ([docs.github.com](https://docs.github.com/copilot/concepts/policies?utm_source=openai))

#### Where policies live (org level)

Organization owners typically navigate to:

- **Organization settings → Copilot → Policies / Models** to configure enforcement options. ([docs.github.com](https://docs.github.com/copilot/managing-copilot/managing-github-copilot-in-your-organization/setting-policies-for-copilot-in-your-organization?utm_source=openai))

#### Setup steps you should recognize

Setting up Copilot for an org includes:

- subscribe/enable Copilot
- set policies
- networking allowlists if required
- grant access to members ([docs.github.com](https://docs.github.com/en/copilot/how-tos/set-up/setting-up-github-copilot-for-your-organization?utm_source=openai))

### Audit logs for Copilot Business (governance)

Audit logs are available for organizations on **Copilot Business** and can show events such as:

- policy/setting changes
- seat assignments/removals

Audit logs keep events for **the last 180 days** and can be searched using qualifiers like:

- `action:copilot`
- `action:copilot.cfb_seat_assignment_created` ([docs.github.com](https://docs.github.com/copilot/managing-github-copilot-in-your-organization/reviewing-audit-logs-for-copilot-business?utm_source=openai))

#### Exam drill

- Describe how you would confirm *who changed a Copilot policy* and *when*.

---

## Plans & Licensing (Exam-Relevant Summary)

Avoid memorizing marketing pages. Focus on what the exam tests:

- Who manages seats (org owner / enterprise admin)
- Which plans support org policies and audit logs
- Governance expectations for Business/Enterprise

> Note: Plan pricing and packaging can change; treat plan *capabilities* as the durable concept, and verify details against GitHub Docs/Microsoft Learn during final prep. ([learn.microsoft.com](https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/gh-300?utm_source=openai))

---

## Scenario Practice (Exam-Style)

Use these as “short answer” drills. Write your answer in 5–8 bullet points.

### Scenario 1 — Safe refactor

You must refactor a critical auth module. What prompt and constraints do you provide to reduce risk?

### Scenario 2 — Sensitive data

A teammate wants to paste production logs that may include tokens into Copilot Chat. What do you recommend?

### Scenario 3 — Policy governance

Your organization wants to disable a Copilot feature across all members. Where do you do this, and what happens if the enterprise sets a conflicting policy? ([docs.github.com](https://docs.github.com/copilot/concepts/policies?utm_source=openai))

### Scenario 4 — Auditability (Business)

Security asks: “Who assigned Copilot seats last month?” What audit log search would you use and where do you find the audit log? ([docs.github.com](https://docs.github.com/copilot/managing-github-copilot-in-your-organization/reviewing-audit-logs-for-copilot-business?utm_source=openai))

### Scenario 5 — Test generation

A bug report includes steps to reproduce. How do you ask Copilot to generate tests that prevent regressions?

---

## Quick Reference

### Prompt checklist (exam-friendly)

- Single task
- Constraints
- Context attached
- Acceptance criteria
- Request small diffs + explain rationale
- Ask for tests

### Admin checklist (Business/Enterprise)

- Policies (features/models/privacy) configured
- Seat assignment process defined
- Audit logs reviewed periodically ([docs.github.com](https://docs.github.com/copilot/managing-github-copilot-in-your-organization/reviewing-audit-logs-for-copilot-business?utm_source=openai))

---

## References

- Microsoft Learn — GH-300 study guide / skills measured (updated significantly Jan 2026; last updated 2026-02-19) ([learn.microsoft.com](https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/gh-300?utm_source=openai))
- GitHub Docs — Copilot policies concepts (org vs enterprise policy behavior) ([docs.github.com](https://docs.github.com/copilot/concepts/policies?utm_source=openai))
- GitHub Docs — Managing policies and features for Copilot in an organization ([docs.github.com](https://docs.github.com/copilot/managing-copilot/managing-github-copilot-in-your-organization/setting-policies-for-copilot-in-your-organization?utm_source=openai))
- GitHub Docs — Setting up Copilot for your organization ([docs.github.com](https://docs.github.com/en/copilot/how-tos/set-up/setting-up-github-copilot-for-your-organization?utm_source=openai))
- GitHub Docs — Reviewing audit logs for Copilot Business (180-day window + search qualifiers) ([docs.github.com](https://docs.github.com/copilot/managing-github-copilot-in-your-organization/reviewing-audit-logs-for-copilot-business?utm_source=openai))
