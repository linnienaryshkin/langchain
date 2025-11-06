# [LangChain](https://github.com/linnienaryshkin/langchain)

Sandbox for the [LandChain framework](https://docs.langchain.com/oss/javascript/langchain/overview) and LLM overall topics learning.

---

# Usage

**Install dependencies**

```bash
npm ci
```

**Anthropic API keys**

[Anthropic API keys](https://console.anthropic.com/settings/keys)

```bash
touch .env && echo "ANTHROPIC_API_KEY=your-api-key-here" > .env
```

**Run the script**

For each script, there is a corresponding launching command in the script file. E.g.

```bash
tsx src/quickstart.ts |& tee src/quickstart.json
```

**Track your balance**

[Anthropic Credit balance](https://console.anthropic.com/settings/billing)

**Update dependencies**

```bash
npm update --save && npm audit fix
```
