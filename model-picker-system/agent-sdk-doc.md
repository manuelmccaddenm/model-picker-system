OpenAI Agents SDK Overview

The OpenAI Agents SDK is an open-source Python framework for building agentic AI applications. An agent is essentially an LLM (ChatGPT-style model) configured with a name, system instructions, and optional tools. The SDK provides a small set of primitives – Agents, Handoffs, Guardrails, and Sessions – that simplify multi-agent orchestration
openai.github.io
openai.github.io
. In practice, you define one or more agents (each with instructions and tools) and use a Runner to execute a loop of LLM calls, tool invocations, and potential agent-to-agent handoffs until a final answer is produced
openai.github.io
github.com
. Built-in tracing and logging let you visualize and debug the agent workflows
github.com
.

Setup and installation

Environment: Requires Python 3.9+ and an OpenAI API key. (Set OPENAI_API_KEY in your environment before running agents.)

Install: Run pip install openai-agents
openai.github.io
. Optional extras enable extra features: e.g. use pip install 'openai-agents[voice]' for speech support or pip install 'openai-agents[redis]' for Redis-based sessions
github.com
.

Core Concepts and Architecture

Agent: An Agent is the core building block – essentially an LLM “assistant” with a given name and instructions (system prompt). You can also specify its model (e.g. a GPT-4.5) and tuning parameters (temperature, max tokens, etc.)
openai.github.io
. Each agent can be given a list of tools. During execution, the agent’s LLM may choose to call a tool rather than directly answer. For example:

from agents import Agent, function_tool
@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."
agent = Agent(
    name="WeatherBot",
    instructions="Answer questions and use tools as needed.",
    tools=[get_weather]
)


Here get_weather is turned into a tool the agent can invoke. The SDK also provides built-in tools (like WebSearchTool, FileSearchTool, CodeInterpreterTool, etc.) for common actions
openai.com
. When the LLM outputs a tool call (e.g. {"tool": "get_weather", "input": {"city": "Tokyo"}}), the Runner executes the function and feeds the result back to the agent.

Handoffs (Agent-to-Agent): Agents can delegate tasks to other agents via handoffs. An agent’s handoffs list names other Agent instances it can transfer control to. For example, a “Triage Agent” might have handoffs=[billing_agent, refund_agent]. In this SDK, a handoff appears to the LLM as a special tool call (e.g. transfer_to_refund_agent)
openai.github.io
. When the LLM issues that handoff tool, the Runner automatically switches to the designated agent and continues the loop there. This lets you compose multi-agent workflows (e.g. language-based triage or specialist agents) naturally
openai.github.io
.

Guardrails: Guardrails are safety/validation checks that run in parallel with the agent. You can attach input-guardrails or output-guardrails (simple Python checks or even small agents) to an Agent. For example, you might run a cheap model to check whether an input question is disallowed. If a guardrail “tripwire” is triggered (e.g. detects malicious intent), it can abort the run before calling the main LLM
openai.github.io
. This helps avoid wasteful or dangerous calls. Guardrails can use Pydantic models and return structured signals to halt or modify the workflow.

Sessions (Memory): By default each Runner.run() is stateless. The SDK provides built-in session classes (like SQLiteSession or RedisSession) to automatically persist conversation history. When you pass a session to Runner.run, the agent’s entire message history is stored and re-used on subsequent calls
github.com
. This means follow-up queries see prior context “for free” without manual prompt-chaining. For example, after answering “Golden Gate Bridge is in San Francisco”, the agent will recall that if you ask “What state is it in?”
github.com
.

Agent Loop (Planning/Execution): When you invoke an agent, the Runner enters a loop of “plan and act” steps
github.com
. In each turn it: 1) calls the LLM (with the agent’s instructions and current context); 2) checks the LLM output – if it includes a final answer (no pending tools/handoffs), the loop ends and returns that output; if it includes a handoff, the loop switches to the new agent and continues; if it includes tool calls, the Runner executes each tool and appends the results to the conversation; 3) repeats the process. This implicit planning loop continues until an answer is produced. You can cap the number of iterations via the max_turns parameter to avoid runaway loops
github.com
.

Tracing/Observability: The SDK automatically traces each step (agent prompts, model responses, tool inputs/outputs) for debugging and analysis
github.com
. Traces can be exported to monitoring services (Logfire, AgentOps, etc.) or viewed in the OpenAI evaluation tools. This makes it easy to visualize what each agent did at each step.

Implementing and Running a Custom Agent

Define agents and tools: In Python, import the SDK and create Agent objects with names and instructions. Attach any tools (functions or built-ins) via the tools list.
openai.github.io
github.com
 For example:

from agents import Agent, Runner, function_tool
@function_tool
def double(x: int) -> int: return x*2
math_agent = Agent(
    name="MathAgent",
    instructions="I can do arithmetic. Always answer with numbers.",
    tools=[double]
)


Run the agent: Use Runner.run(agent, prompt) for async code or Runner.run_sync(agent, prompt) for sync. For example:

result = Runner.run_sync(math_agent, "What is 6 * 7?")
print(result.final_output)  # e.g. "42"


The returned result object has .final_output (the answer string or model output). If you set an output_type with a Pydantic model on the agent, you can also use result.final_output_as(MyModel) to get structured output. Remember to handle max_turns or catch errors if the agent loops too long.

Use sessions (optional): To maintain context, create a session and pass it each run:

from agents import SQLiteSession
session = SQLiteSession("user123")
Runner.run_sync(math_agent, "First question", session=session)
Runner.run_sync(math_agent, "Second question", session=session)


The agent will see prior Q&A from the same session automatically
github.com
.

Examples of Usage

Hello World: Create a simple agent and run a one-off query. For instance, an agent instructed to write haikus returns:

agent = Agent(name="Poet", instructions="You are a haiku poet.")
result = Runner.run_sync(agent, "Write a haiku about recursion.")
print(result.final_output)


Output: “Code within the code… / Functions calling themselves… / Infinite loop’s dance.”
github.com
.

Multi-Agent Handoff: Define specialist agents and a triage agent. For example, one agent speaks only Spanish, another only English, and a triage agent routes inputs to the right one:

spanish = Agent(name="Spanish", instructions="Solo hablas español.")
english = Agent(name="English", instructions="You only speak English.")
triage = Agent(
    name="Triage",
    instructions="Route to Spanish or English agent based on input language.",
    handoffs=[spanish, english]
)
result = Runner.run_sync(triage, "¿Cómo estás?")
print(result.final_output)  # Spanish response


The SDK automatically makes handoffs into special tool calls, so the triage agent can seamlessly transfer control
openai.github.io
github.com
.

Function Tools: Any Python function can become a tool with @function_tool. In the README example, a get_weather(city) function becomes a tool an agent can call:

@function_tool
def get_weather(city: str) -> str: 
    return f"The weather in {city} is sunny."
weather_agent = Agent(
    name="WeatherAgent",
    instructions="Tell the weather.",
    tools=[get_weather]
)
result = Runner.run_sync(weather_agent, "What's the weather in Tokyo?")
print(result.final_output)  # "The weather in Tokyo is sunny."


This works by the LLM outputting a JSON tool call which Runner executes
github.com
.

Hosted Tools: The SDK includes tools for common APIs. For example, you can use WebSearchTool() to let an agent search the web, or FileSearchTool() to query a vector database. In the blog announcement, a “Shopping Assistant” agent used tools=[WebSearchTool()] to answer product queries
openai.com
. Any tool may incur API usage costs per OpenAI’s pricing.

Sessions/Multi-turn: With sessions, the agent remembers the conversation. For example, using SQLiteSession("conv123"):

Ask “What city is the Golden Gate Bridge in?” → “San Francisco”
github.com
.

Then ask “What state is it in?” without re-stating context. The agent recalls the bridge and answers “California”
github.com
.

Best Practices and Limitations

Limit loop iterations: Always set max_turns (or detect when to stop) to avoid infinite loops
github.com
. By default the loop runs until the agent returns a final output with no pending tools or handoffs.

Use guardrails: Attach input/output guardrails for safety. For instance, run a cheap model to flag disallowed queries; if a guardrail triggers, it can halt the agent early
openai.github.io
. This prevents misuse (e.g. guarding against unwanted instructions before invoking an expensive model).

Structured outputs: When expecting data, set an agent output_type with a Pydantic model so the SDK knows how to recognize the final answer format
github.com
. This ensures the loop stops only when valid structured output is produced.

Maintain context appropriately: Use Sessions for multi-turn dialogue. Ensure session storage (SQLite file, Redis, etc.) is managed securely. Without a session, each run is independent.

Observability: Leverage the built-in tracing to log and analyze agent behavior
github.com
. This is invaluable for debugging and evaluation.

Language and API limits: The SDK currently supports Python (Node.js support is planned)
openai.com
. It works with OpenAI’s Chat Completions and Responses APIs (and any model offering a Chat endpoint)
openai.com
. Note that some built-in tools (like WebSearchTool, FileSearchTool, ComputerTool) require the Responses API and may incur extra cost.

LLM pitfalls: Remember agents rely on LLM capabilities; they can still hallucinate or make mistakes. Handle exceptions (the SDK has specific exceptions for tool errors or guardrail tripwires) and design fallbacks. Always test agents thoroughly with realistic prompts.

Key Resources

GitHub repo: openai/openai-agents-python (source code, README, examples)

Official docs: OpenAI Agents SDK Documentation (detailed guides and API reference)

OpenAI Dev Blog: “New tools for building agents” announcement (describes SDK features)

Cookbook & Tutorials: OpenAI Cookbook Agents topic and community blog posts for examples.