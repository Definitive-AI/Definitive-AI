# AI Agents: An Overview

AI agents are specialized systems designed to perform tasks or solve problems autonomously. They come in various types, each with unique capabilities and use cases. These agents can range from simple tool-calling mechanisms to complex reasoning systems. Below is a list of different AI agent types and their characteristics:

- **ReAct**: The ReAct agent is an AI agent that can use multiple tools, reason over the next action, construct an action command, execute the action, and repeat these steps in an iterative loop until all the tasks are complete.

- **Tool Calling**: Tool calling agent enables the agent with 1 task (no more than 1) to use 1 tool with multiple inputs, however it cannot reason over the next action, use multiple tools or context, repeat steps or handle multiple tasks.

- **OpenAI Tools**: If you are using a recent OpenAI model (1106 onwards). Generic Tool Calling agent recommended instead.

- **OpenAI Functions**: If you are using an OpenAI model, or an open-source model that has been finetuned for function calling and exposes the same functions parameters as OpenAI. Generic Tool Calling agent recommended instead.

- **XML**: A ReAct agent but with XML outputs.

- **Structured Chat**: The structured chat agent is a limited ReAact agent capable of using multi-input tools, but only to interact/chat with a human.

- **JSON Chat**: A ReAct agent but with JSON outputs.

- **Self Ask With Search**: If you are using a simple model and only have one search tool, but only to interact/chat with a human.

- **Reflection**: Reflection enables an AI Agent to observe its past steps to assess, review and improve the quality of the chosen output. It can not use tools or context, or human input, but just improves its own output.

- **Reflexion**: Is an architecture designed to learn through verbal feedback and self-reflection. The agent explicitly critiques its responses for tasks to generate a higher quality final response, at the expense of longer execution time. It can use tools.

- **ReWOO**: Reasoning without Observation is an agent that combines a multi-step planner and variable substitution for effective tool use. It separates reasoning from tool interactions to avoid redundant token usage, thereby significantly reducing computational costs while maintaining performance.
