"""
LangGraph workflow for the predictive maintenance assistant.

Uses a direct tool-call loop to avoid message format incompatibilities
between LangGraph's ToolNode and the google-genai SDK.
"""

import os
import json
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

from .tools import AVAILABLE_TOOLS, initialize_tools
from .prompts import GENERAL_ASSISTANT_PROMPT


def create_llm(model: str = "models/gemini-2.5-flash", temperature: float = 0.1):
    """Create the LLM instance."""
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        convert_system_message_to_human=True
    )


class PredictiveMaintenanceAssistant:
    """
    Main assistant class that handles tool calls manually.

    Avoids LangGraph ToolNode message format issues with the google-genai SDK
    by executing tools directly and building the response.
    """

    def __init__(
        self,
        model_dir: str = "./models",
        data_path: str = "./CMAPSSData",
        dataset_id: str = "FD001",
        llm_model: str = "models/gemini-2.5-flash"
    ):
        self.model_dir = model_dir
        self.data_path = data_path
        self.dataset_id = dataset_id

        # Initialize tools with models
        initialize_tools(model_dir, data_path, dataset_id)

        # Create LLM with tools bound
        self.llm = create_llm(llm_model)
        self.llm_with_tools = self.llm.bind_tools(AVAILABLE_TOOLS)

        # Build a tool lookup map
        self.tool_map = {t.name: t for t in AVAILABLE_TOOLS}

        # System prompt
        self.system_prompt = GENERAL_ASSISTANT_PROMPT

    def chat(self, message: str, history: List[BaseMessage] = None) -> str:
        """
        Process a user message, handle tool calls, return final text.
        """
        messages = [SystemMessage(content=self.system_prompt)]
        if history:
            messages.extend(history)
        messages.append(HumanMessage(content=message))

        # Allow up to 5 rounds of tool calls
        for _ in range(5):
            response = self.llm_with_tools.invoke(messages)

            # If no tool calls, we have the final answer
            if not getattr(response, 'tool_calls', None):
                return self._extract_text(response)

            # Execute each tool call and collect results
            tool_results = []
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]

                if tool_name in self.tool_map:
                    try:
                        result = self.tool_map[tool_name].invoke(tool_args)
                    except Exception as e:
                        result = f"Error running {tool_name}: {e}"
                else:
                    result = f"Unknown tool: {tool_name}"

                tool_results.append({"name": tool_name, "result": str(result)})

            # Build a summary of tool results and ask the LLM to respond
            tool_summary = "\n\n".join(
                f"[Tool: {tr['name']}]\n{tr['result']}" for tr in tool_results
            )

            # Replace message history with: system + user + tool summary
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=message),
                HumanMessage(content=f"Here are the results from the tools I called:\n\n{tool_summary}\n\nPlease provide a comprehensive response based on these results.")
            ]

        # If we exhausted the loop, return whatever we have
        return self._extract_text(response)

    def _extract_text(self, message) -> str:
        """Extract text content from an AI message."""
        if hasattr(message, 'content'):
            content = message.content
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and 'text' in part:
                        text_parts.append(part['text'])
                    elif isinstance(part, str):
                        text_parts.append(part)
                return "".join(text_parts) or "I processed your request but couldn't generate a text response."
            return content if content else "I processed your request but couldn't generate a text response."
        return str(message)


def create_simple_chain(llm_model: str = "models/gemini-2.5-flash"):
    """Create a simple chain without tools for basic queries."""
    llm = create_llm(llm_model)
    prompt = ChatPromptTemplate.from_messages([
        ("system", GENERAL_ASSISTANT_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ])
    return prompt | llm


def create_assistant(
    model_dir: str = "./models",
    data_path: str = "./CMAPSSData",
    dataset_id: str = "FD001"
) -> PredictiveMaintenanceAssistant:
    """Factory function to create the assistant."""
    return PredictiveMaintenanceAssistant(
        model_dir=model_dir,
        data_path=data_path,
        dataset_id=dataset_id
    )
