import os
from collections.abc import Callable

import orjson
from agents import Agent, ModelSettings, Runner, SQLiteSession
from agents.extensions.models.litellm_model import LitellmModel
from openai.types.shared import Reasoning
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall

from tools.code_execution_tool import (
    CodeExecutionContext,
    execute_python_code,
    install_python_libraries,
)


def init_agent():
    if os.getenv("OPENAI_API_KEY") is not None:
        model = "gpt-5-mini-2025-08-07"
    else:
        model = LitellmModel(
            model="ollama_chat/gpt-oss:20b",
            base_url=os.getenv("OLLAMA_ENDPOINT"),
        )

    agent_instructions = (
        "### ROLE ###\n"
        "You are a code agent that can write python code to solve problems.\n\n"
        "### Instructions ###\n"
        "- ALWAYS reason about the problem first to determine the best approach.\n"
        "- ALWAYS write `print` statement to return the result of the code execution.\n"
        "- For any package that is not standard python package, you could check if it is available in the sandboxed environment first and then install it if it is not available.\n"
        "- DON'T show any code in the final text output. Final text output should be human readable and concise.\n\n"
        "### AVAILABLE TOOLS ###\n"
        "- execute_python_code: Execute python code in a sandboxed environment and get the result.\n"
        "- install_python_libraries: Install python libraries in the sandboxed environment.\n"
    )

    agent = Agent[CodeExecutionContext](
        name="code-agent",
        instructions=agent_instructions,
        model=model,
        model_settings=ModelSettings(
            max_tokens=4096,
            reasoning=Reasoning(
                effort="medium",
                summary="detailed",
            ),
        ),
        tools=[execute_python_code, install_python_libraries],
    )

    return agent


async def run_agent(
    code_agent: Agent,
    pool,
    user_query: str,
    session: SQLiteSession,
    reasoning_output_callback: Callable[[str], None],
    code_output_callback: Callable[[str], None],
    text_output_callback: Callable[[str], None],
) -> list[dict]:
    """Run the agent asynchronously and return the outputs."""
    result = Runner.run_streamed(
        code_agent,
        user_query,
        context=CodeExecutionContext(pool=pool),
        session=session,
    )

    outputs = []
    async for event in result.stream_events():
        _reasoning_text = ""
        _output_text = ""
        if event.type == "run_item_stream_event":
            if event.item.type == "reasoning_item":
                if event.item.raw_item.summary:
                    for summary in event.item.raw_item.summary:
                        _reasoning_text += f"{summary.text}\n\n"
                if _reasoning_text:
                    reasoning_output_callback(_reasoning_text)
                    outputs.append({"type": "reasoning", "content": _reasoning_text})
            elif (
                event.item.type == "tool_call_item" and 
                isinstance(event.item.raw_item, ResponseFunctionToolCall) and 
                event.item.raw_item.name == "execute_python_code"
            ):
                json_arguments = orjson.loads(event.item.raw_item.arguments)
                if code := json_arguments.get("code"):
                    code_output_callback(code)
                    outputs.append({"type": "code", "content": code})
            elif event.item.type == "message_output_item":
                if event.item.raw_item.content:
                    for content in event.item.raw_item.content:
                        _output_text += f"{content.text}\n\n"
                if _output_text:
                    text_output_callback(_output_text)
                    outputs.append({"type": "output", "content": _output_text})
    
    return outputs
