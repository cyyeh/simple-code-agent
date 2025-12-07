import os
from collections.abc import Callable

from agents import Agent, ModelSettings, Runner, SQLiteSession
from agents.extensions.models.litellm_model import LitellmModel
from openai.types.shared import Reasoning

from tools.code_execution_tool import CodeExecutionContext, execute_python_code


def init_agent():
    if os.getenv("OPENAI_API_KEY") is not None:
        model = "gpt-5-mini-2025-08-07"
    else:
        model = LitellmModel(
            model="openai/gpt-oss:20b",
            base_url=os.getenv("OLLAMA_ENDPOINT"),
        )

    agent_instructions = (
        "### ROLE ###\n"
        "You are a code agent that can write python code to solve problems.\n\n"
        "### Instructions ###\n"
        "- ALWAYS reason about the problem first to determine the best approach.\n"
        "- ALWAYS write `print` statement to return the result of the code execution.\n"
        "### AVAILABLE TOOLS ###\n"
        "- execute_python_code: Execute python code in a sandboxed environment and get the result.\n"
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
        tools=[execute_python_code],
    )

    return agent


async def run_agent(
    code_agent: Agent,
    pool,
    user_query: str,
    session: SQLiteSession,
    reasoning_output_callback: Callable[[str], None],
    output_callback: Callable[[str], None],
) -> tuple[str, str]:
    """Run the agent asynchronously and return the output text and reasoning text."""
    result = Runner.run_streamed(
        code_agent,
        user_query,
        context=CodeExecutionContext(pool=pool),
        session=session,
    )

    reasoning_text = ""
    output_text = ""
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
                    reasoning_text += _reasoning_text
            if event.item.type == "message_output_item":
                if event.item.raw_item.content:
                    for content in event.item.raw_item.content:
                        _output_text += f"{content.text}\n\n"
                if _output_text:
                    output_callback(_output_text)
                    output_text += _output_text
    
    return output_text, reasoning_text

