import os
import asyncio
import atexit
import uuid

import streamlit as st
from dotenv import load_dotenv

from agents import Agent, ModelSettings, Runner, SQLiteSession
from agents.extensions.models.litellm_model import LitellmModel
from langfuse import get_client
from openai.types.shared import Reasoning
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

from tools.code_execution_tool import (
    init_code_execution_pool,
    execute_python_code,
    CodeExecutionContext,
)

# -------------------------------------------------------------------
# One-time setup (env, instrumentation, langfuse)
# -------------------------------------------------------------------
load_dotenv()
OpenAIAgentsInstrumentor().instrument()


@st.cache_resource
def init_langfuse_and_agent():
    """
    This function is cached so it only runs once per Streamlit session.
    It sets up Langfuse, the code execution pool, and the code agent.
    """
    langfuse = get_client()
    if not langfuse.auth_check():
        print("WARNING: Langfuse auth failed. Check credentials and host.")

    # Init code execution pool once
    pool = init_code_execution_pool(libraries=["numpy", "pandas"])

    # Make sure pool is closed when the process exits (e.g., Ctrl-C on streamlit run)
    atexit.register(lambda: pool.close())

    # Choose model
    if os.getenv("OPENAI_API_KEY") is not None:
        model = "gpt-5-mini-2025-08-07"
    else:
        model = LitellmModel(
            model="openai/gpt-oss:20b",
            base_url=os.getenv("OLLAMA_ENDPOINT"),
        )

    code_agent_instruction = (
        "### ROLE ###\n"
        "You are a code agent that can write python code to solve problems.\n\n"
        "### Instructions ###\n"
        "- ALWAYS reason about the problem first to determine the best approach.\n"
        "- ALWAYS write `print` statement to return the result of the code execution.\n"
        "### AVAILABLE TOOLS ###\n"
        "- execute_python_code: Execute python code in a sandboxed environment and get the result.\n"
    )

    code_agent = Agent[CodeExecutionContext](
        name="code-agent",
        instructions=code_agent_instruction,
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

    return code_agent, pool


async def run_agent(
    code_agent: Agent,
    pool,
    user_query: str,
    session: SQLiteSession,
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
                    with st.expander("Reasoning"):
                        st.markdown(f"{_reasoning_text}\n\n")
                    reasoning_text += _reasoning_text
            if event.item.type == "message_output_item":
                if event.item.raw_item.content:
                    for content in event.item.raw_item.content:
                        _output_text += f"{content.text}\n\n"
                if _output_text:
                    st.markdown(f"{_output_text}\n\n")
                    output_text += _output_text
    
    return output_text, reasoning_text


def main():
    st.set_page_config(page_title="Code Agent Chat", page_icon="💻")
    st.title("💻 Code Agent Chat with Sandbox Execution")

    # Initialize agent + pool (cached)
    code_agent, pool = init_langfuse_and_agent()

    # Chat history state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Conversation session
    if "session" not in st.session_state:
        st.session_state.session = SQLiteSession(f"conversation_session_{str(uuid.uuid4())}")

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and "reasoning" in msg and msg["reasoning"]:
                with st.expander("Reasoning"):
                    st.markdown(msg["reasoning"])
            st.markdown(msg["content"])

    # User input
    user_input = st.chat_input("Ask me something that may need Python code…")

    if user_input:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Run agent and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # We wrap the async agent call in asyncio.run for Streamlit
                output_text, reasoning_text = asyncio.run(run_agent(
                    code_agent,
                    pool,
                    user_input,
                    st.session_state.session,
                ))

        st.session_state.messages.append(
            {"role": "assistant", "content": output_text, "reasoning": reasoning_text}
        )


if __name__ == "__main__":
    main()
