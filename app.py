import asyncio
import atexit
import uuid

import streamlit as st
from agents import SQLiteSession
from dotenv import load_dotenv
from langfuse import get_client
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

from agent import init_agent, run_agent
from tools.code_execution import init_code_execution_pool


@st.cache_resource
def init_langfuse_and_agent():
    """
    This function is cached so it only runs once per Streamlit session.
    It sets up Langfuse, the code execution pool, and the code agent.
    """
    load_dotenv()
    OpenAIAgentsInstrumentor().instrument()

    langfuse = get_client()
    if not langfuse.auth_check():
        print("WARNING: Langfuse auth failed. Check credentials and host.")

    # Init code execution pool once
    pool = init_code_execution_pool()

    # Make sure pool is closed when the process exits (e.g., Ctrl-C on streamlit run)
    atexit.register(lambda: pool.close())

    # Init agent
    agent = init_agent()

    return agent, pool


def render_chat_history(messages: list[dict]):
    for msg in messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                for output in msg["content"]:
                    if output["type"] == "reasoning":
                        render_reasoning_output(output["content"])
                    elif output["type"] == "code":
                        render_code_output(output["content"])
                    elif output["type"] == "output":
                        render_text_output(output["content"])
            else:
                st.markdown(msg["content"])


def render_reasoning_output(reasoning_text: str):
    with st.expander("Reasoning", expanded=True):
        st.markdown(f"{reasoning_text}\n\n")


def render_code_output(code: str):
    with st.expander("Code", expanded=True):
        st.code(code, language="python", line_numbers=True)


def render_text_output(output_text: str):
    st.markdown(f"{output_text}\n\n")


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

    render_chat_history(st.session_state.messages)

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
                outputs = asyncio.run(run_agent(
                    code_agent,
                    pool,
                    user_input,
                    st.session_state.session,
                    reasoning_output_callback=render_reasoning_output,
                    code_output_callback=render_code_output,
                    text_output_callback=render_text_output,
                ))

        st.session_state.messages.append(
            {"role": "assistant", "content": outputs}
        )


if __name__ == "__main__":
    main()
