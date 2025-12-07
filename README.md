# Simple Code Agent

Using [Streamlit](https://github.com/streamlit/streamlit) to demo simple code agent using [OpenAI Agent SDK](https://github.com/openai/openai-agents-python) and [llm-sandbox](https://github.com/vndee/llm-sandbox) under the hood.

## Setup

- install Python 3.12.*
- install [Poetry](https://github.com/python-poetry/poetry)
- run `poetry install`
- generate `.env` based on `.env.example` and fill in environment variables
    - Langfuse(highly recommended): for easier trace tracking and debugging
    - OpenAI API Key
- run `poetry run streamlit run app.py`
