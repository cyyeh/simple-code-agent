import asyncio
from dataclasses import dataclass

from agents import RunContextWrapper, function_tool
from llm_sandbox import SandboxSession
from llm_sandbox.pool import create_pool_manager, PoolConfig
from llm_sandbox.pool.base import ContainerPoolManager

code_execution_pool = None

def init_code_execution_pool(libraries: list[str] | None = None) -> ContainerPoolManager:
    global code_execution_pool

    if code_execution_pool is not None:
        return code_execution_pool

    code_execution_pool = create_pool_manager(
        backend="docker",
        config=PoolConfig(
            max_pool_size=1,
            min_pool_size=1,
            enable_prewarming=True,
        ),
        lang="python",
        image="docker.io/python:3.12-bullseye",
        verbose=True,
    )

    if libraries is not None and len(libraries) > 0:
        with SandboxSession(pool=code_execution_pool, verbose=True) as session:
            session.execute_command(f'pip install {" ".join(libraries)}')

    return code_execution_pool


@dataclass
class CodeExecutionContext:
    pool: ContainerPoolManager


@function_tool
async def execute_python_code(ctx: RunContextWrapper[CodeExecutionContext],code: str) -> dict:
    """
    Give a python code to execute in a sandboxed environment and get the result.

    Args:
        code: The python code to execute.

    Returns:
        A dictionary containing the result of the code execution.
        - success: True if the code executed successfully, False otherwise.
        - stdout: The stdout of the code execution.
        - stderr: The stderr of the code execution.
        - exit_code: The exit code of the code execution.
        - error: The error message if the code execution failed, None otherwise.
    """
    def _run(code: str) -> dict:
        try:
            with SandboxSession(pool=ctx.context.pool, verbose=True) as session:
                result = session.run(code)

                return {
                    "success": result.exit_code == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.exit_code,
                    "error": None,
                }
        except Exception as e:  # noqa: BLE001
            return {
                "success": False,
                "error": str(e),
                "exit_code": -1,
                "stdout": None,
                "stderr": None,
            }

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run, code)
