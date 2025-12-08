import asyncio
from dataclasses import dataclass

from agents import RunContextWrapper, function_tool
from llm_sandbox import SandboxSession
from llm_sandbox.pool import create_pool_manager, PoolConfig
from llm_sandbox.pool.base import ContainerPoolManager

code_execution_pool = None

def init_code_execution_pool() -> ContainerPoolManager:
    global code_execution_pool

    if code_execution_pool is not None:
        return code_execution_pool

    image = "chihyuyeh/python-data-analytics:0.0.1"
    # default libraries to install
    libraries = []
    skip_environment_setup = True

    code_execution_pool = create_pool_manager(
        backend="docker",
        config=PoolConfig(
            max_pool_size=1,
            min_pool_size=1,
            enable_prewarming=True,
        ),
        lang="python",
        skip_environment_setup=skip_environment_setup,
        image=image,
        verbose=True,
    )

    if len(libraries) > 0 and not skip_environment_setup:
        with SandboxSession(pool=code_execution_pool, verbose=True) as session:
            session.execute_command(f'pip install {" ".join(libraries)}')

    return code_execution_pool


@dataclass
class CodeExecutionContext:
    pool: ContainerPoolManager


@function_tool
async def execute_python_code(
    ctx: RunContextWrapper[CodeExecutionContext],
    code: str,
) -> dict:
    """
    Give a python code to execute in a sandboxed environment and get the result.

    Args:
        code: The python code to execute. Type: str

    Returns:
        A dictionary containing the result of the code execution.
        - success: True if the code executed successfully, False otherwise.
        - error: The error message if the code execution failed, None otherwise.
        - stdout: The stdout of the code execution.
        - stderr: The stderr of the code execution.
    """
    def _run(code: str) -> dict:
        try:
            with SandboxSession(pool=ctx.context.pool, verbose=True) as session:
                result = session.run(code)

                return {
                    "success": result.exit_code == 0,
                    "error": None,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
        except Exception as e:  # noqa: BLE001
            return {
                "success": False,
                "error": str(e),
                "stdout": None,
                "stderr": None,
            }

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run, code)


@function_tool
async def install_python_libraries(
    ctx: RunContextWrapper[CodeExecutionContext],
    libraries: list[str],
) -> dict:
    """
    Install python libraries in the sandboxed environment.

    Args:
        libraries: The python libraries to install in the sandboxed environment. Type: list[str]

    Returns:
        A dictionary containing the result of the library installation.
        - success: True if the libraries installed successfully, False otherwise.
        - error: The error message if the library installation failed, None otherwise.
        - stderr: The stderr of the library installation if the library installation failed, None otherwise.
    """
    def _install(libraries: list[str]) -> dict:
        try:
            with SandboxSession(pool=ctx.context.pool, verbose=True) as session:
                result = session.execute_command(f'pip install {" ".join(libraries)}')

                return {
                    "success": result.exit_code == 0,
                    "error": None,
                    "stderr": result.stderr if result.exit_code != 0 else None,
                }
        except Exception as e:  # noqa: BLE001
            return {
                "success": False,
                "error": str(e),
                "stderr": None,
            }

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _install, libraries)
