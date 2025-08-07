import requests
import websocket
import json
import uuid
import logging
from typing import Any, Dict

from .base import BaseTool

LOGGER = logging.getLogger(__name__)

class CodeExecutorTool(BaseTool):
    """
    A tool for executing Python code in a sandboxed Jupyter environment.
    It communicates with a Jupyter Kernel Gateway service to provide a
    persistent and isolated execution session.
    """

    def __init__(self, sandbox_url: str = "http://localhost:8888"):
        self.api_url = f"{sandbox_url}/api/kernels"
        self.ws_url = f"{sandbox_url.replace('http', 'ws')}/api/kernels"
        self._session = requests.Session()
        self._kernel_id = None

    def _start_kernel(self) -> None:
        """Starts a new Jupyter kernel and stores its ID."""
        if self._kernel_id:
            return
        try:
            LOGGER.info("No active kernel found. Starting a new one...")
            response = self._session.post(self.api_url, headers={'Content-Type': 'application/json'})
            response.raise_for_status()
            kernel = response.json()
            self._kernel_id = kernel['id']
            LOGGER.info(f"Started new kernel with ID: {self._kernel_id}")
        except requests.exceptions.RequestException as e:
            LOGGER.error(f"Failed to start new kernel: {e}")
            raise

    def run(self, code: str, **kwargs: Any) -> str:
        """
        Executes the given Python code in the sandbox and returns the result.
        """
        self._start_kernel()
        if not self._kernel_id:
            return "Error: Kernel could not be started. Cannot execute code."

        ws_url = f"{self.ws_url}/{self._kernel_id}/channels"
        LOGGER.info(f"Connecting to websocket: {ws_url}")
        
        try:
            ws = websocket.create_connection(ws_url)
        except Exception as e:
            return f"Error connecting to websocket: {e}"

        # Construct the Jupyter execution message
        msg_id = uuid.uuid4().hex
        msg = {
            "header": {
                "msg_id": msg_id,
                "username": "user",
                "session": uuid.uuid4().hex,
                "msg_type": "execute_request",
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
            },
            "channel": "shell",
        }

        output = ""
        try:
            ws.send(json.dumps(msg))
            
            while True:
                response_str = ws.recv()
                response = json.loads(response_str)
                
                # Check if this message corresponds to our request
                if response.get("parent_header", {}).get("msg_id") != msg_id:
                    continue

                msg_type = response["msg_type"]
                content = response["content"]

                if msg_type == "stream":
                    output += content["text"]
                elif msg_type == "execute_result":
                    output += content["data"].get("text/plain", "")
                elif msg_type == "error":
                    output += f"Error: {content['ename']}\n{content['evalue']}\n"
                    output += "\n".join(content["traceback"])
                elif msg_type == "status" and content["execution_state"] == "idle":
                    # Idle status means execution is finished
                    break
        
        except Exception as e:
            output += f"\nAn error occurred during websocket communication: {e}"
        finally:
            ws.close()

        if not output:
            return "Code executed successfully with no output."
        
        return output.strip()

    @property
    def name(self) -> str:
        return "code_executor"

    @property
    def signature(self) -> str:
        return f"{self.name}(code: str) -> str"

    @property
    def description(self) -> str:
        return (
            "Executes a block of Python code in a sandboxed Jupyter environment. "
            "Use this for calculations, file manipulation, data analysis, etc. "
            "The code runs in a persistent session, so you can define variables "
            "and use them in subsequent calls. The final expression's result is "
            "automatically printed. To see a dataframe, you must print(df)."
        )
