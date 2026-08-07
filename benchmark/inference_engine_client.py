import os
import subprocess
import time
import requests
import base64
import mimetypes
from pathlib import Path
import httpx
from openai import OpenAI, AsyncOpenAI
from benchmark.utils import _start_log_tailer

class InferenceEngineClient:
    """
    Wrapper for an OpenAI‐compatible server.
    launch() will call your existing launch_engine.sh script (which runs Docker in the foreground),
    then poll /v1/completions every second until it returns 200.
    """

    def __init__(self, base_url="http://127.0.0.1:23333/v1", api_key="none"):
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=httpx.Timeout(60.0))
        self._launcher_proc = None
        self.model = None
        self.NAME = "bench360_inference_engine"

    def launch(self, backend: str, model: str, timeout: float = 500.0, dump_server_output: bool = False, script_path: str = None):
        """
        1) Starts your existing launch_engine.sh in a Popen (non-blocking).
        2) Polls `http://127.0.0.1:23333/v1/models` every 2 seconds
           until the desired model shows up (or timeout).

        :param backend: One of {tgi, vllm, mii, sglang, lmdeploy}
        :param model: HF model ID or local path
        :param timeout: Max seconds to wait for the model to appear before raising.
        """
        if script_path is None:
            script_path = os.path.join(os.getcwd(), "benchmark/launch_engine.sh")
        if not os.path.isfile(script_path):
            raise FileNotFoundError(f"Cannot find '{script_path}'.")
        if not os.access(script_path, os.X_OK):
            raise PermissionError(f"'{script_path}' is not executable (chmod +x missing).")

        # 1) Start the launcher in a subprocess.
        cmd = [
            script_path,
            f"--engine={backend}",
            f"--model={model}",
            f"--name={self.NAME}"
        ]
        self._launcher_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,  # ⟵ was DEVNULL
            stderr=subprocess.STDOUT,  # merge both streams
            text=True,
            bufsize=1  # line-buffered
        )

        # Start a daemon thread to read the launcher output
        _start_log_tailer(self, max_lines=10, dump_server_output=dump_server_output)

        # 2) Poll /v1/models until our model_id appears or timeout.
        list_url = self.base_url + "/models"
        start_time = time.time()

        while True:
            # If the launcher process died, report error
            if self._launcher_proc.poll() is not None:
                raise RuntimeError(
                    f"Launcher script exited early with code {self._launcher_proc.returncode}"
                )

            try:
                resp = requests.get(list_url, timeout=2.0)
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    # Check if our model_id is in the loaded list
                    for entry in data:
                        if entry.get("id").lower() == model.lower():
                            # Model is loaded and ready to serve
                            self.model = model
                            return
                # If status is not 200 or model not found yet, keep waiting
            except requests.exceptions.RequestException:
                # Connection refused, etc. → server not up yet
                pass

            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(
                    f"Waited {timeout}s for model '{model}' to appear at {list_url}, but it never did."
                )
            time.sleep(2.0)

    def _format_messages(self, messages, images=None):
        """Helper to standardize message formatting for both sync and async calls."""
        if isinstance(messages, str):
            if images:
                if isinstance(images, str):
                    images = [images]
                content_list = [{"type": "text", "text": messages}]
                for img in images:
                    processed_uri = self._prepare_image_input(img)
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": processed_uri}
                    })
                return [{"role": "user", "content": content_list}]
            else:
                return [{"role": "user", "content": messages}]
        return messages

    def chat_completion(
            self,
            messages,
            images: str | list[str] | None = None,
            model: str | None = None,
            temperature: float = 0.1,
            max_tokens: int = 64,
            top_p: float = 0.9,
            stream: bool = False,
    ):
        """
        Send a request to the chat.completions endpoint.
        """
        model_to_use = model or self.model
        formatted_messages = self._format_messages(messages, images)

        # Create your base arguments
        api_kwargs = {
            "model": model_to_use,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream,
        }

        # Only add chat_template_kwargs if the model isn't a Mistral/Ministral model
        if "mistral" not in model_to_use.lower():
            api_kwargs["extra_body"] = {
                'chat_template_kwargs': {'enable_thinking': False}
            }

        # Unpack the arguments into the client call
        resp = self.client.chat.completions.create(**api_kwargs)

        if stream:
            return resp

        return resp.choices[0].message.content

    async def async_chat_completion(
            self,
            messages,
            images: str | list[str] | None = None,
            model: str | None = None,
            temperature: float = 0.1,
            max_tokens: int = 64,
            top_p: float = 0.9,
    ):
        """
        Send an asynchronous request to the chat.completions endpoint using AsyncOpenAI.
        """
        model_to_use = model or self.model
        formatted_messages = self._format_messages(messages, images)

        # 1. Initialize the async client using an async context manager
        # This ensures all httpx connections are cleanly closed before the event loop ends.
        async with AsyncOpenAI(
                api_key=self.client.api_key,
                base_url=self.base_url,
                timeout=httpx.Timeout(600.0)
        ) as async_client:
            # 2. Define the base parameters that work for all models
            api_kwargs = {
                "model": model_to_use,
                "messages": formatted_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            }

            # 3. Conditionally add the template arguments ONLY if it's not a Mistral model
            if "mistral" not in model_to_use.lower():
                api_kwargs["extra_body"] = {
                    'chat_template_kwargs': {'enable_thinking': False}
                }

            # 4. Unpack the dictionary into the async call using **
            resp = await async_client.chat.completions.create(**api_kwargs)
            return resp.choices[0].message.content

    def completion(
        self,
        prompt,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 64,
        top_p: float = 0.9,
        stream: bool = False,
    ):
        """
        Send one or more prompts. :param prompt: string or list[str]
        """
        is_batch = isinstance(prompt, (list, tuple))

        resp = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=stream,
        )

        if stream:
            return resp

        texts = [c.text for c in resp.choices]
        return texts if is_batch else texts[0]

    def warmup(self, num_iters: int = 3):
        """
        Send a few small dummy requests using the chat endpoint to load the model into memory
        and JIT any kernels so that subsequent inference calls are faster.
        """
        messages = [{"role": "user", "content": "Warmup request."}]

        for _ in range(num_iters):
            try:
                _ = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=1,
                    temperature=0.1,
                )
            except Exception:
                # If the server isn't ready yet, retry after a short sleep
                time.sleep(1.0)
                continue

    def measure_ttft(self) -> float:
        """
        Issue a streaming chat request and measure the time until the first chunk arrives.
        """
        import time
        messages = [{
            "role": "user",
            "content": (
                "Artificial intelligence is a rapidly evolving field with applications in "
                "healthcare, finance, education, and more. One of the most transformative "
                "technologies is"
            )
        }]

        start = time.time()
        stream_resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1,
            temperature=0.1,
            stream=True,
        )

        first_token_time = None
        for chunk in stream_resp:
            # In chat.completions, the text chunk is stored in delta.content
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)

            # If content exists and is not an empty string, we caught the first token
            if content:
                first_token_time = time.time()
                break

        if first_token_time is None:
            # Fallback if the stream ends without yielding valid text
            first_token_time = time.time()

        return first_token_time - start

    def close(self):
        """
        Stop Docker container, terminate subprocess,
        stop background log tailer (if running), and close HTTP client.
        """
        # 1) Attempt to find and stop the container(s) on port 23333
        try:
            subprocess.run(
                ["docker", "stop", self.NAME],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            pass  # Docker might not be running

        # 2) Terminate the launcher subprocess if still alive
        if hasattr(self, "_launcher_proc") and self._launcher_proc:
            if self._launcher_proc.poll() is None:
                self._launcher_proc.terminate()
                try:
                    self._launcher_proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._launcher_proc.kill()

        # 3) Stop the log tailer thread if present
        if hasattr(self, "_stop_tail") and callable(getattr(self, "_stop_tail", None)):
            self._stop_tail.set()
            if hasattr(self, "_tail_thread"):
                self._tail_thread.join(timeout=1.0)

        # 4) Close HTTP client
        try:
            self.client.close()
        except Exception:
            pass

    def _prepare_image_input(self, image_input: str) -> str:
        """
        Helper method to detect if the input is a local file path.
        If it is, it reads the file and returns a Base64 data URI.
        Otherwise, it assumes it's already a URL or data URI and returns it as-is.
        """
        # Check if the string is a valid path to an existing local file
        if os.path.isfile(image_input):
            # Guess the mime type (e.g., image/jpeg, image/png) based on extension
            mime_type, _ = mimetypes.guess_type(image_input)
            mime_type = mime_type or "image/jpeg"  # Fallback

            with open(image_input, "rb") as f:
                encoded_string = base64.b64encode(f.read()).decode("utf-8")

            return f"data:{mime_type};base64,{encoded_string}"

        # If it's not a local file, assume it's a web URL or pre-encoded Base64 string
        return image_input


if __name__ == "__main__":

    # --- Generate Local Dummy Images for Offline Testing ---
    # 1x1 Red Pixel
    img1_path = "test_red.png"
    img1_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    if not os.path.exists(img1_path):
        with open(img1_path, "wb") as f:
            f.write(base64.b64decode(img1_b64))

    # 1x1 Blue Pixel
    img2_path = "test_blue.png"
    img2_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    if not os.path.exists(img2_path):
        with open(img2_path, "wb") as f:
            f.write(base64.b64decode(img2_b64))

    # Instantiate the client directly
    client = InferenceEngineClient()

    try:
        # Launching with the actual multimodal model
        client.launch(backend="sglang", model="Qwen/Qwen3-VL-2B-Instruct")
        client.warmup()

        # 1. Benchmark TTFT
        ttft = client.measure_ttft()
        print(f"TTFT: {ttft:.3f} seconds")

        # 2. Test Case: Single Image (Local Path -> Base64)
        print("\n--- Testing Single Image (Fully Offline) ---")
        single_resp = client.chat_completion(
            messages="What color is the main subject in this image?",
            images=img1_path  # The helper will auto-detect and base64-encode this local file
        )
        print("Response:\n", single_resp)

        # 3. Test Case: Multiple Images (Local Paths -> Base64)
        print("\n--- Testing Multiple Images (Fully Offline) ---")
        multi_resp = client.chat_completion(
            messages="Are these two images identical? Explain the differences.",
            images=[img1_path, img2_path]  # Both local files will be base64-encoded automatically
        )
        print("Response:\n", multi_resp)

    finally:
        # Ensures Docker container and subprocesses are always shut down cleanly
        client.close()

        # Cleanup temporary local dummy files
        if os.path.exists(img1_path):
            os.remove(img1_path)
        if os.path.exists(img2_path):
            os.remove(img2_path)