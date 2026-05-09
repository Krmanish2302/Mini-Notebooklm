from typing import Optional, Iterator
import os
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Flexible LLM client supporting Groq, Ollama, OpenAI, and Gemini.

    All paths expose the same two-method contract:
        invoke(prompt: str) -> str
        stream(prompt: str) -> Iterator[str]

    ``update_tuning()`` mutates temperature / max_tokens at runtime and
    rebuilds self.model using the same code path as __init__ so there is
    no inconsistency between the two.
    """

    SUPPORTED_PROVIDERS = {"groq", "ollama", "openai", "gemini"}

    def __init__(
        self,
        provider: str = "groq",
        model: str = "llama-3.1-70b-versatile",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        **kwargs,
    ):
        self.provider = provider.lower().strip()
        self.model_name = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._extra_kwargs = kwargs

        if self.provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Provider '{provider}' not supported. "
                f"Choose from: {sorted(self.SUPPORTED_PROVIDERS)}"
            )

        self.model = self._build_model()

    # ── Internal factory ──────────────────────────────────────────────────────

    def _build_model(self):
        """Build (or rebuild) the underlying model object from current params."""
        api_key = self._api_key

        if self.provider == "groq":
            try:
                from groq import Groq
            except ImportError:
                raise ImportError("pip install groq  # required for Groq provider")

            client = Groq(api_key=api_key)
            temperature = self.temperature
            max_tokens = self.max_tokens
            model_name = self.model_name

            class _GroqWrapper:
                def invoke(self, prompt: str) -> "_GroqWrapper.Response":
                    class Response:
                        def __init__(self, content: str):
                            self.content = content

                    comp = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return Response(comp.choices[0].message.content)

                def stream(self, prompt: str):
                    class Chunk:
                        def __init__(self, content: str):
                            self.content = content

                    for chunk in client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                    ):
                        delta = chunk.choices[0].delta.content
                        if delta is not None:
                            yield Chunk(delta)

            return _GroqWrapper()

        elif self.provider == "ollama":
            try:
                from langchain_community.chat_models import ChatOllama
            except ImportError:
                raise ImportError("pip install langchain-community  # required for Ollama")
            return ChatOllama(
                model=self.model_name,
                temperature=self.temperature,
                **self._extra_kwargs,
            )

        elif self.provider == "openai":
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("pip install openai  # required for OpenAI provider")

            client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
            temperature = self.temperature
            max_tokens = self.max_tokens
            model_name = self.model_name

            class _OpenAIWrapper:
                def invoke(self, prompt: str):
                    class Response:
                        def __init__(self, content): self.content = content
                    comp = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return Response(comp.choices[0].message.content)

                def stream(self, prompt: str):
                    class Chunk:
                        def __init__(self, content): self.content = content
                    for chunk in client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                    ):
                        delta = chunk.choices[0].delta.content
                        if delta:
                            yield Chunk(delta)

            return _OpenAIWrapper()

        elif self.provider == "gemini":
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError("pip install google-generativeai  # required for Gemini")

            genai.configure(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
            client = genai.GenerativeModel(self.model_name)
            temperature = self.temperature
            max_tokens = self.max_tokens

            class _GeminiWrapper:
                def invoke(self, prompt: str):
                    class Response:
                        def __init__(self, content): self.content = content
                    resp = client.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=temperature,
                            max_output_tokens=max_tokens,
                        ),
                    )
                    return Response(resp.text)

                def stream(self, prompt: str):
                    class Chunk:
                        def __init__(self, content): self.content = content
                    for chunk in client.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=temperature,
                            max_output_tokens=max_tokens,
                        ),
                        stream=True,
                    ):
                        if chunk.text:
                            yield Chunk(chunk.text)

            return _GeminiWrapper()

    # ── Public interface ──────────────────────────────────────────────────────

    def invoke(self, prompt: str) -> str:
        """Generate a full response (blocking). Always returns a plain str."""
        response = self.model.invoke(prompt)
        return response.content

    def stream(self, prompt: str) -> Iterator[str]:
        """Generate a response token by token. Yields plain str tokens."""
        for chunk in self.model.stream(prompt):
            yield chunk.content

    def update_tuning(self, **kwargs) -> None:
        """Update generation parameters at runtime and rebuild the model.

        Accepts: temperature, top_p, max_tokens
        Rebuilds self.model via the same _build_model() factory used in
        __init__ so Groq, OpenAI, Gemini, and Ollama are all consistent.
        """
        changed = False
        for key in ("temperature", "top_p", "max_tokens"):
            if key in kwargs:
                setattr(self, key, kwargs[key])
                changed = True

        if changed:
            self.model = self._build_model()
            logger.info(
                "LLMClient: tuning updated — temp=%.2f max_tokens=%d",
                self.temperature,
                self.max_tokens,
            )
