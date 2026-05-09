from typing import Optional, Dict, Any, Iterator
import os

class LLMClient:
    """
    Flexible LLM client using LangChain.
    Supports: Groq, Ollama, OpenAI, Gemini, and any OpenAI-compatible API.
    """
    
    def __init__(
        self,
        provider: str = "groq",
        model: str = "llama-3.1-70b-versatile",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        **kwargs
    ):
        self.provider = provider
        self.model_name = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        # Set API key if provided
        if api_key:
            env_var = f"{provider.upper()}_API_KEY"
            os.environ[env_var] = api_key
        
        # Initialize model (Custom factory for LangChain 0.1.20)
        if self.provider.lower() == "groq":
            try:
                from groq import Groq
                class GroqWrapper:
                    def __init__(self, model, temp, max_t, api_key):
                        self.client = Groq(api_key=api_key)
                        self.model = model
                        self.temperature = temp
                        self.max_tokens = max_t

                    def invoke(self, prompt):
                        class Response:
                            def __init__(self, content): self.content = content
                        
                        comp = self.client.chat.completions.create(
                            messages=[{"role": "user", "content": prompt}],
                            model=self.model,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                        )
                        return Response(comp.choices[0].message.content)

                    def stream(self, prompt):
                        class ResponseChunk:
                            def __init__(self, content): self.content = content
                                
                        stream = self.client.chat.completions.create(
                            messages=[{"role": "user", "content": prompt}],
                            model=self.model,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                            stream=True,
                        )
                        for chunk in stream:
                            if chunk.choices[0].delta.content is not None:
                                yield ResponseChunk(chunk.choices[0].delta.content)
                                
                self.model = GroqWrapper(self.model_name, self.temperature, self.max_tokens, api_key)
            except ImportError:
                raise ImportError("Please install groq to use Groq provider")
        elif self.provider.lower() == "ollama":
            from langchain_community.chat_models import ChatOllama
            self.model = ChatOllama(
                model=self.model_name,
                temperature=self.temperature,
                **kwargs
            )
        else:
            raise ValueError(f"Provider {self.provider} not supported in this version.")
    
    def invoke(self, prompt: str) -> str:
        """Generate response (blocking)."""
        response = self.model.invoke(prompt)
        return response.content
    
    def stream(self, prompt: str) -> Iterator[str]:
        """Generate response (streaming)."""
        for chunk in self.model.stream(prompt):
            yield chunk.content
    
    def update_tuning(self, **kwargs):
        """Update generation parameters at runtime."""
        if "temperature" in kwargs:
            self.temperature = kwargs["temperature"]
        if "top_p" in kwargs:
            self.top_p = kwargs["top_p"]
        if "max_tokens" in kwargs:
            self.max_tokens = kwargs["max_tokens"]
        
        # Reinitialize with new params
        if self.provider.lower() == "groq":
            from langchain_groq import ChatGroq
            self.model = ChatGroq(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        elif self.provider.lower() == "ollama":
            from langchain_community.chat_models import ChatOllama
            self.model = ChatOllama(
                model=self.model_name,
                temperature=self.temperature
            )