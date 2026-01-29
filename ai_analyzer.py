"""
AI Analyzer module using Ollama with OpenAI-compatible API.
"""

import json
from typing import Optional
import requests


SYSTEM_PROMPT = """You are an expert racing simulation QA engineer analyzing telemetry data.

Your task is to analyze comparison results between a baseline car configuration and a candidate
(modified) configuration. You will receive structured data showing:
- Overall performance score and status
- Per-metric comparisons (baseline vs candidate statistics)
- Requirement violations if any

Provide clear, actionable insights including:
1. A brief executive summary (2-3 sentences)
2. What improved and why it matters
3. What regressed and potential causes
4. Requirement violations and their severity
5. Overall recommendation (approve/reject/needs review)

Use racing domain knowledge to explain potential causes:
- Lower lap times with higher fuel consumption might indicate more aggressive engine mapping
- Higher tire degradation with better cornering G might indicate softer compound or higher downforce
- Temperature increases might indicate brake or cooling issues

Be concise and technical. Format using markdown for readability.
"""


class AIAnalyzer:
    """AI-powered analysis using Ollama's OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "gpt-oss:20b",      # "llama3.2",
        timeout: int = 120
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _chat(self, messages: list[dict]) -> str:
        """Send chat completion request to Ollama."""
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,  # Lower temperature for more consistent analysis
            "max_tokens": 2000,
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            return data["choices"][0]["message"]["content"]

        except requests.exceptions.ConnectionError:
            return "ERROR: Could not connect to Ollama. Is it running? Start with: ollama serve"
        except requests.exceptions.Timeout:
            return "ERROR: Request timed out. The model might be loading or processing a large request."
        except Exception as e:
            return f"ERROR: {str(e)}"

    def analyze(
        self,
        analysis_data: dict,
        additional_context: Optional[str] = None
    ) -> str:
        """
        Generate AI analysis of the comparison results.

        Args:
            analysis_data: Dictionary from Analyzer.to_dict()
            additional_context: Optional extra context (car changes, experiment goals, etc.)

        Returns:
            AI-generated analysis text
        """
        # Build the user message
        user_content = f"""Analyze this racing simulation comparison:

```json
{json.dumps(analysis_data, indent=2)}
```
"""

        if additional_context:
            user_content += f"\nAdditional context:\n{additional_context}\n"

        user_content += "\nProvide your analysis:"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        return self._chat(messages)

    def suggest_improvements(
        self,
        analysis_data: dict,
        focus_areas: Optional[list[str]] = None
    ) -> str:
        """
        Generate suggestions for improving car performance.

        Args:
            analysis_data: Dictionary from Analyzer.to_dict()
            focus_areas: Optional list of metrics to focus on

        Returns:
            AI-generated suggestions
        """
        user_content = f"""Based on this simulation analysis, suggest specific improvements:

```json
{json.dumps(analysis_data, indent=2)}
```
"""

        if focus_areas:
            user_content += f"\nFocus especially on these areas: {', '.join(focus_areas)}\n"

        user_content += """
Suggest concrete parameter changes or design modifications that could improve performance.
Consider trade-offs between metrics (e.g., speed vs. fuel efficiency).
"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        return self._chat(messages)

    def compare_multiple_batches(
        self,
        batch_analyses: list[dict]
    ) -> str:
        """
        Compare multiple candidate batches to find the best configuration.

        Args:
            batch_analyses: List of analysis dictionaries from multiple batches

        Returns:
            AI-generated comparison and recommendation
        """
        user_content = f"""Compare these {len(batch_analyses)} car configurations and recommend the best one:

```json
{json.dumps(batch_analyses, indent=2)}
```

Rank them and explain your reasoning. Consider:
1. Overall score
2. Which requirements are met/violated
3. Trade-offs between configurations
"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        return self._chat(messages)

    def check_connection(self) -> tuple[bool, str]:
        """Check if Ollama is running and the model is available."""
        try:
            # Check if we can reach the API
            response = requests.get(
                f"{self.base_url}/models",
                timeout=5
            )
            if response.status_code == 200:
                models = response.json().get("data", [])
                model_ids = [m.get("id", "") for m in models]
                if any(self.model in m for m in model_ids):
                    return True, f"Connected. Model '{self.model}' available."
                else:
                    available = ", ".join(model_ids[:5])
                    return False, f"Model '{self.model}' not found. Available: {available}"
            return False, f"Unexpected status: {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, "Cannot connect to Ollama. Start it with: ollama serve"
        except Exception as e:
            return False, f"Error: {str(e)}"
