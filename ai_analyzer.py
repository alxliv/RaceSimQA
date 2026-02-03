"""
AI Analyzer module using OpenAI-compatible chat completions API.

Works with any provider exposing the OpenAI /v1/chat/completions format,
including Ollama (local) and OpenAI (cloud).
"""
import re
import time
import json
from typing import Callable, Optional
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
    """AI-powered analysis using OpenAI-compatible chat completions API.

    Supports local providers (Ollama) and cloud providers (OpenAI) via
    the same endpoint format.  Pass api_key for authenticated providers.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "gpt-oss:20b",
        timeout: int = 120,
        api_key: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.api_key = api_key

        self.headers: dict[str, str] = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    @property
    def _provider_label(self) -> str:
        """Human-readable label for error messages."""
        if "openai.com" in self.base_url:
            return "OpenAI"
        if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
            return "Ollama"
        return "LLM API"

    def _chat_raw(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> dict:
        """Send chat completion request and return the raw JSON response.

        Args:
            messages: Conversation messages
            tools: Optional tool definitions (OpenAI format)

        Returns:
            Parsed JSON response dict, or {"error": "..."} on failure.
        """
        url = f"{self.base_url}/chat/completions"

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 16000,
        }
        if tools:
            payload["tools"] = tools

        try:
            print(f"Posting request to {self.model} LLM.")
            start_time = time.perf_counter()
            response = requests.post(url, json=payload, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            elapsed_sec = time.perf_counter() - start_time
            print(f"Received LLM answer in {elapsed_sec:.2f} sec")
            return response.json()

        except requests.exceptions.ConnectionError:
            hint = " Is it running? Start with: ollama serve" if self._provider_label == "Ollama" else ""
            return {"error": f"Could not connect to {self._provider_label}.{hint}"}
        except requests.exceptions.HTTPError as e:
            detail = ""
            try:
                detail = e.response.json().get("error", {}).get("message", "")
            except Exception:
                detail = e.response.text[:200] if e.response is not None else ""
            return {"error": f"{e}{' — ' + detail if detail else ''}"}
        except requests.exceptions.Timeout:
            return {"error": "Request timed out. The model might be loading or processing a large request."}
        except Exception as e:
            return {"error": str(e)}

    def _chat(self, messages: list[dict]) -> str:
        """Send chat completion request and return the assistant text.

        This is the simple interface used by analyze(), suggest_improvements(),
        etc.  For tool-calling support use chat_with_tools().
        """
        data = self._chat_raw(messages)
        if "error" in data:
            return f"ERROR: {data['error']}"
        return data["choices"][0]["message"].get("content", "")

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_executor: Optional[Callable] = None,
        max_rounds: int = 10,
    ) -> str:
        """Chat with optional tool-calling loop.

        Sends messages to the LLM. If the LLM responds with tool_calls,
        executes them via tool_executor and feeds results back, repeating
        until the LLM produces a text response or max_rounds is reached.

        Args:
            messages: Conversation messages (will not be mutated)
            tools: Tool definitions in OpenAI format
            tool_executor: Callable(name, arguments_dict) -> str
            max_rounds: Maximum tool-calling iterations

        Returns:
            Final assistant text response
        """
        msgs = list(messages)  # shallow copy to avoid mutating caller's list
        use_tools = tools if tool_executor else None

        for round_num in range(max_rounds):
            data = self._chat_raw(msgs, tools=use_tools)

            if "error" in data:
                return f"ERROR: {data['error']}"

            choice = data["choices"][0]
            msg = choice["message"]

            # If the LLM returned tool calls, execute them
            tool_calls = msg.get("tool_calls")

            # Fallback: some models (e.g. smaller llama) emit the tool call
            # as text instead of using the structured tool_calls field.
            # Detect JSON like {"name": "...", "parameters": {...}} in content.
            content = msg.get("content", "")
            if not tool_calls and tool_executor and content:
                parsed = self._parse_tool_call_from_text(content, tools or [])
                if parsed:
                    call_id = f"fallback_{round_num}"
                    tool_calls = [{
                        "id": call_id,
                        "function": {
                            "name": parsed["name"],
                            "arguments": parsed["arguments"],
                        },
                    }]
                    # Replace content so we don't echo the raw JSON back
                    msg = dict(msg)
                    msg["tool_calls"] = tool_calls
                    msg["content"] = ""
                else:
                    # Check if the model tried to call a non-existent tool
                    attempted = self._detect_attempted_tool_name(content)
                    if attempted:
                        known = sorted(
                            t["function"]["name"]
                            for t in (tools or [])
                            if "function" in t
                        )
                        print(f"  Unknown tool '{attempted}' — nudging LLM (round {round_num+1})")
                        msgs.append(msg)
                        msgs.append({
                            "role": "user",
                            "content": (
                                f"The tool '{attempted}' does not exist. "
                                f"Available tools are: {', '.join(known)}. "
                                "Please use one of these tools or answer directly "
                                "with what you know."
                            ),
                        })
                        continue  # Retry with correction

            if tool_calls and tool_executor:
                # Append the assistant message (with tool_calls) to history
                msgs.append(msg)

                for tc in tool_calls:
                    fn_name = tc["function"]["name"]
                    fn_args_raw = tc["function"].get("arguments", "{}")
                    try:
                        fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                    except json.JSONDecodeError:
                        fn_args = {}

                    print(f"  Tool call [{round_num+1}/{max_rounds}]: {fn_name}({fn_args})")
                    try:
                        result = tool_executor(fn_name, fn_args)
                    except Exception as e:
                        result = f"Error executing tool: {e}"

                    msgs.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": str(result),
                    })
                continue  # Loop back to let LLM process tool results

            # No tool calls — return text content
            return content if content else "No response generated."

        return "Reached maximum tool-calling rounds. Here is what I have so far."

    @staticmethod
    def _parse_tool_call_from_text(
        text: str, tools: list[dict]
    ) -> Optional[dict]:
        """Try to extract a tool call from plain-text LLM output.

        Looks for JSON objects containing "name" + "parameters"/"arguments"
        where the name matches one of the known tool names.

        Returns {"name": ..., "arguments": "{...}"} or None.
        """
        known_names = {
            t["function"]["name"] for t in tools if "function" in t
        }
        if not known_names:
            return None

        # Find JSON-like blocks in the text
        for match in re.finditer(r'\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]+\}', text):
            try:
                obj = json.loads(match.group())
            except json.JSONDecodeError:
                continue

            name = obj.get("name")
            if name and name in known_names:
                args = obj.get("parameters") or obj.get("arguments") or {}
                return {
                    "name": name,
                    "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                }
        return None

    @staticmethod
    def _detect_attempted_tool_name(text: str) -> Optional[str]:
        """Detect if text contains a tool-call-like JSON and return the name.

        Unlike _parse_tool_call_from_text, this does NOT filter by known names.
        Used to detect when the LLM hallucinates a non-existent tool.
        """
        for match in re.finditer(
            r'\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]+\}', text
        ):
            try:
                obj = json.loads(match.group())
            except json.JSONDecodeError:
                continue
            name = obj.get("name")
            if name and ("parameters" in obj or "arguments" in obj):
                return name
        return None

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
        """Check if the LLM API is reachable and the model is available."""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
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
            hint = " Start it with: ollama serve" if self._provider_label == "Ollama" else ""
            return False, f"Cannot connect to {self._provider_label}.{hint}"
        except Exception as e:
            return False, f"Error: {str(e)}"

    def analyze_telemetry(
        self,
        telemetry_data: dict,
        summary_data: dict = None
    ) -> str:
        """
        Generate AI analysis of telemetry comparison.

        Args:
            telemetry_data: Dictionary from TelemetryAnalyzer.to_dict()
            summary_data: Optional summary metrics data to combine with telemetry

        Returns:
            AI-generated analysis text
        """
        user_content = """Analyze this racing telemetry comparison between baseline and candidate car configurations.

Focus on:
1. **Threshold crossings**: Which thresholds are violated? New violations vs resolved ones?
2. **Channel changes**: Which telemetry channels show significant differences?
3. **Track position context**: Where on track do the biggest differences occur?
4. **Root cause analysis**: What might explain the observed changes?
5. **Safety assessment**: Are there any critical safety concerns?

Telemetry comparison data:
```json
"""
        user_content += json.dumps(telemetry_data, indent=2)
        user_content += "\n```\n"

        if summary_data:
            user_content += "\nSummary metrics for context:\n```json\n"
            user_content += json.dumps(summary_data, indent=2)
            user_content += "\n```\n"

        user_content += "\nProvide your telemetry analysis:"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        return self._chat(messages)

    def analyze_combined(
        self,
        summary_data: dict,
        telemetry_data: dict
    ) -> str:
        """
        Generate comprehensive analysis combining summary metrics and telemetry.

        Args:
            summary_data: Dictionary from Analyzer.to_dict()
            telemetry_data: Dictionary from TelemetryAnalyzer.to_dict()

        Returns:
            AI-generated comprehensive analysis
        """
        user_content = """Provide a comprehensive analysis of this racing simulation comparison,
combining both summary metrics and detailed telemetry data.

## Summary Metrics
```json
"""
        user_content += json.dumps(summary_data, indent=2)
        user_content += """
```

## Telemetry Analysis
```json
"""
        user_content += json.dumps(telemetry_data, indent=2)
        user_content += """
```

Provide:
1. **Executive Summary**: Overall assessment in 2-3 sentences
2. **Performance Analysis**: How do lap times, speeds, and efficiency compare?
3. **Reliability Analysis**: Tire wear, brake temps, threshold violations
4. **Track Position Insights**: Where does the candidate gain or lose?
5. **Risk Assessment**: Safety concerns, critical violations
6. **Recommendation**: Approve/reject/needs-work with justification
"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        return self._chat(messages)
