import anthropic
from typing import List, Optional, Dict, Any

# Maximum number of tool-use rounds per query before Claude must answer.
MAX_TOOL_ROUNDS = 5


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = f""" You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- Search again only if the first results are insufficient or the question spans several courses or lessons; at most {MAX_TOOL_ROUNDS} searches per query
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters. Sonnet 5.5 always thinks (adaptive) and
        # rejects non-default temperature; thinking counts toward max_tokens.
        self.base_params = {
            "model": self.model,
            "max_tokens": 16000,
            # Fields newer than the pinned SDK version are sent via extra_*
            "extra_body": {
                "output_config": {"effort": "low"},
                "fallbacks": "default"  # server-side retry on another model if refused
            },
            "extra_headers": {"anthropic-beta": "server-side-fallback-2026-07-01"}
        }

    @staticmethod
    def _extract_text(response) -> str:
        """Return the answer text; responses may start with thinking blocks"""
        if response.stop_reason == "refusal":
            return "Sorry, I can't help with that request."
        return "".join(block.text for block in response.content if block.type == "text")
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Agentic loop: let Claude call tools until it answers, up to MAX_TOOL_ROUNDS.
        # system and tools stay identical on every call (replayed thinking blocks
        # depend on it); only the messages grow.
        messages = api_params["messages"]
        response = self.client.messages.create(**api_params)
        rounds = 0
        while response.stop_reason == "tool_use" and tool_manager and rounds < MAX_TOOL_ROUNDS:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": self._run_tools(response, tool_manager)})
            rounds += 1
            if rounds >= MAX_TOOL_ROUNDS:
                # Out of rounds: force a text answer.
                api_params = {**api_params, "tool_choice": {"type": "none"}}
            response = self.client.messages.create(**api_params)

        return self._extract_text(response)

    @staticmethod
    def _run_tools(response, tool_manager) -> List[Dict[str, Any]]:
        """Execute the tool calls in a response and return their tool_result blocks"""
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                result = {"content": tool_manager.execute_tool(block.name, **block.input)}
            except Exception as e:
                # Report the failure to Claude instead of failing the whole query.
                result = {"content": f"Error: {e}", "is_error": True}
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, **result})
        return tool_results
