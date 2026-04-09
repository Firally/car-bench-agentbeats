"""
CAR-bench Agent - Purple agent that solves CAR-bench tasks.

This is the agent being tested. It:
1. Receives task descriptions with available tools from the green agent
2. Decides which tool to call or how to respond
3. Returns responses in the expected JSON format wrapped in <json>...</json> tags
"""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path
import sys
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, Message, Part, TextPart, DataPart, Role
from a2a.utils import new_agent_parts_message
from litellm import completion
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_utils import configure_logger
from tool_call_types import ToolCall, ToolCallsData
sys.path.pop(0)

logger = configure_logger(role="agent", context="-")

SYSTEM_PROMPT = """You are a helpful car voice assistant. Follow the policy and tool instructions provided."""


def check_policy_violations(tool_calls: list[dict], previous_tool_names: list[str], current_tool_names: list[str]) -> list[str]:
    """Check if tool calls would violate AUT-POL rules based on conversation history.

    Returns a list of violation descriptions with remediation guidance.
    """
    violations = []

    for tc in tool_calls:
        tc_name = tc["function"]["name"]
        tc_args = json.loads(tc["function"]["arguments"])

        # AUT-POL:005 + AUT-POL:009 — Sunroof prerequisites
        if tc_name == "open_close_sunroof" and tc_args.get("percentage", 0) != 0:
            if "get_weather" not in previous_tool_names:
                violations.append(
                    "AUT-POL:009 VIOLATION: You must call get_weather for the current location BEFORE opening the sunroof. "
                    "Call get_weather first, then open the sunroof on the next turn."
                )
            if "open_close_sunshade" not in current_tool_names:
                violations.append(
                    "AUT-POL:005 VIOLATION: The sunroof can only be opened if the sunshade is fully open or being opened in parallel. "
                    "You must call open_close_sunshade(percentage=100) in the SAME turn as open_close_sunroof."
                )

        # AUT-POL:010 — Window defrost prerequisites
        if tc_name == "set_window_defrost" and tc_args.get("on") is True:
            if tc_args.get("defrost_window") in ("ALL", "FRONT"):
                has_climate_check = "get_climate_settings" in previous_tool_names
                has_manual_setup = (
                    "set_fan_speed" in previous_tool_names
                    and "set_fan_airflow_direction" in previous_tool_names
                    and "set_air_conditioning" in previous_tool_names
                )
                if not has_climate_check and not has_manual_setup:
                    violations.append(
                        "AUT-POL:010 VIOLATION: Before activating front/all window defrost, you must first either: "
                        "(a) call get_climate_settings to check current state, OR "
                        "(b) call set_fan_speed (>=2), set_fan_airflow_direction (include WINDSHIELD), and set_air_conditioning (on). "
                        "Do these prerequisite calls first, then activate defrost on the next turn."
                    )

        # AUT-POL:011 — AC prerequisites
        if tc_name == "set_air_conditioning" and tc_args.get("on") is True:
            has_climate_check = "get_climate_settings" in previous_tool_names
            has_window_check = (
                "get_vehicle_window_positions" in previous_tool_names
                or "open_close_window" in previous_tool_names
            )
            has_fan_speed = "set_fan_speed" in previous_tool_names or "set_fan_speed" in current_tool_names
            # Must have either climate check OR window check
            if not has_climate_check and not has_window_check:
                violations.append(
                    "AUT-POL:011 VIOLATION: Before turning AC on, you must check window positions first. "
                    "Call get_vehicle_window_positions (or get_climate_settings) to check state, "
                    "then close any window open >20% and ensure fan_speed >= 1 before activating AC."
                )
            # Must have fan_speed set (even if climate was checked, fan_speed could still be 0)
            if not has_climate_check and not has_fan_speed:
                violations.append(
                    "AUT-POL:011 VIOLATION: Before turning AC on, fan_speed must be at least 1. "
                    "Call set_fan_speed(level=1) or higher before activating AC."
                )

        # AUT-POL:013 — Fog lights prerequisites
        if tc_name == "set_fog_lights" and tc_args.get("on") is True:
            has_lights_check = "get_exterior_lights_status" in previous_tool_names
            has_manual_setup = (
                "set_head_lights_low_beams" in previous_tool_names
                and "set_head_lights_high_beams" in previous_tool_names
            )
            if not has_lights_check and not has_manual_setup:
                violations.append(
                    "AUT-POL:013 VIOLATION: Before activating fog lights, you must first either: "
                    "(a) call get_exterior_lights_status to check current state, OR "
                    "(b) call set_head_lights_low_beams(on=true) and set_head_lights_high_beams(on=false). "
                    "Do these prerequisite calls first."
                )
            # Also check weather prerequisite for fog lights (LLM-POL:008)
            if "get_weather" not in previous_tool_names:
                violations.append(
                    "LLM-POL:008 VIOLATION: Before activating fog lights, you must call get_weather to check "
                    "current weather conditions. Call get_weather first, then enable fog lights on the next turn."
                )

        # AUT-POL:014 — High beams prerequisites
        if tc_name == "set_head_lights_high_beams" and tc_args.get("on") is True:
            has_lights_check = "get_exterior_lights_status" in previous_tool_names
            has_fog_control = "set_fog_lights" in previous_tool_names
            if not has_lights_check and not has_fog_control:
                violations.append(
                    "AUT-POL:014 VIOLATION: Before activating high beams, you must first either: "
                    "(a) call get_exterior_lights_status to check fog lights, OR "
                    "(b) call set_fog_lights(on=false). "
                    "Do these prerequisite calls first."
                )

    # Check: If navigation was already set up (set_new_navigation called before),
    # don't call set_new_navigation again — use editing tools instead
    for tc in tool_calls:
        tc_name = tc["function"]["name"]
        if tc_name == "set_new_navigation" and "set_new_navigation" in previous_tool_names:
            violations.append(
                "NAVIGATION VIOLATION: Navigation is already active (set_new_navigation was called earlier). "
                "To modify the route, use navigation editing tools: navigation_add_one_waypoint, "
                "navigation_replace_one_waypoint, navigation_replace_final_destination, "
                "navigation_delete_one_waypoint, navigation_delete_final_destination. "
                "Do NOT call set_new_navigation again."
            )

    # TECH-AUT-POL:018 — Navigation editing tools parallel restriction
    navigation_edit_tools = [
        "navigation_add_one_waypoint",
        "navigation_delete_final_destination",
        "navigation_delete_one_waypoint",
        "navigation_replace_final_destination",
        "navigation_replace_one_waypoint",
    ]
    nav_edit_count = sum(1 for tc in tool_calls if tc["function"]["name"] in navigation_edit_tools)
    if nav_edit_count > 1:
        violations.append(
            "TECH-AUT-POL:018 VIOLATION: Only one navigation editing tool can be used per turn. "
            "Use navigation editing tools one at a time in sequence, not in parallel."
        )

    return violations


class CARBenchAgentExecutor(AgentExecutor):
    """Executor for the CAR-bench purple agent using native tool calling."""

    def __init__(self, model: str, temperature: float = 0.0, thinking: bool = False, reasoning_effort: str = "medium", interleaved_thinking: bool = False):
        self.model = model
        self.temperature = temperature
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort  # Can be 'none', 'disable', 'low', 'medium', 'high', or integer token budget
        self.interleaved_thinking = interleaved_thinking  # Whether to use interleaved thinking
        self.ctx_id_to_messages: dict[str, list[dict]] = {}
        self.ctx_id_to_tools: dict[str, list[dict]] = {}
        self.ctx_id_to_previous_tool_names: dict[str, list[str]] = {}  # Track all tool calls per context

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        inbound_message = context.message
        ctx_logger = logger.bind(role="agent", context=f"ctx:{context.context_id[:8]}")
        
        # Initialize or get conversation history
        if context.context_id not in self.ctx_id_to_messages:
            self.ctx_id_to_messages[context.context_id] = []

        messages = self.ctx_id_to_messages[context.context_id]
        tools = self.ctx_id_to_tools.get(context.context_id, [])

        # Parse the incoming A2A Message with Parts
        user_message_text = None
        incoming_tool_results = None  # Structured tool results from green agent
        
        try:
            for part in inbound_message.parts:
                if isinstance(part.root, TextPart):
                    text = part.root.text
                    # Parse system prompt and user message from formatted text
                    if "System:" in text and "\n\nUser:" in text:
                        # First message with system prompt
                        parts = text.split("\n\nUser:", 1)
                        system_prompt = parts[0].replace("System:", "").strip()
                        user_message_text = parts[1].strip()
                        if not messages:  # Only add system prompt once
                            # Append targeted guidance to reinforce most commonly violated policies
                            enhanced_prompt = system_prompt + """

CRITICAL EXECUTION CHECKLIST — follow these rules strictly on EVERY turn:

### Tool & Parameter Integrity (HALLUCINATION PREVENTION)
- ONLY use tools explicitly listed in your tool definitions. If a required tool is MISSING, tell the user you cannot perform this action. NEVER fabricate or invent tool calls.
- ONLY use parameters explicitly listed in each tool's schema. If a parameter you need is MISSING from the schema, you CANNOT use that tool for this purpose — tell the user the specific capability is unavailable. Do NOT pass parameters that don't exist in the schema.
- ONLY use parameter values that match the tool's schema enum values exactly. Check the enum list carefully before every call. For set_seat_heating: seat_zone must be one of "ALL_ZONES", "DRIVER", or "PASSENGER" — there is NO "DRIVER_REAR" or "PASSENGER_REAR". For open_close_sunshade: percentage must be one of [0, 50, 100] — there is NO 40 or other value.
- If a required parameter value is unclear or ambiguous, ASK the user to clarify. Do NOT guess or assume.
- NEVER offer to do something you cannot actually do with available tools. If you cannot fulfill a request, say so clearly and stop. Do NOT say "I can do that" and then fail to call a tool.
- MISSING TOOL RESPONSES: When a tool result is missing expected fields (e.g., get_car_color returns no color, get_vehicle_window_positions is missing some window data, search results return empty), you MUST tell the user "I'm sorry, that information is currently unavailable" and STOP. Do NOT: (a) make up the missing data, (b) calculate it yourself, (c) use a different tool as workaround, (d) proceed as if the data exists, or (e) ask the user for info that should come from the system. The missing data means the system cannot provide it right now.
- MISSING TOOLS: If a tool you need is not in your tool list, say "I'm sorry, I cannot do that — the required capability is not available." Do NOT call a similar-sounding tool or try to work around it.

### Mandatory Pre-checks Before State Changes
- BEFORE turning AC ON (set_air_conditioning on=true): You MUST do ALL of the following in a PRIOR turn:
  1. Call get_vehicle_window_positions (or get_climate_settings) to check windows
  2. Close ANY window open more than 20% using open_close_window
  3. Call set_fan_speed to set fan speed to at least 1 (MANDATORY even if you already checked climate — the fan MUST be on)
  Then call set_air_conditioning in the NEXT turn after all prerequisites are done.
- BEFORE activating window defrost for FRONT/ALL (set_window_defrost on=true): You MUST ensure fan_speed >= 2, airflow direction includes WINDSHIELD, and AC is ON. Check via get_climate_settings or set them yourself BEFORE defrost.
- BEFORE enabling fog lights (set_fog_lights on=true): You MUST first call get_weather to check weather conditions AND call get_exterior_lights_status to check current light state. Low beams MUST be ON (activate if not). High beams MUST be OFF (deactivate if on). Do these in a PRIOR turn BEFORE fog lights.
- BEFORE enabling high beams: Fog lights MUST be OFF first.
- BEFORE opening sunroof: Check weather at current location first. Sunshade must be fully open (100%) or opened in parallel.
- When setting climate controls for multiple zones (driver+passenger), prefer "ALL_ZONES" as the zone parameter — do NOT set DRIVER and PASSENGER separately when both should get the same value.
- BEFORE adjusting seat heating, call get_seat_heating_level to check current levels. This ensures you know what to change.

### Navigation Rules
- TOLL ROADS — ZERO TOLERANCE: For EVERY route you present, select, or apply, you MUST explicitly say whether it has tolls. Check the includes_toll field. Say "this route includes toll roads" or "this is toll-free". This applies to EVERY route in EVERY segment. Missing a single toll mention = policy violation.
- ROUTE SELECTION — ZERO TOLERANCE: If you select a route without the user telling you which specific route to use, you MUST say BOTH: (a) "I have selected the [fastest/shortest] route" AND (b) "Would you like to hear about alternative routes?" You MUST include BOTH parts. This applies per-segment for multi-stop routes. Even when removing a waypoint or replacing a destination, if you pick a route, you must say this. Failing to say EITHER part = policy violation.
- Present the fastest and shortest routes with details (distance, duration, toll info). For other alternatives, mention only the count (e.g., "There are also 2 other routes available").
- NAVIGATION STATE: ALWAYS call get_current_navigation_state FIRST when the user asks about navigation or wants to modify it. This tells you if navigation is active, what the current route is, and what waypoints exist. If the user asks to "restart" or "resume" navigation, check the navigation state first — it may contain the previous route information.
- ACTIVE NAVIGATION: If navigation is already active, you MUST use navigation editing tools (navigation_add_one_waypoint, navigation_replace_one_waypoint, navigation_replace_final_destination, navigation_delete_one_waypoint, navigation_delete_final_destination). NEVER call set_new_navigation when navigation is active — it will fail with an error.
- Use navigation editing tools ONE AT A TIME in sequence, never in parallel.
- Route start must always be current location.

### Calendar and Time Rules
- ALL times must be in 24-hour format (e.g., 14:30, not 2:30 PM).
- Use metric system: kilometers, meters, Celsius.
- When looking up calendar events, use today's actual date. Do NOT ask the user what today's date is — you already know it from the system context.
- When the user says "today", "tomorrow", "this week", calculate the actual date yourself.

### Format & Communication
- Do NOT use markdown, lists, bold, or non-speakable characters — your output goes to text-to-speech.
- Keep responses concise and natural for voice interaction.

### Disambiguation — CRITICAL, READ CAREFULLY
When ANY parameter is ambiguous (you don't know the exact value to use), you MUST resolve it in this order:
1. Policy rules (e.g., safety requirements)
2. Explicit user request in current conversation
3. CALL get_user_preferences — this is NOT optional. You MUST call it BEFORE asking the user.
4. Context clues (e.g., only driver seat doesn't have the requested temp → set DRIVER zone)
5. Heuristic defaults
6. ONLY THEN ask the user

EXAMPLES OF WHEN TO CALL get_user_preferences (you MUST do this):
- User says "turn on the fan" but doesn't say which level → call get_user_preferences for climate_control
- User says "change air circulation mode" → call get_user_preferences for climate_control
- User says "turn on steering wheel heating" → call get_user_preferences for climate_control/vehicle_settings
- User says "set my temperature" → call get_user_preferences for climate_control
- User says "find a charging station" and multiple options exist → call get_user_preferences for charging_stations
- User says "send email" for business → call get_user_preferences for email (might need to CC secretary)

EXAMPLES OF WHEN TO USE CONTEXT instead of asking:
- User says "set temperature to 22" and only driver zone differs → set DRIVER zone (context resolves it)
- User says "turn on headlights" and low beams are already on → turn on high beams (context resolves it)
- User says "turn on the beams" and low beams are on → high beams are the only option

NEVER ask the user to pick from options (like "1, 2, or 3?" or "fresh air, recirculation, or auto?") without FIRST checking preferences AND context. If you ask without checking, the task WILL fail.

### Confirmation Rules
- If a tool description starts with REQUIRES_CONFIRMATION: you MUST tell the user what you are about to do and get explicit "yes" confirmation BEFORE calling that tool. For example, send_email requires confirmation — tell the user the recipient(s) and message content, then wait for "yes".
- For all OTHER tools (without REQUIRES_CONFIRMATION): do NOT ask for confirmation. If the user explicitly asks for an action, PROCEED IMMEDIATELY — do the prerequisite checks and execute the action. Do not ask "Is that what you want?" or "Shall I proceed?".
- Example: if the user says "Turn on the AC", just do the prerequisites and turn it on — no confirmation needed.
- Example: if the user says "Send an email to Frank", you must compose the email, present it to the user, and wait for confirmation before calling send_email.

### Scope Control
- Only perform actions the user explicitly requested. Do NOT proactively suggest or perform additional actions beyond what was asked.
- If you successfully complete the user's request, respond with confirmation and stop. Do not offer further services unprompted."""
                            messages.append({"role": "system", "content": enhanced_prompt})
                    else:
                        # Regular user message
                        user_message_text = text
                
                elif isinstance(part.root, DataPart):
                    # Extract tools or tool results from DataPart
                    data = part.root.data
                    if "tools" in data:
                        tools = data["tools"]
                        self.ctx_id_to_tools[context.context_id] = tools
                    elif "tool_results" in data:
                        # Structured tool results from the green agent
                        incoming_tool_results = data["tool_results"]
            
            # Fallback if no text part and no structured tool results found
            if not user_message_text and not incoming_tool_results:
                user_message_text = context.get_user_input()
            
            ctx_logger.info(
                "Received user message",
                context_id=context.context_id[:8],
                turn=len(messages) + 1,
                message_preview=(user_message_text[:100] if user_message_text else
                                 f"[{len(incoming_tool_results)} tool results]" if incoming_tool_results else "")
            )
            ctx_logger.debug(
                "Message details",
                context_id=context.context_id[:8],
                message=user_message_text,
                num_parts=len(inbound_message.parts),
                has_tools=bool(tools),
                num_tools=len(tools) if tools else 0,
                has_tool_results=bool(incoming_tool_results),
                num_tool_results=len(incoming_tool_results) if incoming_tool_results else 0
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse message parts: {e}, using fallback")
            user_message_text = context.get_user_input()

        # Check if previous message had tool calls - if so, format as tool results
        if messages and messages[-1].get("role") == "assistant" and messages[-1].get("tool_calls"):
            prev_tool_calls = messages[-1]["tool_calls"]

            if incoming_tool_results:
                # Structured tool results from green agent — match each result
                # to its corresponding tool_call_id by tool name
                tool_call_by_name = {}
                for tc in prev_tool_calls:
                    name = tc["function"]["name"]
                    # If multiple calls to the same tool, use a list
                    tool_call_by_name.setdefault(name, []).append(tc)

                tool_results = []
                for tr in incoming_tool_results:
                    tr_name = tr.get("tool_name", "")
                    matching_calls = tool_call_by_name.get(tr_name, [])
                    if matching_calls:
                        # Pop the first matching call to handle duplicate tool names
                        matched_tc = matching_calls.pop(0)
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": matched_tc["id"],
                            "content": tr.get("content", ""),
                        })
                    else:
                        # Fallback: no matching tool_call found, use first unmatched
                        ctx_logger.warning(
                            "No matching tool_call_id for tool result",
                            tool_name=tr_name,
                        )
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tr.get("tool_call_id", f"unknown_{tr_name}"),
                            "content": tr.get("content", ""),
                        })
            else:
                # Fallback: no structured tool results, use the text message
                # for all tool calls (legacy behavior)
                tool_results = []
                for tc in prev_tool_calls:
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": user_message_text or "",
                    })
            
            # Add all tool result messages
            messages.extend(tool_results)
            
            ctx_logger.debug(
                "Formatted tool results",
                num_tools=len(tool_results),
                tool_call_ids=[tr["tool_call_id"] for tr in tool_results]
            )
        else:
            # Regular user message
            messages.append({"role": "user", "content": user_message_text})

        # Call LLM with native tool calling and retry logic
        try:
            # Configure prompt caching (guard against empty lists)
            if tools:
                tools[-1]["function"]["cache_control"] = {"type": "ephemeral"}
            if messages:
                messages[0]["cache_control"] = {"type": "ephemeral"}

            completion_kwargs = {
                "model": self.model,
                "tools": tools if tools else None,
                "temperature": self.temperature,
            }

            # Configure reasoning effort / thinking
            if self.thinking:
                    if self.model == "claude-opus-4-6":
                        completion_kwargs["thinking"] = {
                            "type": "adaptive"
                        }
                    else:
                        if self.reasoning_effort in [
                            "none",
                            "disable",
                            "low",
                            "medium",
                            "high",
                        ]:
                            completion_kwargs["reasoning_effort"] = self.reasoning_effort
                        else:
                            try:
                                thinking_budget = int(self.reasoning_effort)
                            except ValueError:
                                raise ValueError(
                                    "reasoning_effort must be 'none', 'disable', 'low', 'medium', 'high', or an integer value"
                                )
                            completion_kwargs["thinking"] = {
                                "type": "enabled",
                                "budget_tokens": thinking_budget,
                            }
                        if self.interleaved_thinking:
                            completion_kwargs["extra_headers"] = {
                                    "anthropic-beta": "interleaved-thinking-2025-05-14"
                                }

            # Retry logic for rate limits
            max_retries = 3
            response = None
            for attempt in range(max_retries):
                try:
                    response = completion(
                        messages=messages,
                        **completion_kwargs
                    )
                    break  # Success, exit retry loop
                except Exception as retry_err:
                    error_str = str(retry_err).lower()
                    retryable = ("rate_limit" in error_str or "rate limit" in error_str or "429" in error_str
                                 or "disconnect" in error_str or "connection" in error_str or "eof" in error_str
                                 or "server" in error_str or "timeout" in error_str)
                    if retryable:
                        wait_time = (attempt + 1) * 3  # 3s, 6s, 9s
                        ctx_logger.warning(
                            f"Retryable error, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries}): {str(retry_err)[:100]}"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        raise  # Non-retryable error

            if response is None:
                raise Exception("Max retries exceeded due to rate limiting")
            
            # Get the message from LLM
            llm_message = response.choices[0].message
            assistant_content = llm_message.model_dump(exclude_unset=True)
            
            # Extract tool calls from assistant content
            tool_calls = assistant_content.get("tool_calls")
            
            ctx_logger.info(
                "LLM response received",
                has_tool_calls=bool(tool_calls),
                num_tool_calls=len(tool_calls) if tool_calls else 0,
                has_content=bool(assistant_content.get("content")),
                content_length=len(assistant_content.get("content") or ""),
                has_thinking=bool(assistant_content.get("thinking_blocks") or assistant_content.get("reasoning_content"))
            )
            ctx_logger.debug(
                "LLM response details",
                context_id=context.context_id[:8],
                content=assistant_content.get("content"),
                tool_calls=[{"name": tc["function"]["name"], "args": tc["function"]["arguments"]} for tc in tool_calls] if tool_calls else None,
                reasoning_content=assistant_content.get("reasoning_content")
            )

            # Validate tool calls against schema and policies, with re-prompting loop
            max_validation_retries = 2
            for validation_attempt in range(max_validation_retries + 1):
                validated_tool_calls = []
                hallucination_errors = []
                if assistant_content.get("tool_calls") and tools:
                    tool_schema_map = {t["function"]["name"]: t["function"].get("parameters", {}) for t in tools}
                    for tc in assistant_content["tool_calls"]:
                        tc_name = tc["function"]["name"]
                        # Fix dotted tool names (e.g. "navigation_tool.get_current_navigation_state" → "get_current_navigation_state")
                        if "." in tc_name:
                            tc_name = tc_name.split(".")[-1]
                            tc["function"]["name"] = tc_name
                        tc_args = json.loads(tc["function"]["arguments"])
                        if tc_name not in tool_schema_map:
                            hallucination_errors.append(f"Tool '{tc_name}' is not available.")
                            continue
                        schema = tool_schema_map[tc_name]
                        schema_props = schema.get("properties", {})
                        invalid_params = [p for p in tc_args if p not in schema_props]
                        if invalid_params:
                            hallucination_errors.append(f"Tool '{tc_name}' does not accept parameter(s): {', '.join(invalid_params)}.")
                            continue
                        # Validate enum values
                        for param_name, param_value in tc_args.items():
                            param_schema = schema_props.get(param_name, {})
                            if "enum" in param_schema and param_value not in param_schema["enum"]:
                                hallucination_errors.append(
                                    f"Tool '{tc_name}' parameter '{param_name}' must be one of {param_schema['enum']}, got '{param_value}'."
                                )
                                break
                        else:
                            validated_tool_calls.append(tc)
                elif assistant_content.get("tool_calls"):
                    validated_tool_calls = assistant_content["tool_calls"]

                if hallucination_errors:
                    error_text = "I'm sorry, I cannot do that. " + " ".join(hallucination_errors) + " This capability is currently not available."
                    assistant_content["content"] = error_text
                    assistant_content["tool_calls"] = None
                    validated_tool_calls = []
                    ctx_logger.warning("Blocked hallucinated tool calls", errors=hallucination_errors)
                    break  # Don't retry hallucination — just refuse

                # Check AUT-POL policy violations on validated tool calls
                if validated_tool_calls:
                    previous_tool_names = self.ctx_id_to_previous_tool_names.get(context.context_id, [])
                    current_tool_names = [tc["function"]["name"] for tc in validated_tool_calls]
                    policy_violations = check_policy_violations(validated_tool_calls, previous_tool_names, current_tool_names)

                    if policy_violations and validation_attempt < max_validation_retries:
                        # Re-prompt the LLM with violation feedback
                        violation_feedback = (
                            "POLICY VIOLATION DETECTED — your tool calls would violate car policies. "
                            "You MUST fix this before proceeding:\n" +
                            "\n".join(f"- {v}" for v in policy_violations) +
                            "\n\nPlease revise your response. Call the prerequisite tools first, "
                            "or adjust your approach to comply with all policies."
                        )
                        ctx_logger.warning(
                            f"Policy violation detected (attempt {validation_attempt + 1}), re-prompting",
                            violations=policy_violations
                        )
                        # Add the failed assistant message + violation feedback to messages
                        messages.append({
                            "role": "assistant",
                            "content": assistant_content.get("content"),
                            "tool_calls": assistant_content.get("tool_calls"),
                        })
                        messages.append({"role": "user", "content": violation_feedback})

                        # Re-call the LLM
                        try:
                            retry_response = completion(messages=messages, **completion_kwargs)
                            llm_message = retry_response.choices[0].message
                            assistant_content = llm_message.model_dump(exclude_unset=True)
                            ctx_logger.info(
                                f"Re-prompt response (attempt {validation_attempt + 2})",
                                has_tool_calls=bool(assistant_content.get("tool_calls")),
                                has_content=bool(assistant_content.get("content")),
                            )
                            # Remove the violation feedback from history (we don't want green agent to see it)
                            messages.pop()  # Remove user violation feedback
                            messages.pop()  # Remove failed assistant message
                            continue  # Re-validate
                        except Exception as e:
                            ctx_logger.error(f"Re-prompt failed: {e}")
                            messages.pop()
                            messages.pop()
                            break
                    elif policy_violations:
                        # Max retries exhausted — log but let it through (LLM may have fixed partially)
                        ctx_logger.warning("Policy violations remain after re-prompting", violations=policy_violations)

                break  # No violations or hallucinations, proceed

            # Update previous tool names tracking
            if validated_tool_calls:
                if context.context_id not in self.ctx_id_to_previous_tool_names:
                    self.ctx_id_to_previous_tool_names[context.context_id] = []
                self.ctx_id_to_previous_tool_names[context.context_id].extend(
                    tc["function"]["name"] for tc in validated_tool_calls
                )

            # Build proper A2A Message with Parts
            parts = []

            # Add TextPart if there's content
            if assistant_content.get("content"):
                parts.append(Part(root=TextPart(
                    kind="text",
                    text=assistant_content["content"]
                )))

            # Add DataPart if there are validated tool calls
            if validated_tool_calls and not hallucination_errors:
                tool_calls_list = [
                    ToolCall(
                        tool_name=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"]),
                    )
                    for tc in validated_tool_calls
                ]
                tool_calls_data = ToolCallsData(tool_calls=tool_calls_list)
                parts.append(Part(root=DataPart(
                    kind="data",
                    data=tool_calls_data.model_dump()
                )))
            
            # Add reasoning_content as DataPart for debugging (if present)
            if assistant_content.get("reasoning_content"):
                parts.append(Part(root=DataPart(
                    kind="data",
                    data={"reasoning_content": assistant_content["reasoning_content"]}
                )))
            
            # If no parts, add empty text
            if not parts:
                parts.append(Part(root=TextPart(
                    kind="text",
                    text=assistant_content.get("content") or ""
                )))
            
            ctx_logger.debug(
                "Sending response",
                context_id=context.context_id[:8],
                num_parts=len(parts),
                parts_summary=[{"kind": p.root.kind, "has_data": bool(p.root.text if hasattr(p.root, 'text') else p.root.data)} for p in parts]
            )
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            # Error response as Parts
            parts = [Part(root=TextPart(
                kind="text",
                text=f"Error processing request: {str(e)}"
            ))]
            # Create a simple assistant_content for error case
            assistant_content = {"content": f"Error processing request: {str(e)}"}

        # Add to history - preserve complete assistant message including thinking blocks
        # Store the full assistant_content to preserve thinking blocks and reasoning_content
        assistant_message_for_history = {
            "role": "assistant",
            "content": assistant_content.get("content"),
        }
        
        # Preserve tool calls in proper format for LLM API
        if assistant_content.get("tool_calls"):
            assistant_message_for_history["tool_calls"] = assistant_content["tool_calls"]
        
        # Preserve thinking blocks and reasoning content for Claude extended thinking
        if assistant_content.get("thinking_blocks"):
            assistant_message_for_history["thinking_blocks"] = assistant_content["thinking_blocks"]
        if assistant_content.get("reasoning_content"):
            assistant_message_for_history["reasoning_content"] = assistant_content["reasoning_content"]
        
        messages.append(assistant_message_for_history)

        # Send response via A2A - use new_agent_parts_message
        response_message = new_agent_parts_message(
            parts=parts,
            context_id=context.context_id
        )
        await event_queue.enqueue_event(response_message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the current execution."""
        logger.bind(role="agent", context=f"ctx:{context.context_id[:8]}").info(
            "Canceling context",
            context_id=context.context_id[:8]
        )
        if context.context_id in self.ctx_id_to_messages:
            del self.ctx_id_to_messages[context.context_id]
        if context.context_id in self.ctx_id_to_tools:
            del self.ctx_id_to_tools[context.context_id]
        if context.context_id in self.ctx_id_to_previous_tool_names:
            del self.ctx_id_to_previous_tool_names[context.context_id]
