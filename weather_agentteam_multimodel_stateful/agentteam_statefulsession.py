# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Import necessary libraries
import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any

import asyncio
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.base_tool import BaseTool
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.genai import types # For creating message Content/Parts

import logging
logging.basicConfig(level=logging.ERROR)

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

# Use one of the model constants defined in .env
load_dotenv(os.path.join('weather_agent_multimodel', '.env'))  # take environment variables


# @title Define the get_weather Tool
def get_weather(city: str, tool_context: ToolContext) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city (e.g., "New York", "London", "Tokyo").

    Returns:
        dict: A dictionary containing the weather information.
              Includes a 'status' key ('success' or 'error').
              If 'success', includes a 'report' key with weather details.
              If 'error', includes an 'error_message' key.
    """
    print(f"--- Tool: get_weather called for city: {city} ---") # Log tool execution
    
    # --- Read preference from state ---
    preferred_unit = tool_context.state.get("user_preference_temperature_unit", "Celsius") # Default to Celsius
    print(f"--- Tool: Reading state 'user_preference_temperature_unit': {preferred_unit} ---")
    
    city_normalized = city.lower().replace(" ", "") # Basic normalization

    # Mock weather data (always stored in Celsius internally)
    mock_weather_db = {
        "newyork": {"temp_c": 25, "condition": "sunny"},
        "london": {"temp_c": 15, "condition": "cloudy"},
        "tokyo": {"temp_c": 18, "condition": "light rain"},
    }

    if city_normalized in mock_weather_db:
        data = mock_weather_db[city_normalized]
        temp_c = data["temp_c"]
        condition = data["condition"]

        # Format temperature based on state preference
        if preferred_unit == "Fahrenheit":
            temp_value = (temp_c * 9/5) + 32 # Calculate Fahrenheit
            temp_unit = "°F"
        else: # Default to Celsius
            temp_value = temp_c
            temp_unit = "°C"

        report = f"The weather in {city.capitalize()} is {condition} with a temperature of {temp_value:.0f}{temp_unit}."
        result = {"status": "success", "report": report}
        print(f"--- Tool: Generated report in {preferred_unit}. Result: {result} ---")

        # Example of writing back to state (optional for this tool)
        tool_context.state["last_city_checked_stateful"] = city
        print(f"--- Tool: Updated state 'last_city_checked_stateful': {city} ---")

        return result
    else:
        # Handle city not found
        error_msg = f"Sorry, I don't have weather information for '{city}'."
        print(f"--- Tool: City '{city}' not found. ---")
        return {"status": "error", "error_message": error_msg}

    print("✅ State-aware 'get_weather_stateful' tool defined.")

def say_hello(name: Optional[str] = None) -> str:
    """Provides a simple greeting. If a name is provided, it will be used.

    Args:
        name (str, optional): The name of the person to greet. Defaults to a generic greeting if not provided.

    Returns:
        str: A friendly greeting message.
    """
    if name:
        greeting = f"Hello, {name}!"
        print(f"--- Tool: say_hello called with name: {name} ---")
    else:
        greeting = "Hello there!" # Default greeting if name is None or not explicitly passed
        print(f"--- Tool: say_hello called without a specific name (name_arg_value: {name}) ---")
    return greeting

def say_goodbye() -> str:
    """Provides a simple farewell message to conclude the conversation."""
    print(f"--- Tool: say_goodbye called ---")
    return "Goodbye! Have a great day."

def create_greeting_agent(
        agent_name,
        model,
        agent_model,
    ):
    greeting_agent = None
    try:
        greeting_agent = Agent(
            name=agent_name,
            model=model,
            instruction="You are the Greeting Agent. Your ONLY task is to provide a friendly greeting to the user. "
                        "Use the 'say_hello' tool to generate the greeting. "
                        "If the user provides their name, make sure to pass it to the tool. "
                        "Do not engage in any other conversation or tasks.",
            description="Handles simple greetings and hellos using the 'say_hello' tool.", # Crucial for delegation
            tools=[say_hello],
        )
        print(f"Agent '{greeting_agent.name}' created using model '{agent_model}'.")
    except Exception as e:
        print(f"Failed to create greeting agent using model '{agent_model}'. Error: {e}")
    return greeting_agent

# --- Farewell Agent ---
def create_farewell_agent(
        agent_name,
        model,
        agent_model,
    ):
    farewell_agent = None
    try:
        farewell_agent = Agent(
            name=agent_name,
            model = model,
            instruction="You are the Farewell Agent. Your ONLY task is to provide a polite goodbye message. "
                        "Use the 'say_goodbye' tool when the user indicates they are leaving or ending the conversation "
                        "(e.g., using words like 'bye', 'goodbye', 'thanks bye', 'see you'). "
                        "Do not perform any other actions.",
            description="Handles simple farewells and goodbyes using the 'say_goodbye' tool.", # Crucial for delegation
            tools=[say_goodbye],
        )
        print(f"Agent '{farewell_agent.name}' created using model '{agent_model}'.")
    except Exception as e:
        print(f"Failed to create farewell agent using model '{agent_model}'. Error: {e}")
    return farewell_agent

def create_weather_agent_team(
        agent_name,
        model,
        agent_model,
        sub_agents,
        output_key
    ):
    weather_agent_team = None # Initialize to None
    try:
        weather_agent_team = Agent(
            name=agent_name,
            model=model, # Can be a string for Gemini or a LiteLlm object
            description="The main coordinator agent. Handles weather requests and delegates greetings/farewells to specialists.",
            instruction="You are the main Weather Agent coordinating a team. Your primary responsibility is to provide weather information. "
                        "Use the 'get_weather' tool ONLY for specific weather requests (e.g., 'weather in London'). "
                        "You have specialized sub-agents: "
                        "1. 'greeting_agent': Handles simple greetings like 'Hi', 'Hello'. Delegate to it for these. "
                        "2. 'farewell_agent': Handles simple farewells like 'Bye', 'See you'. Delegate to it for these. "
                        "Analyze the user's query. If it's a greeting, delegate to 'greeting_agent'. If it's a farewell, delegate to 'farewell_agent'. "
                        "If it's a weather request, handle it yourself using 'get_weather'. "
                        "For anything else, respond appropriately or state you cannot handle it.",
            tools=[get_weather], # Pass the function directly
            sub_agents=sub_agents,
            output_key=output_key, # <<< Auto-save agent's final weather response
            before_model_callback=block_keyword_guardrail,
            before_tool_callback=block_city_tool_guardrail
        )
        print(f"Root Agent '{weather_agent_team.name}' created using model '{agent_model}' with sub-agents: {[sa.name for sa in weather_agent_team.sub_agents]}.")
    except Exception as e:
        print(f"Failed to create weather agent team using model '{agent_model}'. Error: {e}")
    return weather_agent_team

def block_keyword_guardrail(
        callback_context: CallbackContext,
        llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
    """
    Inspects the latest user message for 'BLOCK'. If found, blocks the LLM call
    and returns a predefined LlmResponse. Otherwise, returns None to proceed.
    """
    agent_name = callback_context.agent_name # Get the name of the agent whose model call is being intercepted
    print(f"--- Callback: block_keyword_guardrail running for agent: {agent_name} ---")

    # Extract the text from the latest user message in the request history
    last_user_message_text = ""
    if llm_request.contents:
        # Find the most recent message with role 'user'
        for content in reversed(llm_request.contents):
            if content.role == 'user' and content.parts:
                # Assuming text is in the first part for simplicity
                if content.parts[0].text:
                    last_user_message_text = content.parts[0].text
                    break # Found the last user message text

    print(f"--- Callback: Inspecting last user message: '{last_user_message_text[:100]}...' ---") # Log first 100 chars

    # --- Guardrail Logic ---
    keyword_to_block = "BLOCK"
    if keyword_to_block in last_user_message_text.upper(): # Case-insensitive check
        print(f"--- Callback: Found '{keyword_to_block}'. Blocking LLM call! ---")
        # Optionally, set a flag in state to record the block event
        callback_context.state["guardrail_block_keyword_triggered"] = True
        print(f"--- Callback: Set state 'guardrail_block_keyword_triggered': True ---")

        # Construct and return an LlmResponse to stop the flow and send this back instead
        return LlmResponse(
            content=types.Content(
                role="model", # Mimic a response from the agent's perspective
                parts=[types.Part(text=f"I cannot process this request because it contains the blocked keyword '{keyword_to_block}'.")],
            )
            # Note: You could also set an error_message field here if needed
        )
    else:
        # Keyword not found, allow the request to proceed to the LLM
        print(f"--- Callback: Keyword not found. Allowing LLM call for {agent_name}. ---")
        return None # Returning None signals ADK to continue normally

    print("✅ block_keyword_guardrail function defined.")

def block_city_tool_guardrail(
        tool: BaseTool,
        args: Dict[str, Any],
        tool_context: ToolContext
    ) -> Optional[Dict]:
    """
    Checks if 'get_weather_stateful' is called for defined cities.
    If so, blocks the tool execution and returns a specific error dictionary.
    Otherwise, allows the tool call to proceed by returning None.
    """
    tool_name = tool.name
    agent_name = tool_context.agent_name # Agent attempting the tool call
    print(f"--- Callback: block_city_tool_guardrail running for tool '{tool_name}' in agent '{agent_name}' ---")
    print(f"--- Callback: Inspecting args: {args} ---")

    # --- Guardrail Logic ---
    target_tool_name = "get_weather" # Match the function name used by FunctionTool
    blocked_cities = ["paris"]

    # Check if it's the correct tool and the city argument matches the blocked city
    if tool_name == target_tool_name:
        city_argument = args.get("city", "") # Safely get the 'city' argument
        if city_argument and city_argument.lower() in blocked_cities:
            print(f"--- Callback: Detected blocked city '{city_argument}'. Blocking tool execution! ---")
            # Optionally update state
            tool_context.state["guardrail_tool_block_triggered"] = True
            print(f"--- Callback: Set state 'guardrail_tool_block_triggered': True ---")

            # Return a dictionary matching the tool's expected output format for errors
            # This dictionary becomes the tool's result, skipping the actual tool run.
            return {
                "status": "error",
                "error_message": f"Policy restriction: Weather checks for '{city_argument.capitalize()}' are currently disabled by a tool guardrail."
            }
        else:
             print(f"--- Callback: City '{city_argument}' is allowed for tool '{tool_name}'. ---")
    else:
        print(f"--- Callback: Tool '{tool_name}' is not the target tool. Allowing. ---")


    # If the checks above didn't return a dictionary, allow the tool to execute
    print(f"--- Callback: Allowing tool '{tool_name}' to proceed. ---")
    return None # Returning None allows the actual tool function to run

    print("✅ block_city_tool_guardrail function defined.")

def get_model_constants(
        model_short_name,
        model_gpt=None,
        model_claude=None,
        model_gemini=None):
    if model_short_name == "gpt":
        if not model_gpt:
            model_gpt = os.getenv("MODEL_GPT")
            if not model_gpt:
                raise ValueError("MODEL_GPT is not defined in .env")
        agent_model = model_gpt
        model = LiteLlm(model=agent_model)
    elif model_short_name == "claude":
        if not model_claude:
            model_claude = os.getenv("MODEL_CLAUDE")
            if not model_claude:
                raise ValueError("MODEL_CLAUDE is not defined in .env")
        agent_model = model_claude
        model = LiteLlm(model=agent_model)
    elif model_short_name == "gemini":
        if not model_gemini:
            model_gemini = os.getenv("MODEL_GEMINI")
            if not model_gemini:
                raise ValueError("MODEL_GEMINI is not defined in .env")
        agent_model = model_gemini
        model = agent_model  #If Gemini, use the model directly

    return agent_model, model

def create_agent_team(
        model_short_name,
        output_key):
    
    # Using a potentially different/cheaper model for simple tasks
    agent_model, model = get_model_constants('gpt', 'openai/gpt-4.1-mini')
    greeting_agent = create_greeting_agent(
        agent_name=f"greeting_agent_gptmini",
        model=model,
        agent_model=agent_model
    )
    agent_model, model = get_model_constants('gpt', 'openai/gpt-4.1-mini')
    farewell_agent = create_farewell_agent(
        agent_name=f"farewell_agent_gptmini",
        model=model,
        agent_model=agent_model
    )

    # Create the weather agent team as the root agent
    # Make sure to use a capable model for the root agent to handle orchestration
    agent_model, model = get_model_constants(model_short_name)
    agent_team = create_weather_agent_team(
        agent_name=f"weather_agent_{model_short_name}",
        model=model,
        agent_model=agent_model,
        sub_agents=[greeting_agent, farewell_agent],
        output_key=output_key
    )

    return agent_team

async def call_agent_async(query: str, runner, user_id, session_id):
    """Sends a query to the agent and prints the final response."""
    print(f"\n>>> User Query: {query}")

    # Prepare the user's message in ADK format
    content = types.Content(role='user', parts=[types.Part(text=query)])

    final_response_text = "Agent did not produce a final response." # Default

    # Key Concept: run_async executes the agent logic and yields Events.
    # We iterate through events to find the final answer.
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        # You can uncomment the line below to see *all* events during execution
        # print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

        # Key Concept: is_final_response() marks the concluding message for the turn.
        if event.is_final_response():
            if event.content and event.content.parts:
                # Assuming text response in the first part
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate: # Handle potential errors/escalations
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            # Add more checks here if needed (e.g., specific error codes)
            break # Stop processing events once the final response is found

    print(f"<<< Agent Response: {final_response_text}")

async def run_stateful_team_conversation():
        model_short_name="gemini"  # ['gpt', 'claude', 'gemini']
        agent_team = create_agent_team(
            model_short_name,
            output_key="last_weather_report"
        )
        
        app_name_root="weather_app"
        user_id_root="user_1"
        session_id_root="session_001" # Using a fixed ID for simplicity
        app_name = f"{app_name_root}_{model_short_name}"
        user_id = f"{user_id_root}_{model_short_name}"
        session_id = f"{session_id_root}_{model_short_name}"

        # --- Session Management ---
        # Key Concept: SessionService stores conversation history & state.
        # InMemorySessionService is simple, non-persistent storage for this tutorial.
        session_service = InMemorySessionService()

        # Define initial state data - user prefers Celsius initially
        initial_state = {
            "user_preference_temperature_unit": "Celsius"
        }

        # Create the specific session where the conversation will happen
        session_stateful = await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            state=initial_state
        )
        print(f"Session created: App='{app_name}', User='{user_id}', Session='{session_id}'")
        
        # Verify the initial state was set correctly
        retrieved_session = await session_service.get_session(
            app_name=app_name,
            user_id=user_id,
            session_id = session_id)

        print("\n--- Initial Session State ---")
        if retrieved_session:
            print(retrieved_session.state)
        else:
            print("Error: Could not retrieve session.")

        # --- Runner ---
        # Key Concept: Runner orchestrates the agent execution loop.
        runner_stateful = Runner(
            agent=agent_team, # The agent team
            app_name=app_name,   # Associates runs with our app
            session_service=session_service # Uses our session manager
        )
        print(f"Runner created for agent '{runner_stateful.agent.name}'.")

        # Use the runner for the agent with the callback and the existing stateful session ID
        # Define a helper lambda for cleaner interaction calls
        interaction_func = lambda query: call_agent_async(query,
                                                         runner_stateful,
                                                         user_id, # Use existing user ID
                                                         session_id # Use existing session ID
                                                        )
        
        print("\n--- Testing State: Temp Unit Conversion & output_key ---")

        # 1. Check weather (Uses initial state: Celsius)
        print("--- Turn 1: Requesting weather in London (expect Celsius) ---")
        await interaction_func(query="What's the weather in London?")

        # 2. Manually update state preference to Fahrenheit - DIRECTLY MODIFY STORAGE
        print("\n--- Manually Updating State: Setting unit to Fahrenheit ---")
        try:
            # Access the internal storage directly - THIS IS SPECIFIC TO InMemorySessionService for testing
            # NOTE: In production with persistent services (Database, VertexAI), you would
            # typically update state via agent actions or specific service APIs if available,
            # not by direct manipulation of internal storage.
            stored_session = session_service.sessions[app_name][user_id][session_id]
            stored_session.state["user_preference_temperature_unit"] = "Fahrenheit"
            # Optional: You might want to update the timestamp as well if any logic depends on it
            # import time
            # stored_session.last_update_time = time.time()
            print(f"--- Stored session state updated. Current 'user_preference_temperature_unit': {stored_session.state.get('user_preference_temperature_unit', 'Not Set')} ---") # Added .get for safety
        except KeyError:
            print(f"--- Error: Could not retrieve session '{session_id}' from internal storage for user '{user_id}' in app '{app_name}' to update state. Check IDs and if session was created. ---")
        except Exception as e:
             print(f"--- Error updating internal session state: {e} ---")

        # # 3 Request containing the blocked keyword (Callback intercepts)
        # print("\n--- Turn 3: Requesting with blocked keyword (expect blocked by input guardrail) ---")
        # await interaction_func(query="BLOCK the request for weather in Tokyo") # Callback should catch "BLOCK"

        # 3. Blocked city (Should pass model callback, but be blocked by tool callback)
        print("\n--- Turn 3: Requesting weather in Paris (expect blocked by tool guardrail) ---")
        await interaction_func("How about Paris?") # Tool callback should intercept this

        # 4. Check weather again (Tool should now use Fahrenheit)
        # This will also update 'last_weather_report' via output_key
        print("\n--- Turn 4: Requesting weather in New York (expect Fahrenheit) ---")
        await interaction_func(query="Tell me the weather in New York.")

        # 5. Test basic delegation (should still work)
        # This will update 'last_weather_report' again, overwriting the NY weather report
        print("\n--- Turn 5: Sending a farewell ---")
        await interaction_func(query="Thanks, bye!")

        print("\n--- Inspecting Final Session State ---")
        final_session = await session_service.get_session(
            app_name=app_name,
            user_id= user_id,
            session_id=session_id)
        if final_session:
            # Use .get() for safer access to potentially missing keys
            print(f"Final Preference: {final_session.state.get('user_preference_temperature_unit', 'Not Set')}")
            print(f"Final Last Weather Report (from output_key): {final_session.state.get('last_weather_report', 'Not Set')}")
            print(f"Final Last City Checked (by tool): {final_session.state.get('last_city_checked_stateful', 'Not Set')}")
            # Print full state for detailed view
            # print(f"Full State Dict: {final_session.state}") # For detailed view
        else:
            print("\n❌ Error: Could not retrieve final session state.")

if __name__ == "__main__":
    try:
        asyncio.run(
            run_stateful_team_conversation()
        )

        # close the session?
        # session.close()
    except Exception as e:
        print(f"An error occurred: {e}")