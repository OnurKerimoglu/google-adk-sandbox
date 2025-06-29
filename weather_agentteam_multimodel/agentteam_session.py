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
from typing import Optional

import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types # For creating message Content/Parts

import logging
logging.basicConfig(level=logging.ERROR)

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

# Use one of the model constants defined in .env
load_dotenv(os.path.join('weather_agent_multimodel', '.env'))  # take environment variables


# @title Define the get_weather Tool
def get_weather(city: str) -> dict:
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
    city_normalized = city.lower().replace(" ", "") # Basic normalization

    # Mock weather data
    mock_weather_db = {
        "newyork": {"status": "success", "report": "The weather in New York is sunny with a temperature of 25°C."},
        "london": {"status": "success", "report": "It's cloudy in London with a temperature of 15°C."},
        "tokyo": {"status": "success", "report": "Tokyo is experiencing light rain and a temperature of 18°C."},
    }

    if city_normalized in mock_weather_db:
        return mock_weather_db[city_normalized]
    else:
        return {"status": "error", "error_message": f"Sorry, I don't have weather information for '{city}'."}

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

def create_weather_agent_team(
        agent_name,
        model,
        agent_model,
        sub_agents
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
            sub_agents=sub_agents
        )
        print(f"Root Agent '{weather_agent_team.name}' created using model '{agent_model}' with sub-agents: {[sa.name for sa in weather_agent_team.sub_agents]}.")
    except Exception as e:
        print(f"Failed to create weather agent team using model '{agent_model}'. Error: {e}")
    return weather_agent_team

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

def create_agent_team(model_short_name):
    
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
        sub_agents=[greeting_agent, farewell_agent]
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

async def run_team_conversation():
        model_short_name="gemini"  # ['gpt', 'claude', 'gemini']
        agent_team = create_agent_team(model_short_name)
        
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

        # Define constants for identifying the interaction context

        # Create the specific session where the conversation will happen
        session = await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id
        )
        print(f"Session created: App='{app_name}', User='{user_id}', Session='{session_id}'")

        # --- Runner ---
        # Key Concept: Runner orchestrates the agent execution loop.
        runner_agent_team = Runner(
            agent=agent_team, # The agent team
            app_name=app_name,   # Associates runs with our app
            session_service=session_service # Uses our session manager
        )
        print(f"Runner created for agent '{runner_agent_team.agent.name}'.")


        # --- Interactions using await (correct within async def) ---
        await call_agent_async(query = "Hello there!",
                               runner=runner_agent_team,
                               user_id=user_id,
                               session_id=session_id)
        await call_agent_async(query = "What is the weather in New York?",
                               runner=runner_agent_team,
                               user_id=user_id,
                               session_id=session_id)
        await call_agent_async(query = "Thanks, bye!",
                               runner=runner_agent_team,
                               user_id=user_id,
                               session_id=session_id)

if __name__ == "__main__":
    try:
        asyncio.run(
            run_team_conversation()
        )
        # close the session?
        # session.close()
    except Exception as e:
        print(f"An error occurred: {e}")