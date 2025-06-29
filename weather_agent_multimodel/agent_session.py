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

async def create_session(
        MODEL_SHORT_NAME="gpt",  # ['gpt', 'claude', 'gemini']
        APP_NAME_ROOT="weather_app",
        USER_ID_ROOT="user_1",
        SESSION_ID_ROOT="session_001" # Using a fixed ID for simplicity
    ):
    
    APP_NAME = f"{APP_NAME_ROOT}_{MODEL_SHORT_NAME}"
    USER_ID = f"{USER_ID_ROOT}_{MODEL_SHORT_NAME}"
    SESSION_ID = f"{SESSION_ID_ROOT}_{MODEL_SHORT_NAME}"
    
    # Create the agent
    weather_agent = None # Initialize to None
    runner = None      # Initialize runner to None

    if MODEL_SHORT_NAME == "gpt":
        MODEL_GPT = os.getenv("MODEL_GPT")
        if not MODEL_GPT:
            raise ValueError("MODEL_GPT is not defined in .env")
        AGENT_MODEL = MODEL_GPT
        model = LiteLlm(model=AGENT_MODEL)
    elif MODEL_SHORT_NAME == "claude":
        MODEL_CLAUDE = os.getenv("MODEL_CLAUDE")
        if not MODEL_CLAUDE:
            raise ValueError("MODEL_CLAUDE is not defined in .env")
        AGENT_MODEL = MODEL_CLAUDE
        model = LiteLlm(model=AGENT_MODEL)
    elif MODEL_SHORT_NAME == "gemini":
        MODEL_GEMINI = os.getenv("MODEL_GEMINI")
        if not MODEL_GEMINI:
            raise ValueError("MODEL_GEMINI is not defined in .env")
        AGENT_MODEL = MODEL_GEMINI
        model = AGENT_MODEL  #If Gemini, use the model directly

    try:
        weather_agent_gpt = Agent(
            name=f"weather_agent_{MODEL_SHORT_NAME}",
            model=model, # Can be a string for Gemini or a LiteLlm object
            description="Provides weather information for specific cities.",
            instruction="You are a helpful weather assistant. "
                        "When the user asks for the weather in a specific city, "
                        "use the 'get_weather' tool to find the information. "
                        "If the tool returns an error, inform the user politely. "
                        "If the tool is successful, present the weather report clearly.",
            tools=[get_weather], # Pass the function directly
        )
        print(f"Agent '{weather_agent_gpt.name}' created using model '{AGENT_MODEL}'.")
    except Exception as e:
        print(f"Failed to create GPT agent using model '{AGENT_MODEL}'. Error: {e}")

    # --- Session Management ---
    # Key Concept: SessionService stores conversation history & state.
    # InMemorySessionService is simple, non-persistent storage for this tutorial.
    session_service = InMemorySessionService()

    # Define constants for identifying the interaction context

    # Create the specific session where the conversation will happen
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    print(f"Session created: App='{APP_NAME}', User='{USER_ID}', Session='{SESSION_ID}'")

    # --- Runner ---
    # Key Concept: Runner orchestrates the agent execution loop.
    runner = Runner(
        agent=weather_agent_gpt, # The agent we want to run
        app_name=APP_NAME,   # Associates runs with our app
        session_service=session_service # Uses our session manager
    )
    print(f"Runner created for agent '{runner.agent.name}'.")
    return session, runner


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

async def run_conversation(
        runner,
        USER_ID,
        SESSION_ID
    ):
    await call_agent_async("What is the weather like in London?",
                                       runner=runner,
                                       user_id=USER_ID,
                                       session_id=SESSION_ID)

    await call_agent_async("How about Paris?",
                                       runner=runner,
                                       user_id=USER_ID,
                                       session_id=SESSION_ID) # Expecting the tool's error message

    await call_agent_async("Tell me the weather in New York",
                                       runner=runner,
                                       user_id=USER_ID,
                                       session_id=SESSION_ID)


if __name__ == "__main__":
    MODEL_SHORT_NAME = "gpt"  # ['gpt', 'claude', 'gemini']
    USER_ID_ROOT = "user_1"
    SESSION_ID_ROOT = "session_001"
    session, runner = asyncio.run(
        create_session(
            MODEL_SHORT_NAME=MODEL_SHORT_NAME,
            APP_NAME_ROOT="weather_app",
            USER_ID_ROOT=USER_ID_ROOT,
            SESSION_ID_ROOT=SESSION_ID_ROOT 
        )
    )
    try:
        asyncio.run(
            run_conversation(
                runner,
                USER_ID=f"{USER_ID_ROOT}_{MODEL_SHORT_NAME}",
                SESSION_ID=f"{SESSION_ID_ROOT}_{MODEL_SHORT_NAME}"
            )
        )
        # close the session?
        # session.close()
    except Exception as e:
        print(f"An error occurred: {e}")