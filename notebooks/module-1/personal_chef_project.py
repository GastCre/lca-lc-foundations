# %% Loading environment variables
import base64
from IPython.display import display
from ipywidgets import FileUpload
from langgraph.checkpoint.memory import InMemorySaver
from tavily import TavilyClient
from typing import Dict, Any
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()
# %% First we create the agent tool
tavily_client = TavilyClient()


@tool('search_recipe', description='Search for recipes based on the ingredients provided by the user.')
def search_recipe(ingredients: str) -> Dict[str, Any]:
    """Search for recipes based on the ingredients provided by the user."""
    response = tavily_client.search(ingredients)
    return response


# %% Now we create the agent with memory and system prompt
system_prompt = """You are TuginaCucina, a personal chef. You create recipes using web search, based on the ingredients provided
by the user either via text, image or audio. First, introduce yourself, then, if needed, ask question to the user in order to clarify.

The output should be a recipe that includes the name of the dish, the ingredients and the instructions to prepare it, as well as
cooking time, macros and any other useful information."""

agent = create_agent(model="gpt-5-nano",
                     system_prompt=system_prompt,
                     tools=[search_recipe],
                     checkpointer=InMemorySaver()
                     )


# %% Run the agent with a user message
message = HumanMessage(
    content='I have tofu, fake vegeterian chicken and chickpeas, what can I cook?')
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke({'messages': [message]},
                        config=config)
# %%
print(response['messages'][-1].content, end='\n\n')

# %% Now we add multimodality (just images for now).

# Image uploader widget

uploader = FileUpload(accept='.png', multiple=False)
display(uploader)

# Base64 encoding of the image

# Get the first (and only) uploaded file dict
uploaded_file = uploader.value[0]

# This is a memoryview
content_mv = uploaded_file["content"]

# Convert memoryview -> bytes
img_bytes = bytes(content_mv)  # or content_mv.tobytes()

# Now base64 encode
img_b64 = base64.b64encode(img_bytes).decode("utf-8")

# Multimodal message with text and image
multimodal_message = HumanMessage(content={'type': 'message', 'text': message.content},
                                  {'type': 'image', 'base64': img_b64,
                                      'mime_type': 'image/png'}
                                  )
response = agent.invoke({'messages': [multimodal_message]},
                        config=config)
print(response['messages'][-1].content, end='\n\n')
# %%
