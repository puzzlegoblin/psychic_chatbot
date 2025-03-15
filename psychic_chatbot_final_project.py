import streamlit as st

pip install --upgrade "crewai[tools]" databricks-sdk
pip install "chromadb<=0.5.23" "litellm<=1.60.2" "tokenizers<=0.20.3" "httpx<0.28.0"

import pkgutil
import crewai_tools

print([mod.name for mod in pkgutil.iter_modules(crewai_tools.__path__)])


import os
from google.colab import userdata
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = userdata.get('open_ai_key')
os.environ["SERPER_API_KEY"] = userdata.get('SERPER_API_KEY')

# CrewAI Agent
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool
from IPython.display import Markdown, display

# User-provided inputs
st.title("Streamlit User Input")
date_of_birth = st.text_input("When were you born? (enter your birthday in this format: 15 May 1990)")
time_of_birth = st.text_input("What time were you born? (enter your time of birth in this format: 14:30")
city_of_birth = st.text_input("Where where you born? (enter the closest city to your birth place: such as Seattle or New York")
research_question = st.text_input("How question do you have for the spirits?")
desired_answer_length = st.text_input("How much information would you like from our guides? Select: short, medium, or long")

# Construct research topic
topic = (f"Person born on {date_of_birth} at {time_of_birth} in {city_of_birth}. "
         f"{research_question}")
print(f"DEBUG: Query being used for search: {topic}")

# Define Agents
researcher = Agent(
    name="Researcher",
    role="Astrology Researcher",
    goal="Gather reliable information on a topic and provide advice",
    backstory="An experienced astrologist and psychologist skilled in extracting key insights from online sources",
    tools=[WebsiteSearchTool(), SerperDevTool()],
    verbose=True
)

writer = Agent(
    name="Writer",
    role="Writer",
    goal="Accurately summarize detailed research about psychology and astrology to give personalized advice",
    backstory="An experienced psychic skilled in turning astrology and psychology research into thorough advice",
    tools=[WebsiteSearchTool(), SerperDevTool()],
    verbose=True
)


psychic = Agent(
    name="Psychic",
    role='Mystic Advisor',
    goal="Provide highly-specific observations and personalized advice with eccentric flair",
    temperature=1.3,  # High temperature for creativity
    backstory="You are a mystical fortune teller who gives wildly creative, dramatic, and humorous astrology readings. Be as theatrical and ridiculous as possible!",
    llm='gpt-3.5-turbo',
    tools=[WebsiteSearchTool(), SerperDevTool()],
    verbose=True
) 

# Define Tasks
research_task = Task(
    description=f"Use astrology-related sources to research: {topic}. Focus only on astrological insights and ignore unrelated topics.",
    expected_output=f"Summarise astrological predictions relevant to: {topic}. Ignore any unrelated content.",
    agent=researcher
)

length_mapping = {
    "short": "One or two sentences.",
    "medium": "A short paragraph.",
    "long": "A detailed response of multiple paragraphs."
}

write_task = Task(
    description=f"Write advice based on the research findings for: {topic}.",
    expected_output=f"{length_mapping.get(desired_answer_length, 'Outline detailed notes.')}",
    agent=writer,
    context=[research_task]
)

summary_task = Task(
    description=f"Write advice in a fortune telling style based on the research findings for: {topic}.",
    expected_output=f"{length_mapping.get(desired_answer_length, 'Answer the users question in a whimsical fortune')}",
    agent=psychic,
    context=[research_task]
)

# Assemble Crew
crew = Crew(
    agents=[researcher, psychic],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=True
)

# Execute Research
result = crew.kickoff()
print(f"DEBUG: Query being used for search: {topic}")
