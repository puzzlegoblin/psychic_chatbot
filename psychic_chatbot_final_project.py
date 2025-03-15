# -*- coding: utf-8 -*-
"""psychic chatbot.final project

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1g-41bM5sSizV6e-CvJAmBUJCkught_my
"""

!pip install --upgrade "crewai[tools]" databricks-sdk
!pip install "chromadb<=0.5.23" "litellm<=1.60.2" "tokenizers<=0.20.3" "httpx<0.28.0"
!pip install streamlit

import os
from google.colab import userdata

import streamlit as st

os.environ["OPENAI_API_KEY"] = userdata.get('open_ai_key')
os.environ["SERPER_API_KEY"] = userdata.get('SERPER_API_KEY')

# CrewAI Agent
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool
from IPython.display import Markdown, display

# User-provided inputs
date_of_birth = "15 May 1990"
time_of_birth = "14:30"
city_of_birth = "New York"
research_question = "What career advice would best suit this person based on their astrological chart?"
desired_answer_length = "medium"  # Options: short, medium, long

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

psychic = Agent(
    name="Psychic",
    role="Mystical Advisor",
    goal="Provide highly-specific observations and personalized advice with eccentric flair",
    backstory="An experienced psychic skilled in turning astrology and psychology research into mystical messages",
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
    description=f"Give advice based on the research findings for: {topic}.",
    expected_output=f"{length_mapping.get(desired_answer_length, 'A short paragraph.')}",
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