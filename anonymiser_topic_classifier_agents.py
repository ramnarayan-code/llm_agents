# Databricks notebook source
# MAGIC %pip install -U langchain langgraph langchain_openai 

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import operator
from typing import Annotated, Any, Dict, Optional, Sequence, TypedDict, List, Tuple

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain.chains import create_structured_output_runnable
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langgraph.graph import END, StateGraph

import os
os.environ["OPENAI_API_VERSION"] = ""
os.environ["AZURE_OPENAI_ENDPOINT"] = ""
os.environ["AZURE_OPENAI_API_KEY"] = ""

llm=AzureChatOpenAI(deployment_name="gpt-4")

# COMMAND ----------

class Anonymizer(BaseModel):
    anonymized_review: str = Field(
        description="anonymized reviews"
    )

anonymizer_prompt = ChatPromptTemplate.from_template(
    '''
     You must anonymize the below provided "Text" by replacing with the placeholders: [day of the week], [time of the day], [point in time], [gender], [workplace], [age], [nationality], [appearance]. The text is a food retailer customer review.
    This is the examplary list of entities that you need to replace:
    1. day of the week - Monday, Tuesday, Wednesday, Thursday, Friday, Saturday
    2. time of the day - morning, afternoon, evening, lunchtime
    3. point in time - after opening, before closing, late, early, hour
    4. gender - he, she, his, her, man, woman, gentleman, lady, girl, boy
    5. workplace - cashier, behind the cash register, behind the till, staff at checkout, saleswoman, management, inventory management, manager, branch leader
    6. age - young, old
    7. nationality - accent, speaks foreign language
    8. appearance - blond, brunette, ginger, red hair
    9. person - man, woman, gentleman, lady, girl, boy
    10. person_name - name of the person
    11. location - city, country 


    Text:
    {unanonymized_review}
'''
)

anonymizer_structured_output = llm.with_structured_output(Anonymizer)
anonymizer_agent = anonymizer_prompt | anonymizer_structured_output


# COMMAND ----------

anonymizer_agent.invoke({"unanonymized_review": [HumanMessage(content="A young man named John working in Essen branch is not good")]})

# COMMAND ----------


class TopicSentimentClassifier(BaseModel):
    topic: str = Field(
        description="topics"
    )
    sentiment_score: str = Field(
        description="sentiment_score"
    )

topic_sentiment_prompt = ChatPromptTemplate.from_template(
    """
    You are a Topic classifier
    1. You need to classify the given "Text" under one of the topics:
    - accessibility_and_safety: accessibility and safety of the store
    - parking_spaces: parking spaces
    - store_location: store location
    - friendliness: friendliness
    2. Analyze the review and assign a sentiment score (negative, neutral, positive)
    3. Provide your response in the class "TopicSentimentClassifier" matching its attribu following the example: \'{{"topic": "accessibility_and_safety", "sentiment_score": "positive"}}\' do not provide any comments. If no topics are found just reply with: "No topics found."
    Text:
    {anonymized_review}
 """
)

topic_sentiment_structured_output = llm.with_structured_output(TopicSentimentClassifier)
topic_sentiment_agent = topic_sentiment_prompt | topic_sentiment_structured_output

# COMMAND ----------

topic_sentiment_agent.invoke({"anonymized_review": [HumanMessage(content="A [age] [gender] named [person_name] working in [workplace] [location] branch is  friendly")]})

# COMMAND ----------

class AgentState(TypedDict):
    unanonymized_review: str
    anonymized_review: str
    analyzed_review: str
    topic: str
    sentiment_score: str
     

# COMMAND ----------

def anonymizer(state):
    print(f'Anonymizer agent:')
    response = anonymizer_agent.invoke({"unanonymized_review":state['unanonymized_review']})
    return {'anonymized_review':response.anonymized_review}

def topic_sentiment_classifier(state):
    print(f'Topic Sentiment classifier agent:')
    response = topic_sentiment_agent.invoke({"anonymized_review": state['anonymized_review']})
    return {'topic': response.topic, 'sentiment_score': response.sentiment_score}

# COMMAND ----------

workflow = StateGraph(AgentState)

# Define the nodes
workflow.add_node("anonymizer", anonymizer)
workflow.add_node("topic_sentiment_classifier", topic_sentiment_classifier)

# Build graph
workflow.set_entry_point("anonymizer")
workflow.add_edge("anonymizer", "topic_sentiment_classifier")
workflow.add_edge("topic_sentiment_classifier", END)

app = workflow.compile()

# COMMAND ----------

for stream_msg in app.stream(
    {"unanonymized_review": [HumanMessage(content="A young man named John working in Essen branch is not good")]},
    {"recursion_limit": 100},
):
    if "__end__" not in stream_msg:
        if "anonymizer" in stream_msg:
          anonymized_review = stream_msg["anonymizer"]["anonymized_review"]
          print(anonymized_review)
        elif "topic_sentiment_classifier" in stream_msg:
          print(stream_msg)
        print("----")