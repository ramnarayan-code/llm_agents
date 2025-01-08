# -*- coding: utf-8 -*-
"""agent_creator.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1bbXEmD1oOyF82Aj7kBFm35dRI-E3-G4P

# Agentic Workflow playbook

In this notebook, you will learn how to create
1. Simple Agent without tools
2. Agent with tools
3. RAG Agent
4. Agentic workflow using langgraph

# Pre-requisites
"""

!pip install -U langgraph langchain langchain-openai langchain-community faiss-cpu

import os
import sqlite3

import langgraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from typing import Annotated, Any, Dict, Optional, Sequence, TypedDict, List, Tuple
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph

os.environ["OPENAI_API_KEY"] = "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"

"""# 1. Simple agent without tools"""

class Agent:
  def __init__(self, prompt, tools, model):
    self.__system_message = prompt
    self.__tools = tools
    self.__model = model
    self.__memory = MemorySaver()

  def create(self):
    self.__agent = create_react_agent(self.__model, self.__tools, state_modifier=self.__system_message, checkpointer=self.__memory)

  def invoke(self, input, config=None):
    return self.__agent.invoke({"messages": input}, config)["messages"][-1].content

model = ChatOpenAI(model="gpt-4o")
prompt = """
    You are an Organisation chatbot. Follow the below rules:
    1. When you get questions about employees and their reporting structure, call get_employees tool
    2. When you get questions about departments, route the query to "department_navigator" agent
"""

"""Agent is created without tools. It can answer to general questions but it cannot answer context-specific questions."""

organisation_chatbot_agent = Agent(prompt, [], model)
organisation_chatbot_agent.create()
config = {"configurable": {"thread_id": "test-thread"}}

print(organisation_chatbot_agent.invoke([("user", "Who is Gandhiji?")], config))

print(organisation_chatbot_agent.invoke([("user", "Who is Alice?")], config))

"""# 2. Agent with tools

Create sqlite DB to host employee table
"""

def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)
    return conn

def create_table(conn, create_table_sql):
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except sqlite3.Error as e:
        print(e)

def insert_employee(conn, employee):
    sql = ''' INSERT INTO employee(id, name, designation)
              VALUES(?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, employee)
    conn.commit()
    return cur.lastrowid

def insert_reporting(conn, reporting):
    sql = ''' INSERT INTO reporting(id, manager_id)
              VALUES(?,?) '''
    cur = conn.cursor()
    cur.execute(sql, reporting)
    conn.commit()
    return cur.lastrowid

def main():
    database = "org.db"

    sql_create_employee_table = """ CREATE TABLE IF NOT EXISTS employee (
                                        id integer PRIMARY KEY,
                                        name text NOT NULL,
                                        designation text NOT NULL
                                    ); """

    sql_create_reporting_table = """ CREATE TABLE IF NOT EXISTS reporting (
                                        id integer NOT NULL,
                                        manager_id integer,
                                        FOREIGN KEY (id) REFERENCES employee (id)
                                    ); """

    conn = create_connection(database)

    if conn is not None:
        create_table(conn, sql_create_employee_table)
        create_table(conn, sql_create_reporting_table)

        employees = [(1, 'Alice', 'CEO'),
                     (2, 'Bob', 'CTO'),
                     (3, 'Charlie', 'COO'),
                     (4, 'David', 'CFO'),
                     (5, 'Eve', 'Head of Data & Analytics'),
                     (6, 'Fred', 'HR Head')]

        reportings = [(2, 1),  # Bob reports to Alice
                      (3, 1),  # Charlie reports to Alice
                      (4, 1),  # David reports to Alice
                      (5, 2),  # Eve reports to Bob
                      (6, 3)] # Fred reports to Charlie

        for employee in employees:
            insert_employee(conn, employee)

        for reporting in reportings:
            insert_reporting(conn, reporting)

        print("Database created and tables populated successfully.")
    else:
        print("Error! Cannot create the database connection.")

if __name__ == '__main__':
    main()

"""Tool function definition"""

from langchain_core.tools import tool

@tool
def get_employees():
  """Gets the employees list."""
  conn = create_connection("test.db")
  employees = {}
  try:
      cur = conn.cursor()
      cur.execute("""
          SELECT e.id, e.name, e.designation, r.manager_id, m.name as manager_name
          FROM employee e
          LEFT JOIN reporting r ON e.id = r.id
          LEFT JOIN employee m ON r.manager_id = m.id
      """)
      rows = cur.fetchall()
      for row in rows:
          emp_id, name, designation, manager_id, manager_name = row
          employees[emp_id] = {
              "name": name,
              "designation": designation,
              "manager_id": manager_id,
              "manager_name": manager_name
          }
  except sqlite3.Error as e:
      print(e)
  return employees

"""Agent is created with tools"""

organisation_chatbot_agent = Agent(prompt, tools, model)
organisation_chatbot_agent.create()
config = {"configurable": {"thread_id": "test-thread"}}

print(organisation_chatbot_agent.invoke([("user", "Who is Alice?")], config))
print(organisation_chatbot_agent.invoke([("user", "How many employees are reporting to Alice?")], config))

"""# 3. RAG Agent"""

class LLMRAGBasedNavigator:
    def __init__(self):
        self.__model = ChatOpenAI()
        self.__navigator_prompt_template = template = """
                    You are a department navigator. Provide the department location based on the context:
                    {context}

                    Question: {question}
                  """

    def __create_retreiver(self, context):
        self.__vectorstore = FAISS.from_texts(context, embedding=OpenAIEmbeddings())
        self.__retriever = self.__vectorstore.as_retriever()

    def __create_prompt(self):
        self.__prompt = ChatPromptTemplate.from_template(self.__navigator_prompt_template)

    def create_llm_chat_context(self, context):
        self.__create_retreiver(context)
        self.__create_prompt()
        self.__llm_chain = (
            {"context": self.__retriever, "question": RunnablePassthrough()}
            | self.__prompt
            | self.__model
            | StrOutputParser()
        )

    def chat_with_llm(self, question):
        response = self.__llm_chain.invoke(question)
        return response

context_for_department_navigation =  ["HR is located on the first floor",
                                  "Finance is located on the second floor",
                                  "Data & Analytics is located on the third floor",
                                  "Sales is located on the fourth floor",
                                  "Marketing is located on the fifth floor"]
llm_navigator = LLMRAGBasedNavigator()
llm_navigator.create_llm_chat_context(context_for_department_navigation)

llm_navigator.chat_with_llm("where is HR located?")

"""# 4. Agentic Workflow using langgraph"""

class AgentState(TypedDict):
    query: Sequence[BaseMessage]
    result: str

def org_chatbot(state):
    print(f'Org agent:')
    query = state['query']
    organisation_chatbot_agent = Agent(prompt, tools, model)
    organisation_chatbot_agent.create()
    config = {"configurable": {"thread_id": "test-thread"}}
    result = organisation_chatbot_agent.invoke([("user", query)], config)
    return {'result': result}

def route(state):
    result = state['result']
    if "department_navigator" in result:
        return "department_navigator"
    else:
        return END

def department_navigator(state):
    print(f'department_navigator agent:')
    query = state['query']
    result = llm_navigator.chat_with_llm(query)
    return {'result':result}

"""Organisation Chatbot agent follows ReAct prompting technique which thinks and decides to take action on how to answer general and context-specific questions like:
*   Answers to the general questions based on its pre-trained knowledge

*   Answers to the context-specific questions referring to the tool and routing to the related specialised agent

Organisation Chatbot agent created with tools is connected to Department Navigator RAG agent to answer more context-specific questions. Likewise, many agents can be developed and connected to the Org chatbot agent to widen its knowledge to answer more context-specific questions



"""

workflow = StateGraph(AgentState)

# Define the nodes
workflow.add_node("org_chatbot", org_chatbot)
workflow.add_node("department_navigator", department_navigator)

# Build graph
workflow.set_entry_point("org_chatbot")
workflow.add_conditional_edges("org_chatbot", route)
workflow.add_edge("org_chatbot", END)

app = workflow.compile()

print("Welcome to Virtual Org!. I am Virtual Org Chatbot. I can answer to both general and company-specific questions")

while True:
  query_input = input('Enter the query: ')
  if query_input == 'exit':
    print("See you next time...")
    break

  for stream_msg in app.stream({"query": query_input}):
    if "__end__" not in stream_msg:
        if "org_chatbot" in stream_msg:
          print(stream_msg["org_chatbot"])
        elif "department_navigator" in stream_msg:
          print(stream_msg["department_navigator"])
        print("----")