
# Agentic Workflow playbook

In this notebook, you will learn how to create
1. LLM Agent without tools
2. LLM Agent with tools
3. RAG Agent
4. Agentic workflow using langgraph

# Pre-requisites
"""

!pip install -U langgraph langchain langchain-openai langchain-community faiss-cpu

"""*Import the packages*"""

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

"""*Set up OPENAI API Key in the environment variable*"""

os.environ["OPENAI_API_KEY"] = ""

"""*Generic agent class definition*"""

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

"""# 1. LLM agent without tools

A Large Language Model (LLM) agent created without the use of additional tools can only answer general questions based on the knowledge it was trained on up to a specific point in time.

*Create a LLM agent with an instruction prompt*
"""

model = ChatOpenAI(model="gpt-4o")
prompt = """
    You are an Employee Infobank. When you get questions about employees and their reporting structure, call get_employees tool
"""

"""*Agent can answer general question.*


"""

organisation_chatbot_agent = Agent(prompt, [], model)
organisation_chatbot_agent.create()
config = {"configurable": {"thread_id": "test-thread"}}

print(organisation_chatbot_agent.invoke([("user", "Who is Gandhiji?")], config))

"""*Despite configured with a Role-specific instruction prompt, LLM agent cannot answer context-specific question but it is sensible enough to ask for more context from the user.*"""

print(organisation_chatbot_agent.invoke([("user", "Who is Alice?")], config))

"""# 2. LLM Agent with tools

Let's create a LLM Agent with a tool querying an example Organisation Employee database to answer employee specific questions which is out of its pre-trained knowledge.

*Create sqlite DB to host employee table*
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

"""*Tool function definition to retrieve the employee list from an employee database*"""

from langchain_core.tools import tool

@tool
def get_employees():
  """Gets the employees list."""
  conn = create_connection("org.db")
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

"""*LLM Agent is created with "get_employees" tool*

You can observe that LLM agent has the capability now to answer employee-specific questions
"""

tools = [get_employees]
organisation_chatbot_agent = Agent(prompt, tools, model)
organisation_chatbot_agent.create()
config = {"configurable": {"thread_id": "test-thread"}}

print(organisation_chatbot_agent.invoke([("user", "Who is Alice?")], config))
print(organisation_chatbot_agent.invoke([("user", "How many employees are reporting to Alice?")], config))

"""# 3. RAG Agent

A Retrieval-Augmented Generation (RAG) agent is like a smart assistant for complex questions. Imagine you have a friend who has access to an enormous library and a fast way to find the exact information you need. This friend also has a talent for putting together the information into clear, coherent answers. That's what a RAG agent does – it tackles your complex questions by searching through a wealth of information and generating detailed responses based on what it finds.

Comparison between RAG Agent and LLM Agent with Tools

**RAG Agent:**

A RAG agent excels at handling complex, nuanced queries. When you ask a complicated question that requires gathering and combining information from various sources, the RAG agent searches through its vast dataset, retrieves the most relevant pieces, and synthesizes a comprehensive response. This makes it highly effective for answering intricate questions where the answer isn't straightforward or readily available.

**LLM Agent with Tools:**

An LLM (Large Language Model) agent with tools is designed to be a quick, reliable responder for clear and definite questions. These tools might include access to calculators, databases, or APIs that provide precise data. For instance, if you need a specific fact or a clear-cut answer, the LLM agent uses these tools to provide a speedy and accurate response. However, it might struggle with more complex queries that require nuanced understanding and extensive information retrieval.

*RAG Agent class definition which can provide department location info of our example organisation*
"""

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

"""# 4. Agentic Workflow using langgraph

In this section, we will build an all-in-one Organisation chatbot agent by integrating both LLM agent with tools and RAG agent

*Definition of agent state attributes used in the agentic workflow*
"""

class AgentState(TypedDict):
    query: Sequence[BaseMessage]
    result: str

"""*Definition of agent functions*"""

# Based on LLM agent with tools
def org_chatbot(state):
    print(f'Org agent:')
    query = state['query']
    prompt = """
      You are an Organisation chatbot. Follow the below rules:
      1. When you get questions about employees and their reporting structure, call get_employees tool
      2. When you get questions about departments, route the query to "department_navigator" agent
    """
    tools = [get_employees]
    organisation_chatbot_agent = Agent(prompt, tools, model)
    organisation_chatbot_agent.create()
    config = {"configurable": {"thread_id": "test-thread"}}
    result = organisation_chatbot_agent.invoke([("user", query)], config)
    return {'result': result}

# Routing the department location questions to RAG agent "department_navigator"
def route(state):
    result = state['result']
    if "department_navigator" in result:
        return "department_navigator"
    else:
        return END

# Based on RAG agent
def department_navigator(state):
    print(f'department_navigator agent:')
    query = state['query']
    result = llm_navigator.chat_with_llm(query)
    return {'result':result}

"""Organisation Chatbot agent follows ReAct prompting technique which thinks and decides to take action on how to answer general and context-specific questions like:
*   Answers the general questions based on its pre-trained knowledge

*   Responds to the context-specific questions either by using tool or by routing to the another specialised agent

Many specialised agents like department-specific, process-specific can be developed and connected to the Organisation chatbot agent to expand its capability to answer more context-specific complex questions

*Agentic workflow definition using langgraph*
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

"""*Organisation chatbot demo*"""

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