#!/usr/bin/env python
# coding: utf-8

# In[19]:


#INSTALANDO AS LIBS
#!pip install yfinance
#!pip install crewai
#!pip install 'crewai[tools]'
#!pip install langchain
#!pip install langchain-openai
#!pip install langchain_community


# In[2]:


import numpy as np
np.float_ = np.float64


# In[18]:


#IMPORT LIBS
import json
import os 
from datetime import datetime
 
import yfinance as yf
from crewai import Agent, Task, Crew, Process

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st


# In[4]:


#CRIANDO YAHOO FINANCE TOOL
def fetch_stock_price(ticket):
    stock = yf.download(ticket, start="2023-08-08", end="2024-08-08")
    return(stock)

yahoo_finance_tool = Tool(
    name = "Yahoo Fincance Tool",
    description=" Fetches stocks prices for {ticket} from the lest year about a specific company from Yahoo Finance API",
    func=lambda ticket: fetch_stock_price(ticket)

)


# In[5]:


# IMPORTANTE OPENAI LLM - GPT
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-3,5-turbo")


# In[6]:


stockPriceAnalyst = Agent(  
    role= "Senior stock price Analyst",
    goal= "Find the{ticket} stock price and analyses trends",
    backstory= """You're highly experienced in analyzing the price of an specific stock
and make prediction about its future price. """,
    verbose= True,
    llm= llm,
    max_iter=5,
    memory=True,
    allow_delegation= False,
    tools=[yahoo_finance_tool]

)


# In[7]:


getStockPrice = Task(
    description= "Analyze the stock{ticket} price history and create a trend analyses of up, down or sideways",
    expected_output = """Specify the current trend sotck price - up, down or sideways. eg. stock ='APPL, price up' """,
    agent = stockPriceAnalyst
)


# 
# __________________________________________________________________________________________________
# AGENTE DE NOTICIAS 

# In[8]:


#IMPORTANTO A TOOL DE SEARCH
search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)


# In[9]:


newAnalyst = Agent(
    role="Stock News Analyst",
    goal="""Create a short samary of the news related to the stock {ticket} company. Specify the current trend - up ,down or sideways
    with the news context. For  each request sotck asset, specify a numbet between 0 and 100, where 0 is extreme fear and 100 is extreme gread. """,
    backstory="""You're highly experienced in analyzing the market trends and news and have tracked assest for more then 10 years.
     You'se also master level analyts in the tradicional markets and have deep understanding of human psychology.
     You understand news, theirs tittles and information, but you look at those with a health dose of skepticisn.
     You consider also the source of the news articles . 
     """,
    verbose= True,
    llm= llm,
    max_iter=10,
    memory=True,
    allow_delegation=False,
    tools=[search_tool]
)


# In[10]:


get_news = Task(
    description= f"""Take the stock and include BTC to it (if not request).
    Use the search tool to search each one individually.

    The current date is {datetime.now()}.

    Compose the results into a helpfull report """,
    expected_output = """A sumary of the overall market and one sentence summary for each request asset.
    Include a fear/gread score for each asset bused on the news. Use format:
    <STOCK ASSET>
    <SUMARRY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORE>
    """,
    agent = newAnalyst

)


# _________________________________________________________________________________________________________________
# AGENTE QUE VAI FAZER A ANALISE DE FATO 

# In[11]:


stockAnalystWrite = Agent(
    role="Senior Stock Analyst Writer",
    goal="""Analyze the trends price and news and write an insighfull compelling and informative 3 paragraph long newsletter based on the stock report and price trend.""",
    backstory=""" You're  widely accepted as the best stock analyst in the market. You understand complex concepts and create compeling
    stories and narratives that resonate with wider audiences.

    You  understand macro factors and combine multiple theories - eg. cycle theory and fundamental analyses. 
    You're able to hold multiple opinions when analyzing anything.

     """,
    verbose= True,
    llm= llm,
    max_iter=5, 
    memory=True,
    allow_delegation=True
)


# In[12]:


writeAnalyses = Task(
    description= """Use the stock price trend and the stock news report to create an analyses and write the newsletter about the 
    {ticket} company that is brief and highlights the most important points.
    Focus on the stock price trend , news and fear/greed score . What are the future considerations?
    Include the previous analyses of stock trend and news summary.
""",
    expected_output= """An eloquent 3 paragraphs newsletter formated as markdown in an easy readable manner. It should contain:

    - 3 bullet executive summary 
    - Introduction - set the overall picture and spike up the interest
    - main part provides the meat of the analysis including the news summary and fear/greed scores
    - summary - key facts and concrete future trend prediction - up,down or sideways.
""",
    agent= stockAnalystWrite,
    context= [getStockPrice, get_news ]
)


# In[13]:


crew =Crew(
    agents=[stockPriceAnalyst,newAnalyst, stockAnalystWrite],
    tasks=[getStockPrice, get_news, writeAnalyses],
    verbose= False,
    process= Process.hierarchical,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15
)

#result= crew.kickoff(inputs={'ticket':'AAPL'})



with st.sidebar:
    st.header('Enter the stock to Research')

    with st.form(key='research from'):
        topic = st.text_input('Select ticket')
        submit_button = st.form_submit_button(label="Run Research")

if submit_button:
    if not topic:
        st.error("Please fill the ticket field")

    else:
        result= crew.kickoff(inputs={'ticket':topic})


        st.subheader("Results of your research")
        st.write(result['final_output'])



