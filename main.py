''' Controlling the main functionality of the program '''
import configparser
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import random
import datetime
from datetime import date
load_dotenv()

from langchain import LLMChain
from langchain.chains.base import Chain
from langchain.prompts.prompt import PromptTemplate
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.chains import SimpleSequentialChain
from langchain.memory import ConversationBufferMemory

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

open_ai=os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(temperature=0.9)
prompt_product = PromptTemplate(
    input_variables=["product"],
    template="""What is a good name for a company that makes {product}?"""
)
prompt_product.format(product="cat houses")


chain1 = LLMChain(prompt = prompt_product, llm=llm)
messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="I love programming.")
]

class DataStore:
    idx = 0
    chat_history = [f"Researching our topic of choice"]
    topicStore = ["Does humanity have free will?", "What is the nature of consciousness?", "How do rockets work?"]
    topic = {"topic": random.choice(topicStore),
             "maincontent": chat_history[-1],
             }
    def __init__(self, topic=None):
        if topic == None:
            self.genRes = DataStore(self)
            self.topic = self.genRes.topic
            self.chat_history = self.genRes.chat_history
            
                  

    @classmethod
    def add_history(cls, message):
        add_msg = cls.chat_history.append(message) 
        return add_msg
        
    @classmethod
    def persistent_storage(cls, topic=None):
        cls.topic=topic
        cls.res_id = f"{datetime.date.today()}_{cls.topic}"
        cls.idx = cls.idx
        cls.peresistent_Q = {"idx": cls.idx, "topic":cls.topic, "research": []}
        cls.analysis = {"res_id":cls.res_id, "topic":topic, "questions":[], "analysis": []}


class GenerateResearch(LLMChain):
    DS_class = DataStore()
    res_data:List[dict] = DS_class.persistent_storage()
    
    prompt: Optional[str] = None
    llm: Optional[Any] = None
    chat: Optional[ChatOpenAI] = None 
    
    def __init__(self, open_api_key=open_ai):
        super().__init__()
        self.chat = ChatOpenAI(openai_api_key=open_ai)
        
    @classmethod
    def create_chain(cls, topic:str=None, verbose: bool = False) -> LLMChain:
        """creating LLM chains for research
        """
        if topic is None:
            topic = DataStore().topic
        prompt_template1 = ("""
            You are an expert at research who is a professor at a University.\n
            Do not forget who you are, use formal language and develop a clear analysis of the topic\n
            You will admit when you do not know the answer to the question but will use tools where reqiured.\n
                
            Your first task is to research a '{topic}' to a high formal academic standard. Compose a list of bullet points of two sides of the topic\n
            Format is:
            Arguments For:
            Arguments Against: 
            {maincontent}.\n
            """)
        final_prompt = PromptTemplate(input_variables=["topic", "maincontent"], template=prompt_template1)
        final_prompt.format(topic=topic, maincontent=DataStore.self.data[-1])
        memory = ConversationBufferMemory(memory_key="chat_history")
        llm_chain1 = LLMChain(llm=llm, prompt=final_prompt, verbose=True, memory=True)
    

        #Chain2
        prompt_template2 = ("""
            You are an expert at research who is a professor at a University.\n
            Do not forget who you are, use formal language and develop a clear analysis of the topic\n
            You will admit when you do not know the answer to the question but will use tools where reqiured.\n
                
            Your second task is to expand on the key points to 3 full paragraphs. Use a formal tone and high standard of english'\n
            Your style of discouse is {style}
            You will draw your data from your previous work
            {maincontent}.\n
           """)
        final_prompt2 = PromptTemplate(input_variables=["style", "maincontent"], template=prompt_template2)
        final_prompt2.format(style="Skeptical", maincontent=RunResearch.self.data[-1])
        memory = ConversationBufferMemory(memory_key="chat_history")
        llm_chain2 = LLMChain(llm=llm, prompt=final_prompt2, verbose=True, memory=memory)

        overall_chain = SimpleSequentialChain(chains=[llm_chain1, llm_chain2], verbose=True)
        print(overall_chain)
        return overall_chain
    
class RunAgent(LLMChain):
    def __init__(self):
        pass
    def run_analysis(self, chain_instance):
        batch_messages = [
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="I love programming.")
    ],
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="I love artificial intelligence.")
    ],
]
        result = chat.generate(batch_messages)
        return (print(result))

class RunAnalysis:
    def __init__(self, dataStore=DataStore()):
        self.transient_data = dataStore.chat_history
        self.persistent_data = dataStore.persistent_storage(self, topic=DataStore.topic)
    
class RunResearch(LLMChain):
    idx: int = 0    
    def __init__(self):
        self.ds = DataStore(self) 
        self.data = self.ds.chat_history
        self.topic = self.ds.topic
        self.ds.idx = RunResearch.idx
        self.memory = ConversationBufferMemory(memory_key="chat_history")

    def init_chains(self):   
        run_research = GenerateResearch(LLMChain)
        print(f"run research started")
        llm_chains = run_research.create_chain(topic=self.topic)
        self.data.append(llm_chains)
        print(llm_chains)
    
        cls.idx += 1
        return(llm_chains)

def research():
    research = RunResearch()
    execute = research.init_chains()
    print(execute)
    return execute

print(research())