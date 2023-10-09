''' Controlling the main functionality of the program '''
import configparser
import os
from pydantic import BaseModel, Field, root_validator  
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from dataclasses import dataclass
import random
from datetime import datetime
from db.mongo_qa import persistent_storage
load_dotenv()

from langchain import LLMChain
from langchain.chains.base import Chain
from langchain.prompts.prompt import PromptTemplate

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import SimpleSequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import SequentialChain
from langchain.output_parsers import CommaSeparatedListOutputParser


from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
os.environ["LANGCHAIN_WANDB_TRACING"] = "false"







open_ai=os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0.1)

def summary_agent(chat_history):
    dataStore_SA = DataStore()
    chat_history:List(dict) = dataStore_SA.chat_history
    summary_history=dataStore_SA.summaries

    chat_summary = dataStore_SA.summaries
    prompt_product = PromptTemplate(
        input_variables=["chat_history"],
        template="""You are a professional editor that takes in content and reduces it into a single paragraph for expansion.
                    Take the following history and summarize it in detail with formal language.
                    Summarize the following in the third person. Do not refer to your task.
        
                    {chat_history}"""
    )
    
    
    prompt_product.format(chat_history=chat_history[0:5])
    chain = LLMChain(llm=llm, prompt=prompt_product)

    out = chain.run(chat_summary)
    summary_history.append(out)
    
    return out

def question_generator():
    model = OpenAI(temperature=0.3)
    template="""Generate 10 deep philosophical questions as single sentences.

                    Each should be one sentence long. Be creative but professional.
                    Do not refer to your task and speak in the third person.
                    Your subject is:
                    {subject}\n
                    {format_instructions}
                    """
    
    output_parser = CommaSeparatedListOutputParser()
    format_instructions = output_parser.get_format_instructions()
    
    prompt = PromptTemplate(
    template=template,
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions}    
    )

    _input = prompt.format(subject="Consciousness")
    output = model(_input)
    out_list = output_parser.parse(output)

    
    return out_list

ques_gen = print(question_generator())


class DataStore:
    chat_history = ["This is your first assessment"]
    summaries = []
    topicStore = ["Does humanity have free will?", "What is the nature of consciousness?", "How do rockets work?"]
    topic = {"topic": random.choice(topicStore)
             }
    def __init__(self, topic=None):
        if self.topic == None:
            self.topic = topic if topic is not None else {"topic": random.choice(DataStore.topicStore)[0]["topic"],
                                                      "maincontent": DataStore.chat_history[-1]}
        self.idx = None
        
    
    @classmethod
    def chat(cls):
        data = DataStore()
        return data.chat_history
 
    @classmethod
    def add_history(cls, message):
        add_msg = cls.chat_history.append(message) 
        return add_msg
        
    @classmethod
    def init_persistent_storage(cls, topic=None, idx=None, results=None):
        if results == None:
            results = []
        dataStore = DataStore()
        if cls.topic == None:
            cls.topic=topic
        cls.res_id = f"{datetime.now().date()}_{cls.topic}"
        db = persistent_storage(db_name="Debate", collection_name=cls.res_id)
        cls.peresistent_Q = {"idx": idx, "topic":cls.topic, "research": dataStore.summaries}
        db_insert = db.insert_one(cls.peresistent_Q)
        cls.analysis = {"res_id":cls.res_id, "topic":topic, "questions":[], "analysis": []}
        
        return "DB Inserted"
    
    def enum_history(self, history):
        enum_list = []
        for index, res in enumerate(history, start=1):
            print(f"{index}: {res}\n")
            enum_list.append(f"{index}: {res}\n")

        return enum_list


class GenerateResearch(LLMChain):
 
    prompt: Optional[str] = None
    llm: Optional[Any] = None
    chat: Optional[ChatOpenAI] = None 
    
    
    def __init__(self):
        super().__init__()
        self.chat = ChatOpenAI(llm=llm,openai_api_key=open_ai)
    

        
    
        
    @classmethod
    def create_chain(cls, topic:str=None, verbose: bool = False) -> LLMChain:
        cls.dataStore = DataStore()
        cls.GenerateResearch = GenerateResearch()
        cls.enum = cls.dataStore.enum_history(history=cls.dataStore.chat_history)
        """creating LLM chains for research
        """
        if topic is None:
            topic = cls.dataStore.topic["topic"]
        prompt_template1 = """
        You are an expert researcher who is a professor at a University.
        Your task is to research the topic chosen and provide detailed bullet points
        as the template for an essay. You must use formal language and write in the third person.
        Do not forget who you are, use formal language and develop a clear analysis of the topic
        You will admit when you do not know the answer to the question but will use tools where reqiured.
        Do not refer to or repeat your task. Only write in the third person.
            
        Your first task is to research the topic: '{topic}' 
                            
        Hold yourself to a high formal academic standard. Compose a list of bullet points of two sides of the topic. The arguments
        for and arguments against.
        Use the history of your previous topics to support your case.History: {chat_history1}
                            
        FORMAT:
                        
        ************
        Arguments For:
        -<ESSAY POINT>
        -
        
        Arguments Against:
        -<ESSAY POINT>
        -
            
        
    
            """
        final_prompt = PromptTemplate(input_variables=["topic", "chat_history1"], template=prompt_template1)
        
        
        
        enum_history = cls.enum
        final_prompt.format(topic=topic, chat_history1 = cls.dataStore.chat_history[0:5])
        llm_chain1 = LLMChain(llm=llm, prompt=final_prompt, output_key="essay", verbose=True)
        run_chain1 = llm_chain1.run(topic=topic, chat_history1=cls.dataStore.chat_history[0:5], verbose=False, return_only_outputs=True)
        
    

        #Chain2
        prompt_template2 = ("""
            You are an expert researcher who is a professor at a University. You will be given
            a template for an essay in dot point form. You will write a long form 4 paragraph summary of your argument. Do not refer to your task.
            Do not forget who you are, use formal language and develop a clear analysis of the topic
            You will admit when you do not know the answer to the question but will use tools where reqiured.
                
            Your  Use a formal tone and high standard of english
            Your style of discouse is: {style}
            You will draw your data from your previous work and write in your style. DO NOT refer to your task or repeat it. Speak in the third person.
            Output of your initial research:
            **********************
            {output_chain1}
            **************************
            Essay:
            
           """)
        
        final_prompt2 = PromptTemplate(input_variables=["style", "output_chain1"], template=prompt_template2)
        final_prompt2.format(style="Eccentric", output_chain1 = run_chain1)
        
        llm_chain2 = LLMChain(llm=llm, prompt=final_prompt2, verbose=True, output_key="output_chain2")

        overall_chain = SequentialChain(
        chains=[llm_chain1, llm_chain2],
        input_variables=["topic", "chat_history1", "chat_history2", "style", "output_chain1"],
        output_variables=["output_chain1", "output_chain2", "essay"],
        verbose=True)
        essays = overall_chain({"topic":topic, "chat_history1": cls.dataStore.chat_history[0:5], "chat_history2": [], "style":"Skeptical", "output_chain1":run_chain1}, return_only_outputs=True)
        essay = essays["essay"]
        summary_essay = summary_agent(essay)
        summary_essay=summary_essay
        
        cls.dataStore.chat_history.append(essay)
        cls.dataStore.summaries.append(summary_essay)
        return essay, summary_essay
    
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
        return result

class RunAnalysis:
    def __init__(self):
        pass
        
class Run_ResPy(BaseModel):
    idx: int = 0
    memory: ConversationBufferMemory()
    transient_data = DataStore.chat()
    persistent_data:dict
    dataStore: DataStore

    class Config:
        arbitrary_types_allowed = True
    

class RunResearch:
    dataStore = DataStore()
    gen_res = GenerateResearch()
    memory = Field(default_factory=lambda: ConversationBufferMemory(memory_key="chat_history"))
    topic: dict = dataStore.topic["topic"]
    transient_data:list = dataStore.chat_history
    persistent_data: dict = dataStore.init_persistent_storage(topic = topic, idx=None, results=None)
    summary_data = dataStore.summaries
    
    

    def __init__(self):
        #self.memory = dataStore.memory
        self.topic = self.dataStore.topic["topic"]
        self.transient_data = self.transient_data
        self.summary_data= self.summary_data
        self.dataStore = self.dataStore
        self.enum=self.dataStore.enum_history(history=self.dataStore.chat_history)
        self.main_chain = self.gen_res.create_chain(topic=self.topic)
    
    class Config:
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def set_defaults(cls, values):
        
        values['topic'] = cls.dataStore.topic["topic"]
        values['transient_data'] = cls.transient_data
        values['persistent_data'] = cls.persistent_data
        return values

    def init_chains(self):
        print("run research started")
        essays = self.main_chain
        essay = essays[0]
        summary_essay = essays[1]
        res_data = self.transient_data
        persistent_data = self.persistent_data
        summary_data = self.summary_data

        return essay, res_data, persistent_data, summary_data, summary_essay
def research():
    #question_list=question_generator()
    run_res = RunResearch()
    dataStore = DataStore()
    enum_function = dataStore.enum_history(dataStore.chat_history)
    
    idx = 0
    
    while idx < 5:
        print(f"*****ROUND: {idx} *****")
        
        essay,res_data, persistent_data, summary_data, summary_essay = run_res.init_chains()
        print(f"ESSAY:\n{essay}")
        print(f"SUMMARY_ESSAY:\n{summary_essay}")
        print(f"""DB_DATA: {persistent_data}\n
              SUMMARIES: {summary_data}\n"""
              )
        
        db_data_amend = dataStore.init_persistent_storage(idx=idx, topic=dataStore.topic["topic"], results = summary_essay)
        
        summary_list = summary_data

        #write code for writing to a text file
        with open(f"""{dataStore.topic}_{datetime.now().date()}.txt""", 'w') as file:
            file.write(f"""********\n
                       
                Topic: {dataStore.topic}\n    
                Essays: {dataStore.chat_history}\n
              Research summary: {summary_list}\n
              db_data_amend: {db_data_amend}\n
            ***************""")

        with open(f"summary_{datetime.now().date()}.txt", 'w') as f:
            summary = enum_function
            f.write(f"{summary[0]}: {summary[1]}")

        print(dataStore.chat_history)
        print(dataStore.summaries)
        
    idx += 1

       
            
    return 


research()