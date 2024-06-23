from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from typing import List
from multiprocessing import Pool

def create_guided_reflection_agent(llm: BaseChatModel, prompt: str, msgs:List[str]) -> Runnable:
    class GuidedReflectionAgent(Runnable):
        def __init__(self, llm: BaseChatModel, prompt: str, msgs):
            self.llm = llm
            self.prompt = SystemMessage(content=prompt)
            self.msgs = msgs

        def invoke(self, query: str) -> str:
            generate_history = [self.prompt] #, HumanMessage(content=query)
            output = ""

            for msg in self.msgs:
                generate_history.append(HumanMessage(content=msg))
                output_response = self.llm._generate(generate_history)
                generate_history.append(AIMessage(content=output_response.content))
                
                output = output_response.content                
            return output

        async def arun(self, query: str) -> str:
            return self.run(query)

    return GuidedReflectionAgent(llm, prompt, msgs)

def create_guided_list_reflection_agent(llm: BaseChatModel, prompt: str, msgs:List[str]) -> Runnable:
    class GuidedReflectionAgent(Runnable):
        def __init__(self, llm: BaseChatModel, prompt: str, msgs):
            self.llm = llm
            self.prompt = SystemMessage(content=prompt)
            self.msgs = msgs

        def invoke(self, query: str) -> str:
            generate_history = [self.prompt] #, HumanMessage(content=query)
            output = []

            for msg in self.msgs:
                generate_history.append(HumanMessage(content=query))
                output_response = self.llm._generate(generate_history)
                generate_history.append(AIMessage(content=output_response.generations[0].text))
                
                output.append(output_response.generations[0].text)             
            return output

        async def arun(self, query: str) -> str:
            return self.run(query)

    return GuidedReflectionAgent(llm, prompt, msgs)

def create_iterative_reflection_agent(llm: BaseChatModel, prompt: str, msgs:List[str]) -> Runnable:
    class GuidedReflectionAgent(Runnable):
        def __init__(self, llm: BaseChatModel, prompt: str, msgs):
            self.llm = llm
            self.prompt = SystemMessage(content=prompt)
            self.msgs = msgs

        def invoke(self, query: str) -> str:
            output = ""
            for msg in self.msgs:
                output_response = self.llm._generate([self.prompt, HumanMessage(content=query['input'] + msg)])   
                output += output_response.generations[0].text + "\n\n"
            return output

        async def arun(self, query: str) -> str:
            return self.run(query)

    return GuidedReflectionAgent(llm, prompt, msgs)

def create_iterative_list_reflection_agent(llm: BaseChatModel, prompt: str, msgs:List[str]) -> Runnable:
    class GuidedReflectionAgent(Runnable):
        def __init__(self, llm: BaseChatModel, prompt: str, msgs):
            self.llm = llm
            self.prompt = SystemMessage(content=prompt)
            self.msgs = msgs

        def invoke(self, query: str) -> List[str]:
            def generate_response(msg):
                # This function will be executed in parallel for each message
                output_response = self.llm._generate([self.prompt, HumanMessage(content=msg)])
                return output_response.generations[0].text

            # Use Pool to run generate_response in parallel for each message
            with Pool(len(self.msgs)) as pool: # Adjust the number of processes as needed
                output = pool.map(generate_response, self.msgs)
            return output

        async def arun(self, query: str) -> str:
            return self.run(query)

    return GuidedReflectionAgent(llm, prompt, msgs)

def create_iterative_tuple_list_reflection_agent(llm: BaseChatModel, prompt: str, msgs:List[tuple[str, str]]) -> Runnable:
    class GuidedReflectionAgent(Runnable):
        def __init__(self, llm: BaseChatModel, prompt: str, msgs):
            self.llm = llm
            self.prompt = SystemMessage(content=prompt)
            self.msgs = msgs

        def invoke(self, query: str) -> List[tuple[str, str]]:
            output = []

            for (q,msg) in self.msgs:
                output_response = self.llm._generate([self.prompt, HumanMessage(content=query + "\n\n" + str(q + "\n" + msg))])                
                output.append((q,output_response.generations[0].text))
            return output

        async def arun(self, query: str) -> str:
            return self.run(query)

    return GuidedReflectionAgent(llm, prompt, msgs)

