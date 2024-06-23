from collections import defaultdict
from typing import List

from langchain.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from langgraph.prebuilt import ToolNode
from typing import Literal
from langchain_anthropic import ChatAnthropic
import json
from langchain.agents import tool

from langchain_community.document_loaders import WebBaseLoader
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.document_loaders import AsyncChromiumLoader
import re

import datetime

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langsmith import traceable
from langgraph.graph import END, MessageGraph


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    specifications: str = Field(description="Critique of meeting specifications.")
    superfluous: str = Field(description="Critique of what is superfluous.")

class FinalAnswer(BaseModel):    
    answer: str = Field(description="Final Answer to the question.")

class AnswerQuestion(BaseModel):
    """Answer the question. Provide an answer, reflection, and then follow up with search queries to improve the answer."""
    answer: str = Field(description='A detailed answer to the question.')
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    tools: str = Field(
        description="Results from tools for improvements to address the critique of your current answer."
    )

class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question. Explain your reasoning."""
    references: list[str] = Field(
        description="Reasons for your updated answer. Explain your reasoning."
    )    

class ResponderWithRetries:
    def __init__(self, runnable, validator):
        self.runnable = runnable
        self.validator = validator

    def respond(self, state: list):
        response = []
        for attempt in range(2):
            response = self.runnable.invoke(
                {"messages": state}, {"tags": [f"attempt:{attempt}"]}
            )
            try:
                self.validator.invoke(response)
                return response
            except ValidationError as e:
                state = state + [
                    response,
                    ToolMessage(
                        content=f"{repr(e)}\n\nPay close attention to the function schema.\n\n"
                        + self.validator.schema_json()
                        + " Respond by fixing all validation errors.",
                        tool_call_id=response.tool_calls[0]["id"],
                    ),
                ]
        return response
    
class Responder:
    def __init__(self, runnable, validator):
        self.runnable = runnable
        self.validator = validator  

    def respond(self, state: list):
        last = state[-1].content[-1]
        input = last["input"]
        answer = input["answer"]
        feedback = str(input["reflection"]) #"\nFeedback: " + feedback + 
        fst = str(len(answer.split()))
        len1 = len(answer.split())
        lines = answer.splitlines()

        midpoint = len(lines) // 2
        first_half_lines = lines[:midpoint]
        second_half_lines = lines[midpoint:]
        first_half = "\n".join(first_half_lines)
        second_half = "\n".join(second_half_lines)

        example = rpa_cloud_migration_retriever()

        if len1 < 700:
            response1 = self.runnable.invoke({"input": example + "Briefly expand, improve readability with subheadlines and lists or bullets, and improve with more detail on first half of the draft." +  "\n\nFirst Half Draft: " + first_half})
            response2 = self.runnable.invoke({"input": example + "Briefly expand, improve readability with subheadlines and lists or bullets, and improve with more detail on second half of the draft." + "\n\nSecond Half Draft: " + second_half + "\n\nExpanded first half:\n" + response1.content})
            return AIMessage(content=response1.content + "\n\n" + response2.content)
        else:
            response1 = self.runnable.invoke({"input": example + "Briefly expand, improve readability with subheadlines and lists or bullets, and improve with more detail on the draft." +  answer})
            return AIMessage(content=response1.content)
  
class ReflectionChain:
    def __init__(self, llm: ChatAnthropic, llm2: ChatAnthropic, prompt, tools, max_iterations: int = 5):
        self.llm = llm
        self.llm2 = llm2
        self.prompt = prompt
        self.tools = tools #tavily_tool
        self.max_iterations = max_iterations
        #self.tool_executor = ToolExecutor(tools)
        self.actor_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system", self.prompt + "\n{first_instruction}",
                ),
                MessagesPlaceholder(variable_name="messages"),
                (
                    "user",
                    "\n\n<system>Reflect on the user's original question and the"
                    " actions taken thus far. Respond using the {function_name} function.</reminder>",
                ),
            ]
        )
        self.initial_answer_chain = self.actor_prompt_template.partial(
            first_instruction="Reflect and critique your answer. Be severe to maximize improvement. Use the tools to research information and improve your answer.", 
            function_name=AnswerQuestion.__name__,
         ) | self.llm.bind_tools(tools=[AnswerQuestion])
        #AnswerQuestion.__name__
        self.validator = PydanticToolsParser(tools=[AnswerQuestion])

        self.first_responder = ResponderWithRetries(
            runnable=self.initial_answer_chain, validator=self.validator
        )
        self.revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
    - You should use the previous critique to remove superfluous information from your answer.
    - Be sure the answer conforms to the user specifications."""        
        self.revision_chain = self.actor_prompt_template.partial(
            first_instruction=self.revise_instructions,
            function_name=ReviseAnswer.__name__,
        ) | self.llm.bind_tools(tools=[ReviseAnswer])

        self.revision_validator = PydanticToolsParser(tools=[ReviseAnswer])
        self.revisor = ResponderWithRetries(runnable=self.revision_chain, validator=self.revision_validator)
        #self.tool_node = ToolNode(self.tools)        
        self.tool_node = ToolNode(
                [
                    StructuredTool.from_function(rpa_cloud_migration_retriever, name=AnswerQuestion.__name__),
                    StructuredTool.from_function(rpa_cloud_migration_retriever, name=ReviseAnswer.__name__),
                ]
            )        
        self.final_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system", self.prompt + "\n{first_instruction}",
                ),
                #MessagesPlaceholder(variable_name="messages"),
                (
                    "user",
                    "{input}",
                ),
            ]
        )        

        self.final_answer_chain = self.final_prompt_template.partial(
            first_instruction="Given the previous iterations and research, provide a final comprehensive answer to the original question. Be sure the final answer conforms to these specifications.", 
         ) | self.llm2.bind()
        
        self.final_responder = Responder(
            runnable=self.final_answer_chain, validator=self.validator
        )
        
        self.graph = self._build_graph()


    def tool_list(self,tools):
        structured_tools = [
            (StructuredTool.from_function(tool1, name=AnswerQuestion.__name__),
            StructuredTool.from_function(tool1, name=ReviseAnswer.__name__))
            for tool1 in tools 
        ]
        flattened_tools = [item for sublist in structured_tools for item in sublist]    
        return flattened_tools

    def _get_num_iterations(self, state: list):
        i = 0
        for m in state[::-1]:
            if m.type not in {"tool", "ai"}:
                break
            i += 1
        return i
    
    def execute_tools(self,data):
        # Get the most recent agent_outcome - this is the key added in the `agent` above
        agent_action = data['agent_outcome']
        output = self.tool_executor.invoke(agent_action)
        return {"intermediate_steps": [(agent_action, str(output))]}
    
    def event_loop(self, state: list) -> Literal["execute_tools", "revise", "final_answer", "__end__"]:
        num_iterations = self._get_num_iterations(state)
        if num_iterations == 1:
            return "execute_tools"
        elif num_iterations < self.max_iterations:
            return "revise"
        else:
            return "final_answer"
        
    def _build_graph(self):
        builder = MessageGraph()
        builder.add_node("draft", self.first_responder.respond)
        builder.add_node("execute_tools", self.tool_node)
        builder.add_node("revise", self.revisor.respond)
        builder.add_node("final_answer", self.final_responder.respond)
        builder.add_edge("draft", "execute_tools")
        builder.add_edge("execute_tools", "revise")
        builder.add_conditional_edges("revise", self.event_loop)
        builder.set_entry_point("draft")
        builder.set_finish_point("final_answer")
        return builder.compile()    
    
    def run(self, question: str) -> str:
        events = self.graph.stream(
            [HumanMessage(content=question)],
            stream_mode="values",
        )
        final_answer = None
        for i, step in enumerate(events):
            print(f"Step {i}")
            step[-1].pretty_print()
            if isinstance(step[-1], AIMessage):
                final_answer = step[-1].content
        return final_answer[0] 

