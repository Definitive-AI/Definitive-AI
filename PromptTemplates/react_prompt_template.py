from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate




def react_system_template():
    system = """{system} \nIf you are unable to answer the question with a tool, then answer the question with your own knowledge."""

    react_prompt = """Do the preceeding tasks and answer the following questions as best you can. You have access to the following tools:
    [{tools}]
    Use the following format:
    Input: the inputs to the tasks you must do
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now have completed all the tasks
    Final Answer: the final answer to the original input 

    IMPORTANT: Every <Thought:> must either come with an <Action: and Action Input:> or <Final Answer:>

    Begin!
    Question: {input}
    Thought:{agent_scratchpad}"""

    messages = [    SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=system)), 
                    MessagesPlaceholder(variable_name='chat_history', optional=True), 
                    HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['tool_names', 'tools', 'agent_scratchpad', 'input',], template=react_prompt))]

    return ChatPromptTemplate.from_messages(messages)


def usageExample(system):
    react_system_template().format(system=system, tools="{tools}",tool_names="{tool_names}",chat_history="{chat_history}", input = "{input}", agent_scratchpad = "{agent_scratchpad}")
