import os
from groq import Groq
import json
from dotenv import load_dotenv
from tavily import TavilyClient
load_dotenv() 
MODEL = 'llama3-8b-8192'
client = Groq(
    api_key=os.getenv('GROQ_API_KEY'),

)



tavily_client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))


def get_response(question):
    return json.dumps({"question": question})
def mental_assistance(context):
    return json.dumps({"context":search_results(get_query(context))})
def get_query(input):
    chat_completion =  client.chat.completions.create(
    messages=[
        {"role":"system",
         "content": "You are a query generating agent for a mental health care bot based on the input given by the user generate a query for web search. Just return  a single line of query. "
         },
        {   "role": "user",
            "content": f'Generate a web search query for {input}',
        }
    ],
    model="llama3-8b-8192",
)

    return chat_completion.choices[0].message.content

def search_results(query):
    query = query + "get some resources and references"
    context = tavily_client.get_search_context(query=query)
    print(context)
    return context
tools = [
        {
        "type": "function",
        "function": {
            "name": "get_response",
            "description": "Responding a casual chat",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Responding a casual chat",
                    }
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function":{
            "name": "mental_assistance",
            "description": "Help the user if he seems to be dealing out with emotions or mental issues or seeks help emotionally by searching the website and getting the references. Dont forget to show the reference and links from context.",
            "parameters": {
                "type":"object",
                "properties":{
                    "context": {
                        "type": "string",
                        "description": "The feelings emotions or mental issues of the user(e.g: 'I am feeling sad', 'I am depressed help me)"
                    }
                },
                "required": ["context"],
            },    
              }
    }]

def rag_func(prompt):
    messages=[
        {
            "role":"system",
            "content":"You are a function calling LLM that helps the user to deal with mental health isssues by providing them suggestions and solutions with the context and references extracted  from mental_assistance function to answer and help the user"
        },

        {   "role": "user",
            "content": prompt.content,
        }
    ]
    
    chat_completion =  client.chat.completions.create(
    messages=messages,
    model="llama3-8b-8192",
    tools = tools,
    tool_choice = "auto",
    max_tokens = 4096
)   
    response_message = chat_completion.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        available_functions = {
            "get_response": get_response,
            "mental_assistance": mental_assistance,
        }
        messages.append(response_message)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(**function_args)
            
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            })

        second_response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        final_response = second_response.choices[0].message.content
    else:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=4096
        )
        messages.append(response_message)
        final_response = response.choices[0].message.content

    response_data = {
        "model": "llama3-8b-8192",
        "response": final_response
    }
    
    return response_data