from groq import Groq
from dotenv import load_dotenv
import os
load_dotenv() 
client = Groq(
    api_key=os.getenv('GROQ_API_KEY'),

)

def diabetes_convo(data,result, prob):
    
    
    chat_completion =  client.chat.completions.create(
    messages=[
        {
            "role":"system",
            "content":"You are a diabetes result showing agent that tells the result of the user wheteher 0 which is noin diabetic or 1 which is diabeteic, along with that you aloso have to suggest tips based on the data and also the probability of the result provided in the content" },
        {   "role": "user",
            "content": f'data: {data},class: {result}, class_probaility: {prob} '
        }
    ],
    model="llama3-8b-8192",
)   
    
    return chat_completion.choices[0].message.content