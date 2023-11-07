from flask import Flask, request, jsonify
from langchain import PromptTemplate, LLMChain
import chainlit as cl
from langchain import HuggingFaceHub
from translate import Translator as TextTranslator

app = Flask(__name__)

a = []  # Create a list to store the selected language

def select_language(selected_language):
    a.append(selected_language)

@app.route('/select_language', methods=['POST'])
def set_language():
    data = request.get_json()
    selected_language = data.get('language', 'English')
    select_language(selected_language)
    return jsonify({"message": "Language selected successfully"})

repo_id = "tiiuae/falcon-7b-instruct"
# Create the HuggingFaceHub object
llm = HuggingFaceHub(
    huggingfacehub_api_token='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',
    repo_id=repo_id,
    model_kwargs={"temperature": 0.3, "max_new_tokens": 1024}
)

template = """ Task: write a specific answer to a question related to education only, giving reference to the textbooks.
Topic: education
Style: Academic
Tone: Curious
Audience: 5-10 year olds
Length: 1 paragraph
Format: Text
Here's the question. {question}
"""

@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('user_input', '')
    
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    # Call the chain asynchronously
    res = llm_chain.acall(user_input, callbacks=[cl.AsyncLangchainCallbackHandler()])
    
    if a and a[0] == "Spanish":
        translator = TextTranslator(to_lang="es")
    elif a and a[0] == "French":
        translator = TextTranslator(to_lang="fr")
    else:
        translator = TextTranslator(to_lang="en")
    
    response_text = translator.translate(res["text"])

    return jsonify({"response": response_text})

if __name__ == '__main__':
    app.run(debug=True)

