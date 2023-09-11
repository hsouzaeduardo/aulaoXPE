
# Import necessary libraries and modules
import os
import openai
from typing import Dict, List, Any
from dotenv import load_dotenv
from langchain import LLMChain, PromptTemplate
from langchain.llms import BaseLLM
from pydantic import BaseModel, Field
# Import necessary modules
from langchain.chains.base import Chain
from langchain.chat_models import AzureChatOpenAI
from flask_cors import CORS, cross_origin
from StageAnalyzerChain import StageAnalyzerChain
from SalesConversationChain import SalesConversationChain
from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for, jsonify)
from flask_ngrok import run_with_ngrok
from SalesGPT import SalesGPT

# Set up Flask app
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS) for API endpoints
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load environment variables from .env file
load_dotenv()

# Enable CORS support for credentials
CORS(app, support_credentials=True)

# Set up Flask app to run with ngrok
run_with_ngrok(app)

# Set API key for OpenAI
API_KEY = os.getenv("API_KEY", "")
assert API_KEY, "ERROR: Azure OpenAI Key is missing"
openai.api_key =  API_KEY

# Set up OpenAI API endpoint
RESOURCE_ENDPOINT = os.getenv('RESOURCE_ENDPOINT')
assert RESOURCE_ENDPOINT, "ERROR: Azure OpenAI Endpoint is missing"
assert "openai.azure.com" in RESOURCE_ENDPOINT.lower(), "ERROR: Azure OpenAI Endpoint should be in the form: \n\n\t<your unique endpoint identifier>.openai.azure.com"
openai.api_base = RESOURCE_ENDPOINT

# Set up deployment name
DEPLOYMENT_NAME=os.getenv('DEPLOYMENT_NAME')
llm = ''

# Set up AzureChatOpenAI model
llm = AzureChatOpenAI(
    openai_api_base=RESOURCE_ENDPOINT,
    openai_api_version="2023-03-15-preview",
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=API_KEY,
    openai_api_type = "azure",
    temperature = 0.85,verbose=True
)

# Set up StageAnalyzerChain
stage_analyzer_chain = StageAnalyzerChain.from_llm(llm)
stage_analyzer_chain.run(conversation_history='')

# Set up conversation stages
conversation_stages = {'1': "1. Introdução: Começe a conversa se apresentando e apresentando sua empresa. Seja educado e respeitoso, mas mantenha o tom profissional para a conversa.",
                    '2': "2. Qualificação: Qualifique o prospect, confirmando se eles são a pessoa correta para falar sobre o seu produto/serviço. Garanta que eles tem a autoridade de tomada de decisões de compras.",
                    '3': "3. Proposição de valor: Explique de forma rápida como seu produto/serviço pode beneficiar o prospect. Foque nos pontos de vendas únicos e na proposição de valor do seu produto/serviço que diferencia ele da competição.",
                    '4': "4. Necessita análise: Pergunte questões abertas para descobrir as necessidades do prospect. Escute cuidadosamente para sua resposta e tome notas.",
                    '5': "5. Apresentação de Solução: Baseado nas necessidades do prospect, apresente seu produto ou serviço como a solução que resolve seus problemas.",
                    '6': "6 .Lidar com objeções: Endereçe quaisquer objeções que o prospect possa ter sobre o seu produto/serviço. Esteja preparado para providenciar evidências ou testemunhos para suportar seu discurso.",
                    '7': "7. Fechamento: Tentar avançar a venda propondo um próximo passo. Isso pode ser uma demo, uma prova de conceito, uma reunião com os decisores ou outras sugestões. Garante que você sumarizou o que foi discutido e reforçe os benefícios."}

# Set up SalesConversationChain
sales_conversation_utterance_chain = SalesConversationChain.from_llm(llm)

# Set up environment variables
company_business=os.getenv("COMPANY_BUSINESS")
conversation_purpose = os.getenv("CONVERSATION_PURPOSE")
salesperson_name = os.getenv("SALESPERSON_NAME")
salesperson_role= os.getenv("SALESPERSON_ROLE")
company_name= os.getenv("COMPANY_NAME")
company_values = os.getenv("COMPANY_VALUES")

# Run SalesConversationChain
sales_conversation_utterance_chain.run(
    salesperson_name = salesperson_name,
    salesperson_role= salesperson_role,
    company_name=company_name,
    company_business=company_business,
    company_values = company_values,
    conversation_purpose = conversation_purpose,
    conversation_history='Olá, eu sou Seu Agente, da ##Empresa##. Como você está hoje? <END_OF_TURN>\nUser: Estou bem, como você está?<END_OF_TURN>',
    conversation_type="ligação",
    conversation_stage = conversation_stages.get('1', "1. Introdução: Começe a conversa se apresentando e apresentando sua empresa. Seja educado e respeitoso, mas mantenha o tom profissional para a conversa.")
)

# Set up API endpoint for POST requests
@app.route("/api/do", methods=["POST"])
def do_something():
    result = request.get_json()
    return jsonify(data=result["teste"])

# Set up API endpoint for new agent
@app.route("/api/newagent")
def newAgent():
    sales_agent = SalesGPT.from_llm(llm, verbose=False)
    sales_agent.seed_agent()
    returnMessageStep = sales_agent.determine_conversation_stage()
    returnMessage = sales_agent.step()
    sales_agent = SalesGPT.from_llm(llm, verbose=False)    
    
    # init sales agent
    sales_agent.seed_agent()
    result = jsonify(
        conversationStep = sales_agent.determine_conversation_stage(),
        salesAnswer = ''.join(sales_agent.step())
    )

    return result

# Set up API endpoint for conversation step
@app.route("/api/conversation/", methods=["POST"])
@cross_origin(supports_credentials=True)
def conversationStep():
  message = request.get_json()
  sales_agent = SalesGPT.from_llm(llm, verbose=False)
  sales_agent.human_step(str(message["text"]))
  result = jsonify(
        conversationStep = sales_agent.determine_conversation_stage(),
        salesAnswer = ''.join(sales_agent.step())
    )
  return result   
 
    
if __name__ == '__main__':
    app.run()
