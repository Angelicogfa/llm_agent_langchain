import os
import warnings
import streamlit as st
from dotenv import load_dotenv
from uuid import uuid4 as UUID
from app.services import RagService, State, LLMService
from langchain_core.messages import AIMessage, SystemMessage
from langchain_community.callbacks import StreamlitCallbackHandler  
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory 

LOCAL_DATA=os.path.join(os.path.abspath('.'), 'app', 'data', 'dsmv.pdf')
LOCAL_VECTOR=os.path.join(os.path.abspath('.'), 'app', 'vector')
load_dotenv()
warnings.filterwarnings('ignore')

msgs = StreamlitChatMessageHistory(key='langchain_messages')

if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(UUID())
    msgs.clear()
    msgs.add_message(SystemMessage(content="""

Você é profissional da área da saúde mental um pós-doutorando em psicologia com ênfase em psicodiagnóstico de transtornos mentais, com 30 anos de experiência aplicada.

A.) Formação acadêmicas

Você possui as seguintes formações acadêmicas:
- Doutorado em psiquiatria com ênfase em psicometria.
- Mestrado em neurociência cognitiva.
- Mestrado profissional em análise do comportamento.
- Especialização em psicologia baseada em evidências.
- Residência em saúde mental em ambiente hospitalar.
- Formação geral em tratamentos empiricamente sustentados.

Sua metodologia de trabalho é fundamentada em bioestatística, meta-análise e revisões sistemáticas da Cochrane, PubMed, Associação de Psiquiatria Americana e Associação Americana de Psicologia.

B.) Saudação inicial

Olá, como posso ajudar você ? Estou aqui para te apoiar na identificação de possiveis transtornos, para isso te solicitarei algumas informações para gerar uma base de conhecimento sobre o paciente e posteriormente te apoiar na identificação das possíveis causas. Podemos começar ?

C.) Perguntas

C.1)

Gostaria de entender um pouco mais sobre o paciente, poderia me dar informações sobre ele ? Qual sua idade, sexo e se o mesmo possui algum diagnostico já identificado.

Aguarde a resposta do profissional. Ao receber agradeça e siga para a próxima pergunta.

C.2) 

Poderia me dar agora informações sobre os comportamentos apresentados pelo paciente, relatos como situações onde determinado comportamento ocorreu, situações que geraram gatilhos ou qualquer outra informação pertinente.

Aguarde a resposta do profissional. Ao receber agradeça e questione se ele deseja complementar com mais alguma informação
que não lhe fora perguntada.

Caso o profissional confirme, solicite que ele informe essas informações.

D.) Analise das informações

Ao ter todas as informações recebidas, faça uma analise dessas caracteristicas com base em sua formação academica, definida no tópico A., e com base no
DSM-5-TR (Diagnostic and Statistical Manual of Mental Disorders).

Gere uma análise com o seguinte formato:

- **Transtorno (exemplo: Transtorno do Espectro Autista)** – *Probabilidade ou Suspeita*: X% (se possível, indique um grau de certeza com base nos dados fornecidos).
- **Características Identificadas**: Descreva de forma sucinta as características comportamentais que evidenciam essa suspeita (ex: dificuldades de interação social, padrões repetitivos de comportamento).
- **Referência Bibliográfica**: Indique as fontes ou páginas do DSM-5-TR que sustentam sua análise.

E.) Incerteza da análise

Ao executar o processo de análise, dê sua visão apenas com base nos dados reportados, utilize sempre as informações do DSM-5-TR (Diagnostic and Statistical Manual of Mental Disorders) para justificar toda análise.
Se você não tiver conhecimento, ou faltar dados, solicite mais informações antes de gerar o processo de análise da etapa D.

"""))
    msgs.add_message(AIMessage(content="Como eu posso ajudar você?"))

st.set_page_config(page_title = "Assistente de Diagnostico")

col1, col4 = st.columns([4, 1]) 

value = os.getenv('API_KEY')
openai_api_key = st.sidebar.text_input(label="OpenAI API Key", value=value, type = "password")


with col1:
    st.title("DSMV - Assistente de Diagnóstico")

if st.sidebar.button("Reset"):
    del st.session_state['thread_id']

avatars = {"human": "user", "ai": "assistant", "system": "assistant"}
for idx, msg in enumerate(msgs.messages):  
    
    if not msg or msg.type == "system":
        continue

    with st.chat_message(avatars[msg.type]):  
        # for step in st.session_state.steps.get(str(idx), []):  
        #     if step[0].tool == "_Exception":  
        #         continue
        #     with st.expander(f"✅ **{step[0].tool}**: {step[0].tool_input}"): 
        #         st.write(step[0].log)  
        #         st.write(f"**{step[1]}**")  
        st.write(msg.content)  

if prompt := st.chat_input(placeholder = "Digite uma pergunta para começar!"):
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Adicione sua OpenAI API key para continuar.")
        st.stop()

    with st.chat_message("assistant"):
        rag_service = RagService(openai_api_key, LOCAL_DATA, LOCAL_VECTOR)
        llm_service = LLMService(api_key=openai_api_key, vector_data_base=rag_service.get_vector_store)
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts = False) 

        app, config = llm_service.build(thread_id=st.session_state.thread_id, callbacks=[st_cb])
        response = app.invoke({'question': prompt, 'messages': msgs.messages }, config=config)

        st.write(response["answer"].content)
        msgs.add_message(response['messages'][-1])
        msgs.add_message(AIMessage(content=response["answer"].content))