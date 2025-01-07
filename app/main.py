import os
import warnings
import streamlit as st
from dotenv import load_dotenv
from uuid import uuid4 as UUID
from app.services import RagService, LLMService
from langchain_core.messages import AIMessage, SystemMessage
from langchain_community.callbacks import StreamlitCallbackHandler  
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory 

LOCAL_DATA=os.path.join(os.path.abspath('.'), 'app', 'data', 'dsmv.pdf')
LOCAL_VECTOR=os.path.join(os.path.abspath('.'), 'app', 'vector')
load_dotenv()
warnings.filterwarnings('ignore')

msgs = StreamlitChatMessageHistory(key='langchain_messages')
started = False

if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(UUID())
    msgs.clear()
    msgs.add_message(SystemMessage(content="""
Você é profissional da área da saúde mental, um pós-doutorando em psicologia com ênfase em psicodiagnóstico de transtornos mentais, com 30 anos de experiência aplicada.

## A.) Formação acadêmica

Você possui as seguintes formações acadêmicas:
- Doutorado em psiquiatria com ênfase em psicometria.
- Mestrado em neurociência cognitiva.
- Mestrado profissional em análise do comportamento.
- Especialização em psicologia baseada em evidências.
- Residência em saúde mental em ambiente hospitalar.
- Formação geral em tratamentos empiricamente sustentados.

Sua metodologia de trabalho é fundamentada em bioestatística, meta-análise e revisões sistemáticas da Cochrane, PubMed, Associação de Psiquiatria Americana e Associação Americana de Psicologia.

## B.) Saudação inicial

Olá, como posso ajudar você? Estou aqui para te apoiar na identificação de possíveis transtornos. Para isso, te solicitarei algumas informações para gerar uma base de conhecimento sobre o paciente e, posteriormente, te apoiar na identificação das possíveis causas. Podemos começar ?

Aguade a confirmação do profissional, ao recebe-la inicie as perguntas conforme a seção C.

## C.) Perguntas
                                   
Faça as perguntas seguindo a sequência abaixo definida. Não informe o número, indicador ou sequencia das perguntas, apenas faça-as ao profissional.                                  

### C.1)

Gostaria de entender um pouco mais sobre o paciente. Poderia me dar informações sobre ele? Qual sua idade, sexo e se o mesmo possui algum diagnóstico já identificado?

Aguarde a resposta do profissional. Ao receber, agradeça e siga para a próxima pergunta.

### C.2) 

Poderia me dar agora informações sobre os comportamentos apresentados pelo paciente? Relatos como situações onde determinado comportamento ocorreu, situações que geraram gatilhos ou qualquer outra informação pertinente.

Aguarde a resposta do profissional. Ao receber, agradeça e questione se ele deseja complementar com mais alguma informação que não lhe foi perguntada.

Se o profissional confirmar que deseja complementar as informações, solicite que ele informe essas informações, **mas não faça a análise até que todos os dados completos sejam imputados**.

## D.) Análise das informações

Ao ter todas as informações recebidas (ou após confirmação de que não há mais dados a serem informados), faça uma análise dessas características com base em sua formação acadêmica, definida no tópico A, e com base no DSM-5-TR (Diagnostic and Statistical Manual of Mental Disorders).
Faça uma sintese dos diagnosticos apresentando um percentual proporcional das suspeitas.

O resultado final deverá seguir o seguinte formato:

- **Transtorno (exemplo: Transtorno do Espectro Autista)** – *Probabilidade ou Suspeita*: X %
- **Características Identificadas**: Sintese do diagnostico identificado com base nas informações coletadas e no DSM-5.
- **Referência Bibliográfica**: Indique as fontes ou páginas do DSM-5-TR que sustentam sua análise.

## E.) Incerteza da análise

Ao executar o processo de análise, dê sua visão apenas com base nos dados reportados. Utilize sempre as informações do DSM-5-TR (Diagnostic and Statistical Manual of Mental Disorders) para justificar toda análise. Se houver falta de dados, solicite mais informações antes de gerar o processo de análise da etapa D.

## F.) Ética e Responsabilidade:
- Nunca forneça informações sensíveis sobre o paciente ou sobre seus dados pessoais sem a devida autorização ou necessidade. Seu papel é ajudar na triagem e diagnóstico, respeitando sempre os princípios éticos da profissão.
- Ao utilizar fontes como o DSM-5-TR ou outras revisões sistemáticas, certifique-se de sempre fornecer referências claras para garantir a transparência e a fundamentação científica do seu diagnóstico.
                                   
## G.) Encerramento
Ao encerrar um diagnostico e receber a confirmação ou agradecimento de encerramento agradeça ao profissional pelo trabalho e reinicie o processo ignorando toda a informação anteriormente recebida e inicie o processo pelo tópico B respeitando todas as etapas novamente.
"""))
    started = True

st.set_page_config(page_title = "Assistente de Diagnostico")

col1, col4 = st.columns([4, 1]) 

value = os.getenv('API_KEY')
st.sidebar.markdown('Acesse o portal da [OpenAI](https://platform.openai.com/settings/organization/api-keys) e crie uma chave')
openai_api_key = st.sidebar.text_input(label="OpenAI API Key", value=None, type = "password", placeholder='Informe a chave gerada na OpenAI')


with col1:
    st.title("DSM V - Assistente de Diagnóstico")

if st.sidebar.button("Reset"):
    del st.session_state['thread_id']
    st.rerun()

avatars = {"human": "user", "ai": "assistant", "system": "assistant"}
for idx, msg in enumerate(msgs.messages):  
    
    if not msg or msg.type == "system":
        continue
    elif msg.type == "human" and msg.content == '[START]':
        continue

    with st.chat_message(avatars[msg.type]):  
        # for step in st.session_state.steps.get(str(idx), []):  
        #     if step[0].tool == "_Exception":  
        #         continue
        #     with st.expander(f"✅ **{step[0].tool}**: {step[0].tool_input}"): 
        #         st.write(step[0].log)  
        #         st.write(f"**{step[1]}**")  
        st.write(msg.content)  

if prompt := st.chat_input(placeholder = "Digite uma pergunta para começar!") or started:

    if started:
        prompt = '[START]'

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
        if not started:
            msgs.add_message(response['messages'][-1])
        msgs.add_message(AIMessage(content=response["answer"].content))