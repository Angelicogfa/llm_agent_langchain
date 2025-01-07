from typing import List, Callable, Optional
from langgraph.graph import StateGraph
from langchain_core.documents import Document
from langgraph.graph import START, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.vectorstores.base import VectorStore
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage

class State(MessagesState):
    question: str
    query: str
    context: List[Document]
    answer: str

class LLMService:
    def __init__(self, api_key: str, vector_data_base: Callable[[], VectorStore]):
        self.__api_key = api_key
        self.__vector_database = vector_data_base

    def build(self, thread_id: str, callbacks: Optional[List[BaseCallbackHandler]] = None):
        workflow = StateGraph(state_schema=State)
        config = { 'configurable': { 'thread_id': thread_id }}
        
        model = ChatOpenAI(openai_api_key = self.__api_key, streaming = True, callbacks=callbacks, model='gpt-4o', temperature=0.2, top_p=0.4)

        def retrieve(state: State):
            vector_store = self.__vector_database()
            context = vector_store.similarity_search(state["question"], k=5) if state['question'] != '[START]' else []
            return { 'context': context }
        
        def prepare_query(state: State):
            question = state['question']
            context = state['context']

            context_query = str.join('\n\n', [document.page_content for document in context])
            query = context_query + '\n\nQuestion: ' + question
            return {'query': query }

        def call_model(state: State):
            thread = state['messages'].copy()
            thread.append(HumanMessage(content=state['query']))
            response = model.invoke(thread, config=config)
            state['messages'].append(HumanMessage(content=state['question']))
            return {"answer": response, "messages": state['messages'] }
        
        workflow.add_edge(START, "retrieve")
        workflow.add_sequence([retrieve, prepare_query, call_model])

        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
        return app, config
