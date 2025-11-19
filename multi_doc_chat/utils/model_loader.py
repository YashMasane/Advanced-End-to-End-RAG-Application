import os
import sys
import json
from dotenv import load_dotenv
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exceptions.exception import DocumentPortalException
from multi_doc_chat.utils.config_loader import load_config
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq


class ApiKeyManager:
    REQUIRED_KEYS = ['OPENAI_API_KEY', 'GROQ_API_KEY']  

    def __init__(self):
        self.api_keys = {}
        raw = os.getenv('apikeyliveclass')

        if raw:
            try:
                parse = json.loads(raw)
                if not isinstance(parse, json):
                    raise ValueError("API keys is not a valid JSON object")
                self.api_keys = parse
                log.info("Loaded API key from ECS secrete")
            except Exception as e:
                log.warning("Failed to parse API_KEYS as JSON", error=str(e))

        for key in self.REQUIRED_KEYS:
            if not self.api_keys.get(key):
                env_val = os.getenv(key)
                if env_val:
                    self.api_keys[key] = env_val
                    log.info(f"API_KEY {key} loaded individually from environment variable")

        missing = [k for k in self.REQUIRED_KEYS if not self.api_keys.get(k)]
        if missing:
            log.error("Required API keys are missing", missing_keys=missing)
            raise DocumentPortalException("API keys are missing", sys)

        log.info("API keys loaded", keys={k: v[:6] + "..." for k, v in self.api_keys.items()})


    def get(self, key:str)->str:
        val = self.api_keys.get(key)
        if not val:
            raise KeyError(f"API key for {key} is missing")
        return val

class ModelLoader:

    """Loads embedding models and LLMs based on config and environment."""
    def __init__(self):
        if os.getenv("ENV", "local").lower() != 'production':
            load_dotenv()
            log.info('Running in Local mode: .env loaded')
        else:
            log.info('Running in PRODUCTION mode')

        self.api_key_manager = ApiKeyManager()
        self.config = load_config()

        log.info('YAML config loaded', config_keys=list(self.config.keys()))

    def load_embeddings(self):
        """Return embedding model"""
        try:
            model_name = self.config['embedding_model']['model_name']
            log.info("Loading embedding model", model_name=model_name)
            return  OpenAIEmbeddings(model=model_name, 
                                    api_key=self.api_key_manager.get('OPENAI_API_KEY'))
        except Exception as e:
            log.error('Error loading embedding model', error=str(e))
            raise DocumentPortalException('Failed to load embedding model', sys)


    def load_llm(self):
        """ Load and return configured LLM Model"""
        llm_block = self.config.get('llm')
        llm_provider = os.getenv("LLM_PROVIDER", 'openai')

        if llm_provider not in llm_block:
            log.error(f"LLM provider not found in llm config", provider=llm_provider)
            raise ValueError(f"LLM provider {llm_provider} not found in LLM config")

        llm_config = llm_block[llm_provider]
        provider = llm_config.get('provider')
        model_name = llm_config.get('model_name')
        temperature = llm_config.get('temperature')
        max_tokens = llm_config.get('max_output_tokens')

        log.info('Loading LLM', provider=provider, model=model_name)

        if provider == "openai":
            return ChatOpenAI(
                model=model_name,
                api_key=self.api_key_manager.get("OPENAI_API_KEY"),
                temperature=temperature,
                max_tokens=max_tokens
            )

        elif provider == "groq":
            return ChatGroq(
                model=model_name,
                api_key=self.api_key_mgr.get("GROQ_API_KEY"), #type: ignore
                temperature=temperature,
            )


        else:
            log.error("Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")

if __name__ == "__main__":
    loader = ModelLoader()

    # Test Embedding
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    result = embeddings.embed_query("Hello, how are you?")
    print(f"Embedding Result: {result}")

    # Test LLM
    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")
    result = llm.invoke("Hello, how are you?")
    print(f"LLM Result: {result.content}")





