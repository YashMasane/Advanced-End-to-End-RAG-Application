import os
import sys
import json
from dotenv import load_dotenv
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exceptions import DocumentPortalException
from langchain_openai import ChatOpenAI
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

        




