import os
from os.path import join, dirname, exists
from dotenv import load_dotenv
from cfenv import AppEnv
import json
 
class AppConfig:
    def __init__(self):
        dotenv_path = join(dirname(__file__), '.env')
        if exists(dotenv_path):
            load_dotenv(dotenv_path)
        else:
            print(f"Warning: .env file not found at {dotenv_path}")
 
        self.print_env()
 
        self.LOCAL_ENV = os.getenv("ENV", "PROD").upper() == "LOCAL"
 
        if self.LOCAL_ENV:
            self.load_local_env()
        else:
            self.load_production_env()
 
    def load_local_env(self):
        self.load_common_env()
        # self.load_external_env()
        # POSTGRES Details for local environment
        # self.DB_CONN_URL = self.get_env_var("DB_CONN_URL")
        self.SAP_PROVIDER_URL = self.get_env_var("SAP_PROVIDER_URL")+ "/oauth/token"
        self.SAP_CLIENT_ID = self.get_env_var("SAP_CLIENT_ID")
        self.SAP_CLIENT_SECRET = self.get_env_var("SAP_CLIENT_SECRET")
        self.SAP_ENDPOINT_URL_GPT4O = self.get_env_var("SAP_ENDPOINT_URL_GPT4O")+f"/v2/inference/deployments/{self.get_env_var('AZURE_DEPLOYMENT_ID_4O')}/chat/completions?api-version={self.SAP_API_VERSION}"
 
    def load_production_env(self):
        env = AppEnv()
        self.load_common_env()
 
        postgresql = env.get_service(name=self.get_env_var('POSTGRES_SERVICE_NAME', 'postgresql'))
        if postgresql:
            self.DB_USER = postgresql.credentials["username"]
            self.DB_PWD = postgresql.credentials["password"]
            self.DB_URL = postgresql.credentials["hostname"]
            self.DB_PORT = postgresql.credentials["port"]
            self.DB_NAME = postgresql.credentials["dbname"]
            self.DB_CONN_URL = f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PWD}@{self.DB_URL}:{self.DB_PORT}/{self.DB_NAME}"
        else:
            raise ValueError("PostgreSQL service not found. Please check your environment configuration.")
       
        hana = env.get_service(name=self.get_env_var('HANA_SERVICE_NAME', 'hana'))
        if hana:
            self.HDB_USER = hana.credentials["user"]
            self.HDB_PASSWORD = hana.credentials["password"]
            self.HDB_HOST = hana.credentials["host"]
            self.HDB_PORT = hana.credentials["port"]
        else:
            raise ValueError("HANA service not found. Please check your environment configuration.")
 
 
        genai = env.get_service(name=self.get_env_var('AICORE_SERVICE_NAME', 'aicore'))
        if genai:
            self.SAP_PROVIDER_URL = f"{genai.credentials['url']}/oauth/token"
            self.SAP_CLIENT_ID = genai.credentials["clientid"]
            self.SAP_CLIENT_SECRET = genai.credentials["clientsecret"]
            self.SAP_ENDPOINT_URL_GPT35 = f"{genai.credentials['serviceurls']['AI_API_URL']}/v2/inference/deployments/{self.get_env_var('AZURE_DEPLOYMENT_ID')}/chat/completions?api-version={self.SAP_API_VERSION}"
            self.SAP_ENDPOINT_URL_GPT4O = f"{genai.credentials['serviceurls']['AI_API_URL']}/v2/inference/deployments/{self.get_env_var('AZURE_DEPLOYMENT_ID_4O')}/chat/completions?api-version={self.SAP_API_VERSION}"
            self.SAP_EMBEDDING_ENDPOINT_URL = f"{genai.credentials['serviceurls']['AI_API_URL']}/v2/inference/deployments/{self.get_env_var('AZURE_EMBEDDING_DEPLOYMENT_ID')}/embeddings?api-version={self.SAP_API_VERSION}"
        else:
            raise ValueError("AI Core service not found. Please check your environment configuration.")
 
    def load_common_env(self):
        # Common SAP GPT Details
        # self.SAP_API_VERSION= self.get_env_var('SAP_API_VERSION')
        self.MODEL= self.get_env_var('MODEL') 
        self.TEMPERATURE= float(self.get_env_var('TEMPERATURE')) 
        self.MAX_TOKENS= int(self.get_env_var('MAX_TOKENS')) 
        self.SAP_API_VERSION = self.get_env_var("API_VERSION", "2023-05-15")
        self.LEEWAY = int(self.get_env_var("LEEWAY", 100))
        
        # self.CHAT_PROMPT_TEMPLATE = self.get_env_var('CHAT_PROMPT_TEMPLATE')
    
    
    # def load_external_env(self):
    #     self.SERP_API_KEY= self.get_env_var('SERP_API_KEY') 
    #     # self.TWITTER_CONSUMER_KEY=self.get_env_var('TWITTER_CONSUMER_KEY')
    #     # self.TWITTER_CONSUMER_SECRET=self.get_env_var('TWITTER_CONSUMER_SECRET')
    #     # self.TWITTER_ACCESS_TOKEN=self.get_env_var('TWITTER_ACCESS_TOKEN')
    #     # self.TWITTER_ACCESS_TOKEN_SECRET=self.get_env_var('TWITTER_ACCESS_TOKEN_SECRET')


    def get_env_var(self, key, default=None):
        value = os.getenv(key, default)
        if value is None:
            raise ValueError(f"Missing required environment variable: {key}")
        return value
 
    def to_json(self):
        data = self.__dict__.copy()
        return json.dumps(data, indent=4)

    
    def print_env(self):
        for key, value in os.environ.items():
            print(f"{key}={value}")