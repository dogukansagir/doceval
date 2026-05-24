import os
import boto3
import json
from botocore.exceptions import ClientError
from dotenv import load_dotenv

AWS_REGION = "eu-north-1"
load_dotenv()
def get_secrets():
    client = boto3.client("secretsmanager", region_name=AWS_REGION)
    try:
        response = client.get_secret_value(SecretId="doceval/secrets")
        return json.loads(response["SecretString"])
    except ClientError:
        return {}

secrets = get_secrets()

DEEPSEEK_API_KEY = secrets.get("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
GEMINI_API_KEY = secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
QDRANT_API_KEY = secrets.get("QDRANT_API_KEY") or os.getenv("QDRANT_API_KEY")
QDRANT_URL = secrets.get("QDRANT_URL") or os.getenv("QDRANT_URL")

CROSSENCODER_KIN = 30
CROSSENCODER_KOUT = 5
DEEPSEEK_FAST_MODEL = "deepseek-v4-flash"
DEEPSEEK_GOOD_MODEL = "deepseek-v4-pro"
GEMINI_MODEL = "gemini-3.1-flash-lite"
DEEPSEEK_API_URL = "https://api.deepseek.com"
QDRANT_COLLECTION_NAME = "doceval"
S3_BUCKET_NAME = "doceval-pdf-storage"
BM25_WEIGHT = 1.0
COSINE_WEIGHT = 1.5
RETRY_COUNT = 3
CONTEXT_PRECISION_RETRY_COUNT = 3 # retries with >10 won't change the weights anymore, so no need to go further than that
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
FAITHFULNESS_THRESHOLD = 0.70
CONTEXT_PRECISION_THRESHOLD = 0.80
ANSWER_RELEVANCY_THRESHOLD = 0.70
ANSWER_CORRECTNESS_THRESHOLD = 0.70