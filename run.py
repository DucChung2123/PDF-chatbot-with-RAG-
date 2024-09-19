from app import create_app
from config import PORT
import os
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import login

app = create_app()

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    login(token=os.environ.get("HUGGINGFACE_API_KEY"))
    app.run(port=PORT ,debug=True)
