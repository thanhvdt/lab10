from fastapi import FastAPI

# The file where NeuralSearcher is stored
from load_vistral import VistralChat

app = FastAPI()

# Create a neural searcher instance
chatbot = VistralChat()


@app.get("/api/v1/chat")
def search_startup(question: str):
    return {"result": chatbot.conversation(text=question)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)