from flask import Flask, request, jsonify
from connect_memory_with_llm import qa_chain  # Make sure qa_chain is importable

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("message", "")
    response = qa_chain.invoke({'query': user_query})
    return jsonify({"reply": response["result"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3001, debug=True)
