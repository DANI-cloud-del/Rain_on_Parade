from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello from Rain_on_Parade!</p>"

if __name__ == "__main__":
    app.run(debug=True)
