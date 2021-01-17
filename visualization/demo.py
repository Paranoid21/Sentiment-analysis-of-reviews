#
# нажимаем alt+shif+f10, далее выбираем demo, в Run появится сообщение "Preparing classifier", ждем несколько
# секунд, у вас появится ссылка скорее всего формата "http://127.0.0.1:5000/ ", нажимаем на нее и можем играться
# с классификатором
#
from sentiment_classifier import SentimentClassifier
from flask import Flask, render_template, request
import time
app = Flask(__name__)

print("Preparing classifier")
start_time = time.time()
classifier = SentimentClassifier()
print("Classifier is ready")
print(time.time() - start_time, "seconds")


@app.route("/", methods=["POST", "GET"])
def index_page(text="", prediction_message=""):

    if request.method == "POST":
        if request.form['submit'] == 'estimate':
            text = request.form["text"]
            prediction_message = classifier.get_prediction_message(text)
            print(prediction_message)
        if request.form['submit'] == 'clear':
            text = ""
            prediction_message = ""

    return render_template('hello.html', text=text, prediction_message=prediction_message)



if __name__ == "__main__":
    app.run(debug=True)
