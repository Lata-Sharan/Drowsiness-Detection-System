from flask import Flask,redirect, url_for,render_template,request
import os
from DDS_Trial import gen_frames

secret_key = str(os.urandom(24))

app = Flask(__name__)
app.config['TESTING'] = True
app.config['FLASK_ENV'] = 'development'
app.config['SECRET_KEY'] = secret_key

@app.route("/",methods=['GET', 'POST'])
def home():
    print(request.method)
    if request.method == 'POST'and request.form.get('Continue') == 'Continue': 
        return render_template("dds_start.html")

    return render_template("index.html")

@app.route('/live_feed',methods=['GET', 'POST'])
def live_feed():
    print(request.method)
    if request.method == 'POST' and request.form.get('Start') == 'Start':
        gen_frames()
        #return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundry=frame')
    return render_template("index.html")

@app.route('/contact', methods=['GET','POST'])
def contact():
    print(request.method)
    if request.method == 'POST' and request.form.get('Start') == 'Start':
        return render_template("dds_start.html")
    return render_template("think_team.html") 


if __name__ == "__main__":
    app.run(debug=True,port=8080)
