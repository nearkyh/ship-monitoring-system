# Import Ship_Detector
from ship_detector import get_frame

# Import Flask
from flask import Flask, render_template, Response, session, redirect, url_for, flash, abort, request
from flask_bootstrap import Bootstrap
from flask_moment import Moment


app = Flask(__name__)
# Create dummy secrey key so we can use sessions
app.config['SECRET_KEY'] = 'yonghan1234'

# manager = Manager(app)
bootstrap = Bootstrap(app)
moment = Moment(app)

@app.errorhandler(404)
def page_not_found(e):
  return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
  return render_template('500.html'), 500

@app.route('/', methods=['GET', 'POST'])
def index():
  return render_template('index.html')

@app.route('/login', methods=['GET'])
def login():
  if request.form['password'] == 'password' and request.form['username'] == 'admin':
    session['logged_in'] = True
  else:
    flash('Login Error')
  return index()

@app.route('/logout', methods=['GET'])
def logout():
  session.clear()
  return index()

@app.route('/detector', methods=['GET'])
def detector():
  return render_template('detector.html')

@app.route('/location', methods=['GET'])
def location():
  return render_template('location.html')

@app.route('/streaming')
def streaming():
  return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
  app.run(host='192.168.0.33', debug=True, threaded=True)

