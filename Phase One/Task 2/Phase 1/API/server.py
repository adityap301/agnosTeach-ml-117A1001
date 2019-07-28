from flask import Flask
import routes

ml = Flask("ML _Workshop")
ml.register_blueprint(routes.model_predict)
ml.run(host="0.0.0.0", port=5006)   # 0.0.0.0 - accessed by all ip addresses

