from flask import Flask

app = Flask(__name__) # create app instance

# https://stackoverflow.com/a/50361552/10012842
# def create_app(config_class=Config):
#     app = Flask(__name__)
#     # app.config.from_object(config_class)
#     # db.init_app(app)
#     # migrate.init_app(app, db)
#     # ... register blueprints, configure logging etc.
#     with app.app_context():
#         from . import routes
#     return app

import web_app.routes # import webapp routes