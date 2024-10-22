import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dfledlfel51'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'mysql://root:krish2705@localhost/marunthagam'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USERNAME = 'srikrish2705guru@gmail.com'
    MAIL_PASSWORD = 'nvxa rycf lwdt yukq'
