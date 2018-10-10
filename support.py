import mysql.connector

def configureDatabase():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        db='testnet',
        charset='utf8mb4')

    return mydb