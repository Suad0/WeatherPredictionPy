import mysql.connector
from datetime import datetime, timedelta
import random


class WetterDatenbank:
    def __init__(self, host, user, password, database):
        self.connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        self.cursor = self.connection.cursor()

    def insert_beispieldaten(self, stadt, anzahl_tage=10):
        for _ in range(anzahl_tage):
            temperatur = round(random.uniform(15, 30), 2)
            luftfeuchtigkeit = round(random.uniform(40, 80), 2)
            windgeschwindigkeit = round(random.uniform(0, 20), 2)
            niederschlag = round(random.uniform(0, 10), 2)
            datum = datetime(2023, 9, 1) + timedelta(days=_)

            insert_query = "INSERT INTO wetterdaten (stadt, temperatur, luftfeuchtigkeit, windgeschwindigkeit, niederschlag, datum) VALUES (%s, %s, %s, %s, %s, %s)"
            data = (stadt, temperatur, luftfeuchtigkeit, windgeschwindigkeit, niederschlag, datum)

            self.cursor.execute(insert_query, data)

        self.connection.commit()

    def get_daten(self):
        query = "SELECT * FROM WETTERDATEN"

        self.cursor.execute(query)
        self.connection.commit()

    def close(self):
        self.cursor.close()
        self.connection.close()


if __name__ == "__main__":
    # Verbindungsparameter
    host = "localhost"
    user = "root"
    password = "root"
    database = "weather"

    # Stadt und Anzahl der Tage für Beispieldaten
    stadt = "Wien"
    anzahl_tage = 20

    # Erstelle eine Instanz der WetterDatenbank-Klasse
    wetter_db = WetterDatenbank(host, user, password, database)

    # Füge Beispieldaten in die Datenbank ein
    wetter_db.insert_beispieldaten(stadt, anzahl_tage)

    # Schließe die Verbindung zur Datenbank
    wetter_db.close()

    print("Beispieldaten wurden erfolgreich in die Datenbank eingefügt.")
