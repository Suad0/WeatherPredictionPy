import matplotlib.pyplot as plt
import mysql.connector
import mysql.connector
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.optimizers import Adam


def show_data_chart():
    host = "localhost"
    user = "root"
    password = "root"
    database = "weather"

    try:
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )

        if connection.is_connected():
            print(f"Verbindung zur MySQL-Datenbank {database} erfolgreich hergestellt.")

            query = "SELECT datum, temperatur FROM wetterdaten WHERE stadt = 'Wien' ORDER BY datum"
            cursor = connection.cursor()
            cursor.execute(query)

            daten = cursor.fetchall()

            # Daten in separate Listen umwandeln (Datum und Temperatur)
            daten = list(zip(*daten))
            datum = daten[0]
            temperatur = daten[1]

            plt.figure(figsize=(10, 5))
            plt.plot(datum, temperatur, marker='o', linestyle='-', color='b')
            plt.xlabel('Datum')
            plt.ylabel('Temperatur (°C)')
            plt.title('Temperaturverlauf in Wien')
            plt.xticks(rotation=45)
            plt.grid(True)

            plt.tight_layout()
            plt.show()

    except mysql.connector.Error as error:
        print(f"Fehler beim Verbinden zur MySQL-Datenbank: {error}")

    finally:
        # Verbindung schließen, wenn sie geöffnet ist
        if 'connection' in locals():
            connection.close()
            print("Verbindung zur MySQL-Datenbank geschlossen.")


def lr_scheduler(epoch, lr):
    if epoch % 10 == 0:
        return lr * 0.9
    return lr


def create_model(input_dim):
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    model = create_model(X_train.shape[1])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_schedule = LearningRateScheduler(lr_scheduler)

    history = model.fit(X_train, y_train, epochs=50, batch_size=64,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping, lr_schedule],
                        verbose=2)

    loss = model.evaluate(X_test, y_test)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    print(f'Loss: {loss}')
    print(f'Mean Absolute Error: {mae}')

    # Visualisierung des Lernerfolgs
    plt.figure(figsize=(12, 6))

    # Trainingsverlust
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Trainingsverlust')
    plt.plot(history.history['val_loss'], label='Validierungsverlust')
    plt.xlabel('Epochen')
    plt.ylabel('Verlust')
    plt.legend()
    plt.title('Trainings- und Validierungsverlust')

    # Lernrate
    plt.subplot(1, 2, 2)
    plt.plot(history.history['lr'], label='Lernrate', color='red')
    plt.xlabel('Epochen')
    plt.ylabel('Lernrate')
    plt.legend()
    plt.title('Lernratenplan')

    plt.tight_layout()
    plt.show()


def main():
    # Verbindungsparameter
    host = "localhost"
    user = "root"
    password = "root"
    database = "weather"

    # Verbindung zur Datenbank herstellen
    connection = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )

    # SQL-Abfrage, um die Daten aus der Tabelle abzurufen
    query = "SELECT UNIX_TIMESTAMP(datum) as timestamp, temperatur FROM wetterdaten WHERE stadt = 'Berlin' ORDER BY datum"

    # Daten abrufen und in ein Pandas DataFrame laden
    data = pd.read_sql(query, connection)

    # Schließe die Verbindung zur Datenbank
    connection.close()

    # Aufteilung in Trainings- und Testdaten
    X = data[['timestamp']]
    y = data['temperatur']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Daten normalisieren
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Modell trainieren und bewerten
    train_and_evaluate_model(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
