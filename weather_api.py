import requests
import matplotlib.pyplot as plt


class WeatherApi:

    def __init__(self, stadt, format='%C+%t'):
        self.stadt = stadt
        self.format = format
        self.base_url = f'https://wttr.in/{self.stadt}?format={self.format}'

    def get_weather(self):
        response = requests.get(self.base_url)
        if response.status_code == 200:
            wetterdaten = response.text
            return f'Wetter in {self.stadt}: {wetterdaten}'
        else:
            return 'Fehler beim Abrufen der Wetterdaten.'

