import time
from firebase import firebase


firebase = firebase.FirebaseApplication('https://ship-detection.firebaseio.com/', None)

def location():
    while True:
        with open('location.csv', 'r') as reader:
            for line in reader:
                latitude = line.strip().split(',')[0]
                longitude = line.strip().split(',')[1]

                firebase.patch('/location', {'latitude': latitude})
                firebase.patch('/location', {'longitude': longitude})

                print(latitude)
                print(longitude)

                time.sleep(5)

location()

