import csv
import json

# Time, Wind Dir, Wind Speed, Particles

wind = input("Enter the URL to the wind csv")
particles = input("Enter the URL to the particles csv")

windARR = []
parARR = []

output = {}

print("Starting")
index = 0
with open(wind, mode="r") as file:
    windFile = csv.reader(file)
    for lines in windFile:
        windARR.append(lines)
        print(f"Wind: {index}")
        index += 1
index = 0

with open(particles, mode="r") as file:
    particlesFILES = csv.reader(file)
    for lines in particlesFILES:
        parARR.append(lines)
        print(f"Par {index}")
        index += 1

index = 0
for line in windARR:
    if len(parARR) <= index or len(windARR) <= index:
        pass
    else:
        output[windARR[index][0][:13]] = {
            "Wind Speed": windARR[index][1],
            "Wind Dir": windARR[index][2],
            "PM1": parARR[index][1],
            "PM10": parARR[index][3],
            "PM25": parARR[index][2],
        }
    index += 1

with open("output.json", "w") as outfile:
    json.dump(output, outfile)
