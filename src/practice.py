from src.predict import predict_one

sample_input = {
    "grid": 1,
    "driverId": 1,
    "constructorId": 131,
    "year": 2018,
    "round": 24,
    "circuitId": 24,
    "form_last3": 25,
    "age": 28,
    "team_strength": 20
}

print(predict_one(sample_input))

