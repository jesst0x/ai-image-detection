import json

# Save dictionary into json file
def save_dict_to_json(dir, parameters):
    with open(dir, 'w') as f:
        parameters = {key:float(value) for key, value in parameters.items()}
        json.dump(parameters, f, indent=4)
        
# Appending training log and evaluation result into txt file        
def logging(dir, messages):
    with open(dir, 'a') as f:
        f.write(messages + '\n')
        