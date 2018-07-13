import urllib.request
import json


def get_employees_list():

    with urllib.request.urlopen('https://tekathon18.firebaseio.com/employees.json') as f:
        employees = f.read()
        return json.loads(employees)