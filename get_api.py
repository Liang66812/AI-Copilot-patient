import requests

#url = "http://0.0.0.0:6667/chat/"  #127.0.0.1:50055
url = "http://127.0.0.1:50055/chat/"

while True:
    print('患者：')
    query = {"text": input()}
    response = requests.post(url, json=query)

    if response.status_code == 200:
        result = response.json()
        print("照护师:", result["result"])
    else:
        print("Error:", response.status_code, response.text)






