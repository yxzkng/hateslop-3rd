!pip install -r requirements.txt

import openai
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI()
MODEL = "gpt-4o-mini"
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

def get_today_weather(city: str) -> str:
    "지정된 도시의 현재 날씨, 온도 등의 데이터를 가져옵니다. 온도는 섭씨로 제공됩니다."

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric&lang=kr"
    try:
        response = requests.get(url)
        response.raise_for_status() #
        data = response.json()

        if data.get("cod") == "404":
            return json.dumps({"error": f"도시 '{city}'의 날씨 데이터를 찾을 수 없습니다."})

        weather = data["weather"][0]["description"]
        temperature = data["main"]["temp"] 

        return json.dumps({
            "city": city,
            "weather": weather,
            "temperature_celsius": temperature
        })

    except requests.exceptions.HTTPError as e:
        return json.dumps({"error": f"API 요청 실패. 상태 코드: {response.status_code}, 상세: {e}"})
    except Exception as e:
        return json.dumps({"error": f"API 요청 중 예상치 못한 에러 발생: {e}"})

weather_tool_schema = {
    "type": "function", 
    "function" : 	{
        "name" : "get_today_weather", 
        "description" : "특정 도시의 날씨, 온도와 같은 기상 정보를 가져옵니다.",
        "parameters" : {
            "type" : "object",
            "properties" : {"city" : {"type" : "string", "description" : "기상 정보를 가져올 도시명(예: Seoul, Tokyo)"}},
            "required" : ["city"]
        }
    }
}

available_functions = {
    "get_today_weather" : get_today_weather
}

messages = [{"role" : "user", "content": "오늘 서울 날씨 어때?"}]
tools = [weather_tool_schema]

print("1단계: LLM에 Tool 호출 요청")
response = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
response_message = response.choices[0].message
messages.append(response_message)

tool_calls = response_message.tool_calls
if tool_calls:
    print(" LLM이 Tool 사용을 결정했습니다.")
    print("\n 2단계: 결정된 Tool을 실제로 실행")
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)
        print(f"- 실행 함수: {function_name}({', '.join([f'{k}={v}' for k, v in function_args.items()])})")

        function_response = function_to_call(**function_args)
        print(f"- 실행 결과: {function_response}")

        messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": function_response})

    print("\n 3단계: Tool 실행 결과를 바탕으로 최종 답변 생성")
    final_response = client.chat.completions.create(model=MODEL, messages=messages)
    print("\n===== Agent의 최종 답변 =====")
    agent_answer = final_response.choices[0].message.content
    print(agent_answer)
else:
    print(" LLM이 Tool 사용을 결정하지 않았습니다.")
    final_response = client.chat.completions.create(model=MODEL, messages=messages)
    print("\n===== Agent의 최종 답변 =====")
    print(final_response.choices[0].message.content)



