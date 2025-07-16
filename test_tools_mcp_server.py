#fastmcp dev test_tools_mcp_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("StringTools")

@mcp.tool()
def count_words(text: str) -> int:
    """Counts the number of words in a sentence."""
    return len(text.split())

@mcp.tool()
def capitalize_words(text: str) -> str:
    """Capitalizes the first letter of each word in the given string."""
    return text.title()

@mcp.tool()
def get_weather_temperature(city: str) -> str:
    """Get the current weather temperature of the given city in Celsius"""
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    try:
        apikey = '97afb3c84a9f644b9bd5b9586ea5497a'
        query_params = {'q': city, 'appid': apikey, 'units': 'metric'}
        url = 'https://api.openweathermap.org/data/2.5/weather'
        response = requests.get(url, params=query_params, verify=False)
        if 'main' in response.json():
            result = response.json()['main']['temp']
            return str(result)
        else:
            return response.json()['message']
    except Exception as e:
        return f'an error occurred as exception: {str(e)}'


if __name__ == "__main__":
    mcp.run(transport="stdio")
