from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import HttpResponseError

if __name__ == "__main__":
    ml_client = MLClient(
        DefaultAzureCredential(), "81b5566d-7ccd-47cf-8c6c-fdda0b6fa411", "LittleYounessEdu", "Littleyounes-ML-EDU"
    )

    try:
        ml_client.compute.get("LittleyounessEx1")
    except HttpResponseError as error:
        print("Request failed: {}".format(error.message))
