import json
import time
 
def open_claims():
    return {
        "response": [  {
                    "claimId": "claim-123",
                    "policyHolderId": "A134085",
                    "claimStatus": "Open"
                },
                {
                    "claimId": "claim-06",
                    "policyHolderId": "A645987",
                    "claimStatus": "Open"
                }
            ]   
        }

def send_reminders():
    return {
                "response": {
                    "sendReminderTrackingId": "50e8400-e29b-41d4-a716-446655440000",
                    "sendReminderStatus": "InProgress"
                }
            }

def lambda_handler(event, context):
    api_path = event['apiPath']
  
    if api_path == '/claims':
        body = open_claims() 
    elif api_path == '/send-reminders':
        body =  send_reminders()
   
    response_body = {
        'application/json': {
        'body': str(body)
        }
    }

    action_response = {
        'actionGroup': event['actionGroup'],
        'apiPath': event['apiPath'],
        'httpMethod': event['httpMethod'],
        'httpStatusCode': 200,
        'responseBody': response_body
    }

    api_response = {
        'messageVersion': '1.0', 
        'response': action_response}
    return api_response
