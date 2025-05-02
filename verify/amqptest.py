import pika


def check_rabbitmq_connection(
    host="rmq", port=5672, username="user", password="password"
):
    try:
        credentials = pika.PlainCredentials(username, password)
        parameters = pika.ConnectionParameters(
            host=host, port=port, credentials=credentials
        )
        connection = pika.BlockingConnection(parameters)
        connection.close()
        print(f"✅ Connected to RabbitMQ at {host}:{port}")
        return True
    except pika.exceptions.AMQPConnectionError as e:
        print(f"❌ Could not connect to RabbitMQ at {host}:{port} - {e}")
        return False


if __name__ == "__main__":
    check_rabbitmq_connection()
