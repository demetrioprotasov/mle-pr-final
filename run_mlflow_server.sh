set -a
source .env
set +a

export MLFLOW_S3_ENDPOINT_URL=$MLFLOW_S3_ENDPOINT_URL

mlflow server \
  --registry-store-uri postgresql://$DB_DESTINATION_USER:$DB_DESTINATION_PASSWORD@$DB_DESTINATION_HOST:$DB_DESTINATION_PORT/$DB_DESTINATION_NAME \
  --backend-store-uri postgresql://$DB_DESTINATION_USER:$DB_DESTINATION_PASSWORD@$DB_DESTINATION_HOST:$DB_DESTINATION_PORT/$DB_DESTINATION_NAME \
  --default-artifact-root s3://$S3_BUCKET_NAME \
  --no-serve-artifacts \
  --host 0.0.0.0 \
  --port 5000
