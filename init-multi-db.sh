#!/bin/bash
set -e

# POSTGRES_MULTIPLE_DATABASES 환경변수(쉼표구분)에 나열된 DB를 생성
if [ -n "$POSTGRES_MULTIPLE_DATABASES" ]; then
  echo "Creating multiple databases: $POSTGRES_MULTIPLE_DATABASES"
  IFS=',' read -ra DBS <<< "$POSTGRES_MULTIPLE_DATABASES"
  for db in "${DBS[@]}"; do
    if [ "$db" != "airflow" ]; then
      psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
        SELECT 'CREATE DATABASE $db'
        WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$db')\gexec
EOSQL
    fi
  done
fi
