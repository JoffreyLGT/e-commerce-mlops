#! /usr/bin/env bash

# Install SigNoz containers on Docker standalone.


# Store the script name for logging purpose
me=$(basename "$0")

# Get the attributes
while getopts n: flag
do
    case "${flag}" in
        n) network=${OPTARG};;
    esac
done
# Ensure the network was provided as argument
if [ -z "$network" ]
then
      echo "$me - Error: you must provide -n argument followed by network name."
      exit 1
else
    echo "$me - Containers network is $network"
fi

# Ensure we are in the ressources directory
reldir="$( dirname -- "$0"; )";
cd "$reldir/ressources";

# Clone SigNoz repository and replace docker-compose by one without their test app
echo "$me - Clone signoz repository if not already done"
[ ! -d "signoz" ] && git clone -b main https://github.com/SigNoz/signoz.git

echo "$me - Delete signoz docker-compose"
rm signoz/deploy/docker/clickhouse-setup/docker-compose.yaml

echo "$me - Copy our custom signoz docker-compose"
cp signoz-docker-compose.yml signoz/deploy/docker/clickhouse-setup/docker-compose.yaml

# Ensure the network already existsr
echo "$me - Create network $network if it doesn't exist"
docker network create $network

# Create containers
echo "$me - Go into signoz/deploy/docker/clickhouse-setup/"
cd signoz/deploy/docker/clickhouse-setup/

echo "$me - Create signoz containers without starting them"
docker compose -f "docker-compose.yaml" create
