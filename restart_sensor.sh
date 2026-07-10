#!/bin/bash

CONFIG_FILE="/etc/default/smart_sensor"

echo "Current configuration:"
cat $CONFIG_FILE

echo
read -p "MQTT Broker IP: " BROKER
read -p "Source (0,1,video.mp4,...): " SOURCE
read -p "Sensor ID: " SENSOR
read -p "Period (minutes): " PERIOD

sudo tee $CONFIG_FILE > /dev/null <<EOF
SOURCE=$SOURCE
PERIOD_MINS=$PERIOD
SENSOR_ID=$SENSOR
MQTT_BROKER_HOST=$BROKER
EOF

echo
echo "Reloading systemd..."

sudo systemctl daemon-reload
sudo systemctl restart smart_sensor.service

echo
echo "Service status:"
systemctl status smart_sensor.service --no-pager