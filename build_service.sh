#!/bin/bash

# Use cp to copy the file to /etc/systemd/system/
sudo cp wf-oe-f-peak.service /etc/systemd/system/

# Use systemctl to enable the service. It will start on boot.
sudo systemctl enable wf-oe-f-peak.service

# Use systemctl to start the service immediately
sudo systemctl start wf-oe-f-peak.service