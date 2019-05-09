# checking user access with Docker
echo "sudo usermod -a -G docker $USER"
sudo usermod -a -G docker $USER
echo ""
echo ""
echo 'grep /etc/group -e "docker"'
grep /etc/group -e "docker"
echo ""
echo ""
echo 'grep /etc/group -e "sudo"'
grep /etc/group -e "sudo"
echo ""
echo ""
echo "!!!!! Don't forget to reconnect to the VM to have the changes activated !!!!!!"
echo ""
echo ""

# installing SDK is not installed
curl https://sdk.cloud.google.com | bash
cd ~/ && \
wget https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-245.0.0-linux-x86_64.tar.gz && \
tar xvzf google-cloud-sdk-245.0.0-linux-x86_64.tar.gz && \
rm google-cloud-sdk-245.0.0-linux-x86_64.tar.gz && \
source /google-cloud-sdk/install.sh
echo ""
echo ""
echo "!!!!! Don't forget to reconnect to the VM to have the changes activated !!!!!!"
echo ""
echo ""
