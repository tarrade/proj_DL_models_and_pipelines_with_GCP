# install Docker
# https://docs.docker.com/install/linux/docker-ce/debian/
sudo apt-get remove docker docker-engine docker.io containerd runc -y

sudo apt-get update -y
sudo apt-get install apt-transport-https ca-certificates curl gnupg2 software-properties-common -y

curl -fsSL https://download.docker.com/linux/debian/gpg | sudo apt-key add -

sudo apt-key fingerprint 0EBFCD88

sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian $(lsb_release -cs) stable"

sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io -y

sudo docker run hello-world


## checking user access with Docker
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
#echo ""
#echo ""
#echo "!!!!! Don't forget to reconnect to the VM to have the changes activated !!!!!!"

#echo ""
#echo ""
# authorisation to be able to write in the container registry  
gcloud auth configure-docker -q

# test
# docker tag hello-world  gcr.io/docker-ml-dl-28571/hello-world
# docker push gcr.io/docker-ml-dl-28571/hello-world

# installing SDK is not installed
#curl https://sdk.cloud.google.com | bash
#cd ~/ && \
#wget https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-245.0.0-linux-x86_64.tar.gz && \
#tar xvzf google-cloud-sdk-245.0.0-linux-x86_64.tar.gz && \
#rm google-cloud-sdk-245.0.0-linux-x86_64.tar.gz && \
#source /google-cloud-sdk/install.sh
#echo ""
#echo ""
#echo "!!!!! Don't forget to reconnect to the VM to have the changes activated !!!!!!"
#echo ""
#echo ""
