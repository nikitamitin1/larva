#!/bin/bash

if [ "$EUID" -ne 0 ]; then
  echo "Please run this script as root (use 'sudo')."
  exit 1
fi

# VARS
REPO_URL="https://github.com/nikitamitin1/larva"  # URL REPO
INSTALL_DIR="/usr/local/larva"
SERVICE_NAME="larva"
BIN_PATH="/usr/local/bin"
ENV_FILE="$INSTALL_DIR/.env"
PAM_FILE="/etc/pam.d/gdm-password"

# INSTALL DEPENDENCIES
install_dependencies() {
    echo "Installing dependencies..."
#    sudo apt update
#    sudo apt install -y python3 python3-pip python3-opencv
    sudo pip3 install pip install opencv-contrib-python
    sudo pip3 install -r "$INSTALL_DIR/requirements.txt"  # simply dependencies from requirements.txt
}

# CLONE REPO
clone_repo() {
    echo "Cloning repository..."
    git clone "$REPO_URL" "$INSTALL_DIR"
}

# INIT .ENV
setup_env() {
    echo "Setting up environment variables..."
    mkdir -p "$INSTALL_DIR"
    touch "$ENV_FILE"
    sudo chmod 777 "$ENV_FILE"

    # .ENV VARS
    echo "IMAGE_PATH=$INSTALL_DIR/faces/" >> "$ENV_FILE"
    echo "FLAG_FILE=/tmp/face_recognized" >> "$ENV_FILE"
    echo "LOG_FILE=$INSTALL_DIR/logfile.log" >> "$ENV_FILE"


    # DEFAULT SETTINGS FOR ALGHORITHM
    echo "CONFIDENCE=80" >> "$ENV_FILE"
    echo "MAX_ATTEMPTS=5" >> "$ENV_FILE"
    echo "SLEEP_INTERVAL=0" >> "$ENV_FILE"
    echo "REQUIRED_MATCHES=1" >> "$ENV_FILE"

    # IMAGES DIR
    mkdir -p "$INSTALL_DIR/faces"
    sudo chmod 666 "$INSTALL_DIR/faces"

    # LOG FILE
    touch "$INSTALL_DIR/logfile.log"
    sudo chmod 666 "$INSTALL_DIR/logfile.log"
}

# COPY EXECUTABLE SCRIPT
install_script() {
    echo "Installing script to $BIN_PATH..."
    sudo cp "$INSTALL_DIR/main_prod.py" "$BIN_PATH/$SERVICE_NAME"
    sudo chmod +x "$BIN_PATH/$SERVICE_NAME"
}

# PAM SETUP (CHECK UP DOESNT WORK)
setup_pam() {
    echo "Configuring PAM to use face recognition..."
    if sudo grep -Fq "auth sufficient pam_exec.so quiet usr/bin/python3 $BIN_PATH/$SERVICE_NAME" "$PAM_FILE"; then
        echo "PAM already configured."
    else
        sudo sed -i "1i auth sufficient pam_exec.so quiet /usr/bin/python3 $BIN_PATH/$SERVICE_NAME" "$PAM_FILE"
        echo "PAM configuration added."
    fi
}

# FACE CONFIGURATION (ONLY FOR ACTIVE CURRENT USER)
configure_face() {
    echo "Configuring user face..."
    USER_ID=$(id -u "$SUDO_USER")

    # we need user-id here to specify user_id which started installation using sudo
    python3 "$INSTALL_DIR/main_prod.py" --configure-face --user-id "$USER_ID"
}

# MAIN FUNC
main() {
    echo "Starting installation of the Larva..."
    echo "NO GUARANTEE OF WORKABILITY. USE AT YOUR OWN RISK!"

    clone_repo
    install_dependencies
    setup_env
    install_script
    setup_pam

    # FACE SETUP
    echo "Service installed successfully. Would you like to configure user face now? (y/n)"
    read -r setup_choice
    if [ "$setup_choice" = "y" ]; then
        echo "100 photos will be taken automatically. You can change them later by running: sudo $SERVICE_NAME --configure-face"
        configure_face
    else
        echo "You can configure face later by running: sudo $SERVICE_NAME --configure-face"
    fi

    echo "Installation and setup complete."
    echo "Now after every suspend you service will try to detect your face and will unlock your system!"
}


main
